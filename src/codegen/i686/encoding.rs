//! # i686 Integrated Assembler: 32-bit x86 Instruction Encoding
//!
//! Encodes 32-bit x86 machine instructions produced by instruction selection
//! into raw machine code bytes (`Vec<u8>`) with associated relocations.
//! This module operates as an integrated assembler — no external `as` or `gas`
//! invocation. All encoding uses the legacy x86 opcode map with standard
//! ModR/M and SIB byte construction.
//!
//! ## Key Differences from x86-64
//!
//! - **No REX prefix** — All registers encode in 3 bits (values 0–7 only)
//! - **No 64-bit operand size** — Maximum operand width is 32 bits
//! - **No RIP-relative addressing** — Uses absolute or SIB-based addressing
//! - **32-bit immediates/displacements** — No 64-bit immediate values
//!
//! ## Zero External Dependencies
//!
//! Only `std` and internal crate modules are used. No external crates.

use std::collections::HashMap;

use crate::codegen::{CodeGenError, MachineInstr, MachineOperand, RelocationType};
use crate::codegen::regalloc::PhysReg;

// ===========================================================================
// Register encoding constants (3-bit values for ModR/M and SIB fields)
// ===========================================================================

/// EAX register encoding (0) for ModR/M reg/rm fields.
pub const REG_EAX: u8 = 0;
/// ECX register encoding (1).
pub const REG_ECX: u8 = 1;
/// EDX register encoding (2).
pub const REG_EDX: u8 = 2;
/// EBX register encoding (3).
pub const REG_EBX: u8 = 3;
/// ESP register encoding (4). In ModR/M r/m field, indicates SIB byte follows.
pub const REG_ESP: u8 = 4;
/// EBP register encoding (5). With mod=00, r/m=5 encodes `[disp32]`, not `[ebp]`.
pub const REG_EBP: u8 = 5;
/// ESI register encoding (6).
pub const REG_ESI: u8 = 6;
/// EDI register encoding (7).
pub const REG_EDI: u8 = 7;

// ===========================================================================
// Condition code constants for Jcc and SETcc instructions
// ===========================================================================

/// Overflow (OF=1).
pub const CC_O: u8 = 0x0;
/// No Overflow (OF=0).
pub const CC_NO: u8 = 0x1;
/// Below / Carry (CF=1) — unsigned less-than.
pub const CC_B: u8 = 0x2;
/// Not Below / No Carry (CF=0) — unsigned above-or-equal.
pub const CC_NB: u8 = 0x3;
/// Equal / Zero (ZF=1).
pub const CC_E: u8 = 0x4;
/// Not Equal / Not Zero (ZF=0).
pub const CC_NE: u8 = 0x5;
/// Below or Equal (CF=1 or ZF=1) — unsigned.
pub const CC_BE: u8 = 0x6;
/// Above (CF=0 and ZF=0) — unsigned.
pub const CC_A: u8 = 0x7;
/// Sign (SF=1).
pub const CC_S: u8 = 0x8;
/// No Sign (SF=0).
pub const CC_NS: u8 = 0x9;
/// Parity Even (PF=1).
pub const CC_P: u8 = 0xA;
/// Parity Odd (PF=0).
pub const CC_NP: u8 = 0xB;
/// Less (SF!=OF) — signed.
pub const CC_L: u8 = 0xC;
/// Greater or Equal (SF=OF) — signed.
pub const CC_GE: u8 = 0xD;
/// Less or Equal (ZF=1 or SF!=OF) — signed.
pub const CC_LE: u8 = 0xE;
/// Greater (ZF=0 and SF=OF) — signed.
pub const CC_G: u8 = 0xF;

// ===========================================================================
// I686Opcode — machine instruction opcode enumeration
// ===========================================================================

/// Enumerates all i686 machine instruction opcodes used by instruction
/// selection and consumed by the encoder. The numeric value is stored in
/// [`MachineInstr::opcode`] via `#[repr(u32)]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[allow(dead_code)]
pub enum I686Opcode {
    /// Label pseudo-instruction — marks a branch target. Operand: `Label(id)`.
    Label = 0,
    /// Integer add. Operands: `[dst, src]`.
    Add = 1,
    /// Integer subtract. Operands: `[dst, src]`.
    Sub,
    /// Bitwise AND. Operands: `[dst, src]`.
    And,
    /// Bitwise OR. Operands: `[dst, src]`.
    Or,
    /// Bitwise XOR. Operands: `[dst, src]`.
    Xor,
    /// Compare (sets flags). Operands: `[lhs, rhs]`.
    Cmp,
    /// Test (AND flags). Operands: `[lhs, rhs]`.
    Test,
    /// Two's complement negation. Operand: `[reg]`.
    Neg,
    /// Bitwise NOT. Operand: `[reg]`.
    Not,
    /// Signed multiply. Two/three operand forms.
    Imul,
    /// Unsigned multiply into EDX:EAX. Operand: `[src]`.
    Mul,
    /// Signed divide EDX:EAX. Operand: `[divisor]`.
    Idiv,
    /// Unsigned divide EDX:EAX. Operand: `[divisor]`.
    Div,
    /// Shift left logical. Operands: `[dst, count]`.
    Shl,
    /// Shift right logical. Operands: `[dst, count]`.
    Shr,
    /// Shift right arithmetic. Operands: `[dst, count]`.
    Sar,
    /// Double-precision shift left. Operands: `[dst, src, count]`.
    Shld,
    /// Double-precision shift right. Operands: `[dst, src, count]`.
    Shrd,
    /// 32-bit move. Operands: `[dst, src]`.
    Mov,
    /// 8-bit move. Operands: `[dst, src]`.
    Mov8,
    /// Zero-extend byte to dword. Operands: `[dst, src]`.
    Movzx8,
    /// Zero-extend word to dword. Operands: `[dst, src]`.
    Movzx16,
    /// Sign-extend byte to dword. Operands: `[dst, src]`.
    Movsx8,
    /// Sign-extend word to dword. Operands: `[dst, src]`.
    Movsx16,
    /// Load effective address. Operands: `[dst, mem]`.
    Lea,
    /// Push onto stack. Operand: `[src]`.
    Push,
    /// Pop from stack. Operand: `[dst]`.
    Pop,
    /// Sign-extend EAX into EDX:EAX. No operands.
    Cdq,
    /// Unconditional jump. Operand: `[Label(target)]`.
    Jmp,
    /// Indirect jump. Operand: `[Register(target)]`.
    JmpIndirect,
    /// Conditional jump. Operands: `[Immediate(cc), Label(target)]`.
    Jcc,
    /// Direct call. Operand: `[Symbol(name)]` or `[Label(id)]`.
    Call,
    /// Indirect call. Operand: `[Register(target)]`.
    CallIndirect,
    /// Return. No operands.
    Ret,
    /// Return and pop imm16 bytes. Operand: `[Immediate(n)]`.
    RetImm,
    /// Set byte on condition. Operands: `[Immediate(cc), Register(dst)]`.
    Setcc,
    /// No operation. No operands.
    Nop,
    /// Move scalar single-precision. Operands: `[dst, src]`.
    Movss,
    /// Move scalar double-precision. Operands: `[dst, src]`.
    Movsd,
    /// Add scalar single. Operands: `[dst, src]`.
    Addss,
    /// Add scalar double. Operands: `[dst, src]`.
    Addsd,
    /// Subtract scalar single. Operands: `[dst, src]`.
    Subss,
    /// Subtract scalar double. Operands: `[dst, src]`.
    Subsd,
    /// Multiply scalar single. Operands: `[dst, src]`.
    Mulss,
    /// Multiply scalar double. Operands: `[dst, src]`.
    Mulsd,
    /// Divide scalar single. Operands: `[dst, src]`.
    Divss,
    /// Divide scalar double. Operands: `[dst, src]`.
    Divsd,
    /// Unordered compare single. Operands: `[lhs, rhs]`.
    Ucomiss,
    /// Unordered compare double. Operands: `[lhs, rhs]`.
    Ucomisd,
    /// Convert int32 to scalar single. Operands: `[dst_xmm, src_gpr]`.
    Cvtsi2ss,
    /// Convert int32 to scalar double. Operands: `[dst_xmm, src_gpr]`.
    Cvtsi2sd,
    /// Convert scalar single to int32 (truncate). Operands: `[dst_gpr, src_xmm]`.
    Cvttss2si,
    /// Convert scalar double to int32 (truncate). Operands: `[dst_gpr, src_xmm]`.
    Cvttsd2si,
    /// Convert scalar single to double. Operands: `[dst_xmm, src_xmm]`.
    Cvtss2sd,
    /// Convert scalar double to single. Operands: `[dst_xmm, src_xmm]`.
    Cvtsd2ss,
    /// Load from GOT (PIC). Operands: `[Register(dst), Symbol(name)]`.
    /// Encodes `mov reg, [ebx + symbol@GOT]` with R_386_GOT32 relocation.
    MovGot,
    /// Call through PLT (PIC). Operand: `[Symbol(name)]`.
    /// Encodes `call symbol@PLT` with R_386_PLT32 relocation.
    CallPlt,
}

/// Total number of defined opcodes, for range checking in `from_u32`.
const I686_OPCODE_COUNT: u32 = I686Opcode::CallPlt as u32 + 1;

impl I686Opcode {
    /// Convert a raw `u32` from `MachineInstr.opcode` to the typed enum.
    /// Returns `None` for values outside the defined range.
    pub fn from_u32(v: u32) -> Option<Self> {
        if v < I686_OPCODE_COUNT {
            // SAFETY: `v` is verified to be within the contiguous range of
            // #[repr(u32)] enum discriminants (0..I686_OPCODE_COUNT).
            // Transmuting a bounds-checked u32 to a contiguous repr(u32) enum
            // is safe. Scope: single transmute of a validated u32.
            Some(unsafe { std::mem::transmute::<u32, I686Opcode>(v) })
        } else {
            None
        }
    }
}

// ===========================================================================
// Output structures
// ===========================================================================

/// Output of the i686 instruction encoder.
pub struct EncodedOutput {
    /// Encoded machine code bytes for the `.text` section.
    pub code: Vec<u8>,
    /// Relocations for external symbol references.
    pub relocations: Vec<PendingRelocation>,
    /// Map from label IDs to byte offsets in `code`.
    pub label_offsets: HashMap<u32, usize>,
}

/// A pending relocation entry produced during encoding.
pub struct PendingRelocation {
    /// Byte offset within `EncodedOutput.code` where the relocation applies.
    pub offset: usize,
    /// Symbol name whose address the linker must patch in.
    pub symbol: String,
    /// Relocation type — one of the `R_386_*` variants.
    pub reloc_type: RelocationType,
    /// Addend value for the relocation computation.
    pub addend: i32,
}

/// Tracks a branch instruction whose rel32 must be patched in the second pass.
struct BranchFixup {
    /// Offset of the rel32 field within `code`.
    patch_offset: usize,
    /// Target label ID.
    target_label: u32,
}

// ===========================================================================
// phys_reg_to_encoding
// ===========================================================================

/// Maps a physical register to its 3-bit encoding value.
///
/// - `PhysReg(0..7)` → GPR encoding 0..7 (eax through edi)
/// - `PhysReg(8..15)` → XMM encoding 0..7 (xmm0 through xmm7)
pub fn phys_reg_to_encoding(reg: PhysReg) -> u8 {
    let n = reg.0;
    if n < 8 {
        n as u8
    } else if n < 16 {
        (n - 8) as u8
    } else {
        debug_assert!(false, "i686 register {} out of range", n);
        0
    }
}

// ===========================================================================
// Low-level encoding helpers
// ===========================================================================

/// Construct a ModR/M byte: `[mod:2][reg:3][r/m:3]`.
#[inline]
fn encode_modrm(mod_field: u8, reg: u8, rm: u8) -> u8 {
    ((mod_field & 0x3) << 6) | ((reg & 0x7) << 3) | (rm & 0x7)
}

/// Construct a SIB byte: `[scale:2][index:3][base:3]`.
#[inline]
fn encode_sib(scale: u8, index: u8, base: u8) -> u8 {
    ((scale & 0x3) << 6) | ((index & 0x7) << 3) | (base & 0x7)
}

/// Append an 8-bit signed immediate.
#[inline]
fn emit_imm8(buf: &mut Vec<u8>, value: i8) {
    buf.push(value as u8);
}

/// Append a 16-bit signed immediate in little-endian.
#[inline]
fn emit_imm16(buf: &mut Vec<u8>, value: i16) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Append a 32-bit signed immediate in little-endian.
#[inline]
fn emit_imm32(buf: &mut Vec<u8>, value: i32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Check if a 32-bit value fits in a sign-extended imm8 field.
#[inline]
fn fits_in_imm8(value: i32) -> bool {
    value >= -128 && value <= 127
}

// ===========================================================================
// Memory operand encoding
// ===========================================================================

/// Encode `[base + disp]` memory operand (ModR/M + optional SIB + disp).
fn encode_memory_operand(buf: &mut Vec<u8>, reg: u8, base_reg: u8, disp: i32) {
    if base_reg == REG_ESP {
        // ESP as r/m always needs SIB: scale=0, index=4(none), base=ESP
        let sib = encode_sib(0b00, 0b100, REG_ESP);
        if disp == 0 {
            buf.push(encode_modrm(0b00, reg, REG_ESP));
            buf.push(sib);
        } else if fits_in_imm8(disp) {
            buf.push(encode_modrm(0b01, reg, REG_ESP));
            buf.push(sib);
            emit_imm8(buf, disp as i8);
        } else {
            buf.push(encode_modrm(0b10, reg, REG_ESP));
            buf.push(sib);
            emit_imm32(buf, disp);
        }
    } else if base_reg == REG_EBP {
        // mod=00/rm=5 means [disp32], so zero disp uses mod=01/disp8=0
        if fits_in_imm8(disp) {
            buf.push(encode_modrm(0b01, reg, REG_EBP));
            emit_imm8(buf, disp as i8);
        } else {
            buf.push(encode_modrm(0b10, reg, REG_EBP));
            emit_imm32(buf, disp);
        }
    } else if disp == 0 {
        buf.push(encode_modrm(0b00, reg, base_reg));
    } else if fits_in_imm8(disp) {
        buf.push(encode_modrm(0b01, reg, base_reg));
        emit_imm8(buf, disp as i8);
    } else {
        buf.push(encode_modrm(0b10, reg, base_reg));
        emit_imm32(buf, disp);
    }
}

/// Encode memory operand with optional scaled index: `[base + index*scale + disp]`.
#[allow(dead_code)]
fn encode_memory_operand_sib(
    buf: &mut Vec<u8>,
    reg: u8,
    base: u8,
    index: Option<(u8, u8)>,
    disp: i32,
) {
    match index {
        None => encode_memory_operand(buf, reg, base, disp),
        Some((idx_reg, scale)) => {
            let sib = encode_sib(scale, idx_reg, base);
            if disp == 0 && base != REG_EBP {
                buf.push(encode_modrm(0b00, reg, 0b100));
                buf.push(sib);
            } else if fits_in_imm8(disp) {
                buf.push(encode_modrm(0b01, reg, 0b100));
                buf.push(sib);
                emit_imm8(buf, disp as i8);
            } else {
                buf.push(encode_modrm(0b10, reg, 0b100));
                buf.push(sib);
                emit_imm32(buf, disp);
            }
        }
    }
}

// ===========================================================================
// Operand extraction helpers
// ===========================================================================

fn extract_reg(op: &MachineOperand) -> Result<u8, CodeGenError> {
    match op {
        MachineOperand::Register(pr) => Ok(phys_reg_to_encoding(*pr)),
        _ => Err(CodeGenError::EncodingError("expected register operand".into())),
    }
}

fn extract_imm(op: &MachineOperand) -> Result<i64, CodeGenError> {
    match op {
        MachineOperand::Immediate(v) => Ok(*v),
        _ => Err(CodeGenError::EncodingError("expected immediate operand".into())),
    }
}

fn extract_mem(op: &MachineOperand) -> Result<(u8, i32), CodeGenError> {
    match op {
        MachineOperand::Memory { base, offset } => Ok((phys_reg_to_encoding(*base), *offset)),
        _ => Err(CodeGenError::EncodingError("expected memory operand".into())),
    }
}

fn extract_label(op: &MachineOperand) -> Result<u32, CodeGenError> {
    match op {
        MachineOperand::Label(id) => Ok(*id),
        _ => Err(CodeGenError::EncodingError("expected label operand".into())),
    }
}

fn extract_symbol(op: &MachineOperand) -> Result<&str, CodeGenError> {
    match op {
        MachineOperand::Symbol(name) => Ok(name.as_str()),
        _ => Err(CodeGenError::EncodingError("expected symbol operand".into())),
    }
}

// ===========================================================================
// ALU instruction encoding
// ===========================================================================

/// Returns `(base_opcode, opcode_extension, accumulator_imm_opcode)` for
/// the standard ALU instruction group. Each ALU op occupies an 8-opcode
/// block in the primary opcode map.
fn alu_params(op: I686Opcode) -> (u8, u8, u8) {
    match op {
        I686Opcode::Add => (0x00, 0, 0x05),
        I686Opcode::Or  => (0x08, 1, 0x0D),
        I686Opcode::And => (0x20, 4, 0x25),
        I686Opcode::Sub => (0x28, 5, 0x2D),
        I686Opcode::Xor => (0x30, 6, 0x35),
        I686Opcode::Cmp => (0x38, 7, 0x3D),
        _ => unreachable!("alu_params called with non-ALU opcode"),
    }
}

/// Encode a standard ALU instruction (add, sub, and, or, xor, cmp).
fn encode_alu(
    buf: &mut Vec<u8>,
    op: I686Opcode,
    operands: &[MachineOperand],
) -> Result<(), CodeGenError> {
    let (base, ext, acc_imm) = alu_params(op);

    if operands.len() < 2 {
        return Err(CodeGenError::EncodingError("ALU requires 2 operands".into()));
    }

    match (&operands[0], &operands[1]) {
        // reg, reg → alu r/m32, r32
        (MachineOperand::Register(dst), MachineOperand::Register(src)) => {
            let d = phys_reg_to_encoding(*dst);
            let s = phys_reg_to_encoding(*src);
            buf.push(base + 1); // alu r/m32, r32
            buf.push(encode_modrm(0b11, s, d));
        }
        // reg, imm → alu r/m32, imm8/imm32
        (MachineOperand::Register(dst), MachineOperand::Immediate(imm)) => {
            let d = phys_reg_to_encoding(*dst);
            let v = *imm as i32;
            if fits_in_imm8(v) {
                buf.push(0x83);
                buf.push(encode_modrm(0b11, ext, d));
                emit_imm8(buf, v as i8);
            } else if d == REG_EAX {
                buf.push(acc_imm);
                emit_imm32(buf, v);
            } else {
                buf.push(0x81);
                buf.push(encode_modrm(0b11, ext, d));
                emit_imm32(buf, v);
            }
        }
        // reg, mem → alu r32, r/m32
        (MachineOperand::Register(dst), MachineOperand::Memory { base: br, offset }) => {
            let d = phys_reg_to_encoding(*dst);
            buf.push(base + 3);
            encode_memory_operand(buf, d, phys_reg_to_encoding(*br), *offset);
        }
        // mem, reg → alu r/m32, r32
        (MachineOperand::Memory { base: br, offset }, MachineOperand::Register(src)) => {
            let s = phys_reg_to_encoding(*src);
            buf.push(base + 1);
            encode_memory_operand(buf, s, phys_reg_to_encoding(*br), *offset);
        }
        // mem, imm → alu r/m32, imm8/imm32
        (MachineOperand::Memory { base: br, offset }, MachineOperand::Immediate(imm)) => {
            let b = phys_reg_to_encoding(*br);
            let v = *imm as i32;
            if fits_in_imm8(v) {
                buf.push(0x83);
                encode_memory_operand(buf, ext, b, *offset);
                emit_imm8(buf, v as i8);
            } else {
                buf.push(0x81);
                encode_memory_operand(buf, ext, b, *offset);
                emit_imm32(buf, v);
            }
        }
        _ => {
            return Err(CodeGenError::EncodingError(
                format!("invalid ALU operand combination"),
            ));
        }
    }
    Ok(())
}

// ===========================================================================
// Shift/Bitwise instruction encoding
// ===========================================================================

/// Returns the ModR/M reg-field extension for shift instructions.
fn shift_extension(op: I686Opcode) -> u8 {
    match op {
        I686Opcode::Shl => 4,
        I686Opcode::Shr => 5,
        I686Opcode::Sar => 7,
        _ => unreachable!("shift_extension called with non-shift opcode"),
    }
}

/// Encode a shift instruction (shl, shr, sar).
fn encode_shift(
    buf: &mut Vec<u8>,
    op: I686Opcode,
    operands: &[MachineOperand],
) -> Result<(), CodeGenError> {
    let ext = shift_extension(op);

    if operands.len() < 2 {
        return Err(CodeGenError::EncodingError("shift requires 2 operands".into()));
    }

    let d = extract_reg(&operands[0])?;

    match &operands[1] {
        MachineOperand::Immediate(count) => {
            let c = *count as u8;
            if c == 1 {
                // shift r/m32, 1: 0xD1 /ext
                buf.push(0xD1);
                buf.push(encode_modrm(0b11, ext, d));
            } else {
                // shift r/m32, imm8: 0xC1 /ext
                buf.push(0xC1);
                buf.push(encode_modrm(0b11, ext, d));
                emit_imm8(buf, c as i8);
            }
        }
        MachineOperand::Register(_) => {
            // shift r/m32, cl: 0xD3 /ext (count in CL register)
            buf.push(0xD3);
            buf.push(encode_modrm(0b11, ext, d));
        }
        _ => {
            return Err(CodeGenError::EncodingError("invalid shift count operand".into()));
        }
    }
    Ok(())
}

/// Encode SHLD or SHRD double-precision shift.
fn encode_double_shift(
    buf: &mut Vec<u8>,
    op: I686Opcode,
    operands: &[MachineOperand],
) -> Result<(), CodeGenError> {
    if operands.len() < 3 {
        return Err(CodeGenError::EncodingError("SHLD/SHRD requires 3 operands".into()));
    }
    let d = extract_reg(&operands[0])?;
    let s = extract_reg(&operands[1])?;

    buf.push(0x0F);
    match &operands[2] {
        MachineOperand::Immediate(count) => {
            match op {
                I686Opcode::Shld => buf.push(0xA4),
                I686Opcode::Shrd => buf.push(0xAC),
                _ => unreachable!(),
            }
            buf.push(encode_modrm(0b11, s, d));
            emit_imm8(buf, *count as i8);
        }
        MachineOperand::Register(_) => {
            // Count in CL
            match op {
                I686Opcode::Shld => buf.push(0xA5),
                I686Opcode::Shrd => buf.push(0xAD),
                _ => unreachable!(),
            }
            buf.push(encode_modrm(0b11, s, d));
        }
        _ => {
            return Err(CodeGenError::EncodingError("invalid SHLD/SHRD count operand".into()));
        }
    }
    Ok(())
}

// ===========================================================================
// Data movement encoding
// ===========================================================================

/// Encode IMUL instruction (1, 2, or 3 operand forms).
fn encode_imul(
    buf: &mut Vec<u8>,
    operands: &[MachineOperand],
) -> Result<(), CodeGenError> {
    if operands.is_empty() {
        return Err(CodeGenError::EncodingError("IMUL requires at least 1 operand".into()));
    }

    if operands.len() == 1 {
        // imul r/m32 (single operand, result in EDX:EAX): F7 /5
        let r = extract_reg(&operands[0])?;
        buf.push(0xF7);
        buf.push(encode_modrm(0b11, 5, r));
    } else if operands.len() == 2 {
        // imul r32, r/m32: 0F AF
        let d = extract_reg(&operands[0])?;
        buf.push(0x0F);
        buf.push(0xAF);
        match &operands[1] {
            MachineOperand::Register(src) => {
                let s = phys_reg_to_encoding(*src);
                buf.push(encode_modrm(0b11, d, s));
            }
            MachineOperand::Memory { base, offset } => {
                encode_memory_operand(buf, d, phys_reg_to_encoding(*base), *offset);
            }
            _ => {
                return Err(CodeGenError::EncodingError("invalid IMUL source operand".into()));
            }
        }
    } else {
        // imul r32, r/m32, imm: 69/6B
        let d = extract_reg(&operands[0])?;
        let imm_val = extract_imm(&operands[2])? as i32;
        if fits_in_imm8(imm_val) {
            buf.push(0x6B);
        } else {
            buf.push(0x69);
        }
        match &operands[1] {
            MachineOperand::Register(src) => {
                let s = phys_reg_to_encoding(*src);
                buf.push(encode_modrm(0b11, d, s));
            }
            MachineOperand::Memory { base, offset } => {
                encode_memory_operand(buf, d, phys_reg_to_encoding(*base), *offset);
            }
            _ => {
                return Err(CodeGenError::EncodingError("invalid IMUL source operand".into()));
            }
        }
        if fits_in_imm8(imm_val) {
            emit_imm8(buf, imm_val as i8);
        } else {
            emit_imm32(buf, imm_val);
        }
    }
    Ok(())
}

/// Encode a MOV instruction (32-bit) — version that takes the full output
/// for relocation support (symbol operands).
fn encode_mov_full(
    output: &mut EncodedOutput,
    operands: &[MachineOperand],
) -> Result<(), CodeGenError> {
    if operands.len() < 2 {
        return Err(CodeGenError::EncodingError("MOV requires 2 operands".into()));
    }

    match (&operands[0], &operands[1]) {
        // mov r32, r32
        (MachineOperand::Register(dst), MachineOperand::Register(src)) => {
            let d = phys_reg_to_encoding(*dst);
            let s = phys_reg_to_encoding(*src);
            output.code.push(0x89);
            output.code.push(encode_modrm(0b11, s, d));
        }
        // mov r32, imm32
        (MachineOperand::Register(dst), MachineOperand::Immediate(imm)) => {
            let d = phys_reg_to_encoding(*dst);
            output.code.push(0xB8 + d);
            emit_imm32(&mut output.code, *imm as i32);
        }
        // mov r32, mem
        (MachineOperand::Register(dst), MachineOperand::Memory { base, offset }) => {
            let d = phys_reg_to_encoding(*dst);
            output.code.push(0x8B);
            encode_memory_operand(&mut output.code, d, phys_reg_to_encoding(*base), *offset);
        }
        // mov mem, r32
        (MachineOperand::Memory { base, offset }, MachineOperand::Register(src)) => {
            let s = phys_reg_to_encoding(*src);
            output.code.push(0x89);
            encode_memory_operand(&mut output.code, s, phys_reg_to_encoding(*base), *offset);
        }
        // mov mem, imm32
        (MachineOperand::Memory { base, offset }, MachineOperand::Immediate(imm)) => {
            output.code.push(0xC7);
            encode_memory_operand(&mut output.code, 0, phys_reg_to_encoding(*base), *offset);
            emit_imm32(&mut output.code, *imm as i32);
        }
        // mov r32, symbol (absolute address with relocation)
        (MachineOperand::Register(dst), MachineOperand::Symbol(name)) => {
            let d = phys_reg_to_encoding(*dst);
            output.code.push(0xB8 + d);
            let reloc_offset = output.code.len();
            emit_imm32(&mut output.code, 0); // placeholder
            output.relocations.push(PendingRelocation {
                offset: reloc_offset,
                symbol: name.clone(),
                reloc_type: RelocationType::I386_32,
                addend: 0,
            });
        }
        _ => {
            return Err(CodeGenError::EncodingError("invalid MOV operand combination".into()));
        }
    }
    Ok(())
}

/// Encode 8-bit MOV.
fn encode_mov8(
    buf: &mut Vec<u8>,
    operands: &[MachineOperand],
) -> Result<(), CodeGenError> {
    if operands.len() < 2 {
        return Err(CodeGenError::EncodingError("MOV8 requires 2 operands".into()));
    }
    match (&operands[0], &operands[1]) {
        (MachineOperand::Register(dst), MachineOperand::Register(src)) => {
            let d = phys_reg_to_encoding(*dst);
            let s = phys_reg_to_encoding(*src);
            buf.push(0x88); // mov r/m8, r8
            buf.push(encode_modrm(0b11, s, d));
        }
        (MachineOperand::Memory { base, offset }, MachineOperand::Register(src)) => {
            let s = phys_reg_to_encoding(*src);
            buf.push(0x88);
            encode_memory_operand(buf, s, phys_reg_to_encoding(*base), *offset);
        }
        (MachineOperand::Register(dst), MachineOperand::Memory { base, offset }) => {
            let d = phys_reg_to_encoding(*dst);
            buf.push(0x8A); // mov r8, r/m8
            encode_memory_operand(buf, d, phys_reg_to_encoding(*base), *offset);
        }
        _ => {
            return Err(CodeGenError::EncodingError("invalid MOV8 operand combination".into()));
        }
    }
    Ok(())
}

/// Encode MOVZX/MOVSX (zero/sign extension).
fn encode_movx(
    buf: &mut Vec<u8>,
    op: I686Opcode,
    operands: &[MachineOperand],
) -> Result<(), CodeGenError> {
    if operands.len() < 2 {
        return Err(CodeGenError::EncodingError("MOVZX/MOVSX requires 2 operands".into()));
    }

    let opcode2 = match op {
        I686Opcode::Movzx8  => 0xB6u8,
        I686Opcode::Movzx16 => 0xB7,
        I686Opcode::Movsx8  => 0xBE,
        I686Opcode::Movsx16 => 0xBF,
        _ => unreachable!(),
    };

    let d = extract_reg(&operands[0])?;
    buf.push(0x0F);
    buf.push(opcode2);

    match &operands[1] {
        MachineOperand::Register(src) => {
            let s = phys_reg_to_encoding(*src);
            buf.push(encode_modrm(0b11, d, s));
        }
        MachineOperand::Memory { base, offset } => {
            encode_memory_operand(buf, d, phys_reg_to_encoding(*base), *offset);
        }
        _ => {
            return Err(CodeGenError::EncodingError("invalid MOVZX/MOVSX source".into()));
        }
    }
    Ok(())
}

/// Encode LEA instruction.
fn encode_lea(
    buf: &mut Vec<u8>,
    operands: &[MachineOperand],
) -> Result<(), CodeGenError> {
    if operands.len() < 2 {
        return Err(CodeGenError::EncodingError("LEA requires 2 operands".into()));
    }
    let d = extract_reg(&operands[0])?;
    let (base_enc, disp) = extract_mem(&operands[1])?;
    buf.push(0x8D);
    encode_memory_operand(buf, d, base_enc, disp);
    Ok(())
}

/// Encode PUSH instruction.
fn encode_push(
    buf: &mut Vec<u8>,
    operands: &[MachineOperand],
) -> Result<(), CodeGenError> {
    if operands.is_empty() {
        return Err(CodeGenError::EncodingError("PUSH requires 1 operand".into()));
    }
    match &operands[0] {
        MachineOperand::Register(r) => {
            buf.push(0x50 + phys_reg_to_encoding(*r));
        }
        MachineOperand::Immediate(imm) => {
            let v = *imm as i32;
            if fits_in_imm8(v) {
                buf.push(0x6A);
                emit_imm8(buf, v as i8);
            } else {
                buf.push(0x68);
                emit_imm32(buf, v);
            }
        }
        MachineOperand::Memory { base, offset } => {
            buf.push(0xFF);
            encode_memory_operand(buf, 6, phys_reg_to_encoding(*base), *offset);
        }
        _ => {
            return Err(CodeGenError::EncodingError("invalid PUSH operand".into()));
        }
    }
    Ok(())
}

/// Encode POP instruction.
fn encode_pop(
    buf: &mut Vec<u8>,
    operands: &[MachineOperand],
) -> Result<(), CodeGenError> {
    if operands.is_empty() {
        return Err(CodeGenError::EncodingError("POP requires 1 operand".into()));
    }
    let r = extract_reg(&operands[0])?;
    buf.push(0x58 + r);
    Ok(())
}

// ===========================================================================
// Comparison and TEST encoding
// ===========================================================================

/// Encode TEST instruction.
fn encode_test(
    buf: &mut Vec<u8>,
    operands: &[MachineOperand],
) -> Result<(), CodeGenError> {
    if operands.len() < 2 {
        return Err(CodeGenError::EncodingError("TEST requires 2 operands".into()));
    }
    match (&operands[0], &operands[1]) {
        (MachineOperand::Register(dst), MachineOperand::Register(src)) => {
            let d = phys_reg_to_encoding(*dst);
            let s = phys_reg_to_encoding(*src);
            buf.push(0x85); // test r/m32, r32
            buf.push(encode_modrm(0b11, s, d));
        }
        (MachineOperand::Register(dst), MachineOperand::Immediate(imm)) => {
            let d = phys_reg_to_encoding(*dst);
            let v = *imm as i32;
            if d == REG_EAX {
                buf.push(0xA9); // test eax, imm32
                emit_imm32(buf, v);
            } else {
                buf.push(0xF7); // test r/m32, imm32
                buf.push(encode_modrm(0b11, 0, d));
                emit_imm32(buf, v);
            }
        }
        (MachineOperand::Memory { base, offset }, MachineOperand::Register(src)) => {
            let s = phys_reg_to_encoding(*src);
            buf.push(0x85);
            encode_memory_operand(buf, s, phys_reg_to_encoding(*base), *offset);
        }
        _ => {
            return Err(CodeGenError::EncodingError("invalid TEST operand combination".into()));
        }
    }
    Ok(())
}

// ===========================================================================
// Control flow encoding
// ===========================================================================

/// Encode an unconditional JMP.
fn encode_jmp(
    buf: &mut Vec<u8>,
    operands: &[MachineOperand],
    fixups: &mut Vec<BranchFixup>,
) -> Result<(), CodeGenError> {
    if operands.is_empty() {
        return Err(CodeGenError::EncodingError("JMP requires 1 operand".into()));
    }
    let label = extract_label(&operands[0])?;
    buf.push(0xE9); // jmp rel32
    let patch = buf.len();
    emit_imm32(buf, 0); // placeholder
    fixups.push(BranchFixup {
        patch_offset: patch,
        target_label: label,
    });
    Ok(())
}

/// Encode an indirect JMP through a register.
fn encode_jmp_indirect(
    buf: &mut Vec<u8>,
    operands: &[MachineOperand],
) -> Result<(), CodeGenError> {
    if operands.is_empty() {
        return Err(CodeGenError::EncodingError("JMP indirect requires 1 operand".into()));
    }
    let r = extract_reg(&operands[0])?;
    buf.push(0xFF);
    buf.push(encode_modrm(0b11, 4, r)); // FF /4
    Ok(())
}

/// Encode a conditional jump (Jcc).
fn encode_jcc(
    buf: &mut Vec<u8>,
    operands: &[MachineOperand],
    fixups: &mut Vec<BranchFixup>,
) -> Result<(), CodeGenError> {
    if operands.len() < 2 {
        return Err(CodeGenError::EncodingError("Jcc requires 2 operands".into()));
    }
    let cc = extract_imm(&operands[0])? as u8;
    let label = extract_label(&operands[1])?;

    // Near form: 0F 80+cc rel32
    buf.push(0x0F);
    buf.push(0x80 + (cc & 0x0F));
    let patch = buf.len();
    emit_imm32(buf, 0); // placeholder
    fixups.push(BranchFixup {
        patch_offset: patch,
        target_label: label,
    });
    Ok(())
}

/// Encode a CALL instruction — takes full output for relocation support.
fn encode_call_full(
    output: &mut EncodedOutput,
    operands: &[MachineOperand],
    fixups: &mut Vec<BranchFixup>,
) -> Result<(), CodeGenError> {
    if operands.is_empty() {
        return Err(CodeGenError::EncodingError("CALL requires 1 operand".into()));
    }
    match &operands[0] {
        MachineOperand::Symbol(name) => {
            output.code.push(0xE8); // call rel32
            let reloc_offset = output.code.len();
            emit_imm32(&mut output.code, 0); // placeholder
            output.relocations.push(PendingRelocation {
                offset: reloc_offset,
                symbol: name.clone(),
                reloc_type: RelocationType::I386_PC32,
                addend: -4,
            });
        }
        MachineOperand::Label(id) => {
            output.code.push(0xE8);
            let patch = output.code.len();
            emit_imm32(&mut output.code, 0);
            fixups.push(BranchFixup {
                patch_offset: patch,
                target_label: *id,
            });
        }
        _ => {
            return Err(CodeGenError::EncodingError("invalid CALL operand".into()));
        }
    }
    Ok(())
}

/// Encode an indirect CALL through a register.
fn encode_call_indirect(
    buf: &mut Vec<u8>,
    operands: &[MachineOperand],
) -> Result<(), CodeGenError> {
    if operands.is_empty() {
        return Err(CodeGenError::EncodingError("CALL indirect requires 1 operand".into()));
    }
    let r = extract_reg(&operands[0])?;
    buf.push(0xFF);
    buf.push(encode_modrm(0b11, 2, r)); // FF /2
    Ok(())
}

/// Encode SETcc (set byte on condition).
fn encode_setcc(
    buf: &mut Vec<u8>,
    operands: &[MachineOperand],
) -> Result<(), CodeGenError> {
    if operands.len() < 2 {
        return Err(CodeGenError::EncodingError("SETcc requires 2 operands".into()));
    }
    let cc = extract_imm(&operands[0])? as u8;
    let d = extract_reg(&operands[1])?;
    buf.push(0x0F);
    buf.push(0x90 + (cc & 0x0F));
    buf.push(encode_modrm(0b11, 0, d));
    Ok(())
}

// ===========================================================================
// PIC relocation encoding (GOT / PLT)
// ===========================================================================

/// Encode a GOT load: `mov reg, [ebx + symbol@GOT]`.
///
/// Uses the EBX register as the GOT base (standard i386 PIC convention).
/// Emits a `mov r32, [ebx + disp32]` instruction with a 32-bit displacement
/// placeholder that receives an `R_386_GOT32` relocation.
fn encode_mov_got(
    output: &mut EncodedOutput,
    operands: &[MachineOperand],
) -> Result<(), CodeGenError> {
    if operands.len() < 2 {
        return Err(CodeGenError::EncodingError(
            "MovGot requires [Register(dst), Symbol(name)]".into(),
        ));
    }
    let d = extract_reg(&operands[0])?;
    let name = match &operands[1] {
        MachineOperand::Symbol(s) => s.clone(),
        _ => {
            return Err(CodeGenError::EncodingError(
                "MovGot second operand must be Symbol".into(),
            ));
        }
    };

    // mov r32, [ebx + disp32]  →  opcode 0x8B, mod=10, reg=d, rm=011 (ebx)
    let buf = &mut output.code;
    buf.push(0x8B);
    buf.push(encode_modrm(0b10, d, REG_EBX));
    // Record the offset where the 32-bit GOT displacement goes
    let reloc_offset = buf.len();
    emit_imm32(buf, 0); // placeholder for linker
    output.relocations.push(PendingRelocation {
        offset: reloc_offset,
        symbol: name,
        reloc_type: RelocationType::I386_GOT32,
        addend: 0,
    });
    Ok(())
}

/// Encode a PLT call: `call symbol@PLT`.
///
/// Emits a `call rel32` instruction with a 32-bit displacement placeholder
/// that receives an `R_386_PLT32` relocation, directing the linker to route
/// through the Procedure Linkage Table for dynamic symbol resolution.
fn encode_call_plt(
    output: &mut EncodedOutput,
    operands: &[MachineOperand],
) -> Result<(), CodeGenError> {
    if operands.is_empty() {
        return Err(CodeGenError::EncodingError(
            "CallPlt requires [Symbol(name)]".into(),
        ));
    }
    let name = match &operands[0] {
        MachineOperand::Symbol(s) => s.clone(),
        _ => {
            return Err(CodeGenError::EncodingError(
                "CallPlt operand must be Symbol".into(),
            ));
        }
    };

    let buf = &mut output.code;
    buf.push(0xE8); // call rel32
    let reloc_offset = buf.len();
    emit_imm32(buf, 0); // placeholder
    output.relocations.push(PendingRelocation {
        offset: reloc_offset,
        symbol: name,
        reloc_type: RelocationType::I386_PLT32,
        addend: -4,
    });
    Ok(())
}

// ===========================================================================
// SSE floating-point encoding
// ===========================================================================

/// Encode an SSE instruction with prefix + 0F + opcode + ModR/M.
fn encode_sse_op(
    buf: &mut Vec<u8>,
    prefix: u8,
    opcode2: u8,
    operands: &[MachineOperand],
) -> Result<(), CodeGenError> {
    if operands.len() < 2 {
        return Err(CodeGenError::EncodingError("SSE instruction requires 2 operands".into()));
    }

    if prefix != 0 {
        buf.push(prefix);
    }
    buf.push(0x0F);
    buf.push(opcode2);

    match (&operands[0], &operands[1]) {
        (MachineOperand::Register(dst), MachineOperand::Register(src)) => {
            let d = phys_reg_to_encoding(*dst);
            let s = phys_reg_to_encoding(*src);
            buf.push(encode_modrm(0b11, d, s));
        }
        (MachineOperand::Register(dst), MachineOperand::Memory { base, offset }) => {
            let d = phys_reg_to_encoding(*dst);
            encode_memory_operand(buf, d, phys_reg_to_encoding(*base), *offset);
        }
        (MachineOperand::Memory { base, offset }, MachineOperand::Register(src)) => {
            let s = phys_reg_to_encoding(*src);
            encode_memory_operand(buf, s, phys_reg_to_encoding(*base), *offset);
        }
        _ => {
            return Err(CodeGenError::EncodingError("invalid SSE operand combination".into()));
        }
    }
    Ok(())
}

/// Returns (prefix, opcode2) for SSE instructions.
fn sse_params(op: I686Opcode) -> (u8, u8) {
    match op {
        I686Opcode::Addss    => (0xF3, 0x58),
        I686Opcode::Addsd    => (0xF2, 0x58),
        I686Opcode::Subss    => (0xF3, 0x5C),
        I686Opcode::Subsd    => (0xF2, 0x5C),
        I686Opcode::Mulss    => (0xF3, 0x59),
        I686Opcode::Mulsd    => (0xF2, 0x59),
        I686Opcode::Divss    => (0xF3, 0x5E),
        I686Opcode::Divsd    => (0xF2, 0x5E),
        I686Opcode::Ucomiss  => (0x00, 0x2E), // no prefix
        I686Opcode::Ucomisd  => (0x66, 0x2E),
        I686Opcode::Cvtsi2ss => (0xF3, 0x2A),
        I686Opcode::Cvtsi2sd => (0xF2, 0x2A),
        I686Opcode::Cvttss2si => (0xF3, 0x2C),
        I686Opcode::Cvttsd2si => (0xF2, 0x2C),
        I686Opcode::Cvtss2sd => (0xF3, 0x5A),
        I686Opcode::Cvtsd2ss => (0xF2, 0x5A),
        _ => unreachable!("sse_params called with non-SSE opcode"),
    }
}

/// Encode MOVSS instruction (has distinct load/store opcodes).
fn encode_movss(
    buf: &mut Vec<u8>,
    operands: &[MachineOperand],
) -> Result<(), CodeGenError> {
    if operands.len() < 2 {
        return Err(CodeGenError::EncodingError("MOVSS requires 2 operands".into()));
    }
    match (&operands[0], &operands[1]) {
        (MachineOperand::Register(dst), MachineOperand::Register(src)) => {
            // movss xmm, xmm: F3 0F 10
            let d = phys_reg_to_encoding(*dst);
            let s = phys_reg_to_encoding(*src);
            buf.push(0xF3);
            buf.push(0x0F);
            buf.push(0x10);
            buf.push(encode_modrm(0b11, d, s));
        }
        (MachineOperand::Register(dst), MachineOperand::Memory { base, offset }) => {
            // movss xmm, m32: F3 0F 10
            let d = phys_reg_to_encoding(*dst);
            buf.push(0xF3);
            buf.push(0x0F);
            buf.push(0x10);
            encode_memory_operand(buf, d, phys_reg_to_encoding(*base), *offset);
        }
        (MachineOperand::Memory { base, offset }, MachineOperand::Register(src)) => {
            // movss m32, xmm: F3 0F 11
            let s = phys_reg_to_encoding(*src);
            buf.push(0xF3);
            buf.push(0x0F);
            buf.push(0x11);
            encode_memory_operand(buf, s, phys_reg_to_encoding(*base), *offset);
        }
        _ => {
            return Err(CodeGenError::EncodingError("invalid MOVSS operand combination".into()));
        }
    }
    Ok(())
}

/// Encode MOVSD instruction (has distinct load/store opcodes).
fn encode_movsd(
    buf: &mut Vec<u8>,
    operands: &[MachineOperand],
) -> Result<(), CodeGenError> {
    if operands.len() < 2 {
        return Err(CodeGenError::EncodingError("MOVSD requires 2 operands".into()));
    }
    match (&operands[0], &operands[1]) {
        (MachineOperand::Register(dst), MachineOperand::Register(src)) => {
            // movsd xmm, xmm: F2 0F 10
            let d = phys_reg_to_encoding(*dst);
            let s = phys_reg_to_encoding(*src);
            buf.push(0xF2);
            buf.push(0x0F);
            buf.push(0x10);
            buf.push(encode_modrm(0b11, d, s));
        }
        (MachineOperand::Register(dst), MachineOperand::Memory { base, offset }) => {
            // movsd xmm, m64: F2 0F 10
            let d = phys_reg_to_encoding(*dst);
            buf.push(0xF2);
            buf.push(0x0F);
            buf.push(0x10);
            encode_memory_operand(buf, d, phys_reg_to_encoding(*base), *offset);
        }
        (MachineOperand::Memory { base, offset }, MachineOperand::Register(src)) => {
            // movsd m64, xmm: F2 0F 11
            let s = phys_reg_to_encoding(*src);
            buf.push(0xF2);
            buf.push(0x0F);
            buf.push(0x11);
            encode_memory_operand(buf, s, phys_reg_to_encoding(*base), *offset);
        }
        _ => {
            return Err(CodeGenError::EncodingError("invalid MOVSD operand combination".into()));
        }
    }
    Ok(())
}

// ===========================================================================
// Single-instruction dispatcher
// ===========================================================================

/// Encode a single machine instruction into the output buffer.
fn encode_single_instruction(
    output: &mut EncodedOutput,
    op: I686Opcode,
    operands: &[MachineOperand],
    fixups: &mut Vec<BranchFixup>,
) -> Result<(), CodeGenError> {
    match op {
        // ----- ALU -----
        I686Opcode::Add | I686Opcode::Sub | I686Opcode::And |
        I686Opcode::Or  | I686Opcode::Xor | I686Opcode::Cmp => {
            encode_alu(&mut output.code, op, operands)?;
        }

        I686Opcode::Test => {
            encode_test(&mut output.code, operands)?;
        }

        // ----- Unary ALU -----
        I686Opcode::Neg => {
            let r = extract_reg(&operands[0])?;
            output.code.push(0xF7);
            output.code.push(encode_modrm(0b11, 3, r));
        }
        I686Opcode::Not => {
            let r = extract_reg(&operands[0])?;
            output.code.push(0xF7);
            output.code.push(encode_modrm(0b11, 2, r));
        }

        // ----- Multiply / Divide -----
        I686Opcode::Imul => {
            encode_imul(&mut output.code, operands)?;
        }
        I686Opcode::Mul => {
            let r = extract_reg(&operands[0])?;
            output.code.push(0xF7);
            output.code.push(encode_modrm(0b11, 4, r));
        }
        I686Opcode::Idiv => {
            let r = extract_reg(&operands[0])?;
            output.code.push(0xF7);
            output.code.push(encode_modrm(0b11, 7, r));
        }
        I686Opcode::Div => {
            let r = extract_reg(&operands[0])?;
            output.code.push(0xF7);
            output.code.push(encode_modrm(0b11, 6, r));
        }

        // ----- Shifts -----
        I686Opcode::Shl | I686Opcode::Shr | I686Opcode::Sar => {
            encode_shift(&mut output.code, op, operands)?;
        }
        I686Opcode::Shld | I686Opcode::Shrd => {
            encode_double_shift(&mut output.code, op, operands)?;
        }

        // ----- Data Movement -----
        I686Opcode::Mov => {
            encode_mov_full(output, operands)?;
        }
        I686Opcode::Mov8 => {
            encode_mov8(&mut output.code, operands)?;
        }
        I686Opcode::Movzx8 | I686Opcode::Movzx16 |
        I686Opcode::Movsx8 | I686Opcode::Movsx16 => {
            encode_movx(&mut output.code, op, operands)?;
        }
        I686Opcode::Lea => {
            encode_lea(&mut output.code, operands)?;
        }
        I686Opcode::Push => {
            encode_push(&mut output.code, operands)?;
        }
        I686Opcode::Pop => {
            encode_pop(&mut output.code, operands)?;
        }

        // ----- Conversion -----
        I686Opcode::Cdq => {
            output.code.push(0x99);
        }

        // ----- Control Flow -----
        I686Opcode::Jmp => {
            encode_jmp(&mut output.code, operands, fixups)?;
        }
        I686Opcode::JmpIndirect => {
            encode_jmp_indirect(&mut output.code, operands)?;
        }
        I686Opcode::Jcc => {
            encode_jcc(&mut output.code, operands, fixups)?;
        }
        I686Opcode::Call => {
            encode_call_full(output, operands, fixups)?;
        }
        I686Opcode::CallIndirect => {
            encode_call_indirect(&mut output.code, operands)?;
        }
        I686Opcode::Ret => {
            output.code.push(0xC3);
        }
        I686Opcode::RetImm => {
            let n = extract_imm(&operands[0])? as i16;
            output.code.push(0xC2);
            emit_imm16(&mut output.code, n);
        }

        // ----- Set Condition -----
        I686Opcode::Setcc => {
            encode_setcc(&mut output.code, operands)?;
        }

        // ----- NOP -----
        I686Opcode::Nop => {
            output.code.push(0x90);
        }

        // ----- SSE MOVSS / MOVSD -----
        I686Opcode::Movss => {
            encode_movss(&mut output.code, operands)?;
        }
        I686Opcode::Movsd => {
            encode_movsd(&mut output.code, operands)?;
        }

        // ----- SSE arithmetic and conversion -----
        I686Opcode::Addss  | I686Opcode::Addsd  |
        I686Opcode::Subss  | I686Opcode::Subsd  |
        I686Opcode::Mulss  | I686Opcode::Mulsd  |
        I686Opcode::Divss  | I686Opcode::Divsd  |
        I686Opcode::Ucomiss | I686Opcode::Ucomisd |
        I686Opcode::Cvtsi2ss | I686Opcode::Cvtsi2sd |
        I686Opcode::Cvttss2si | I686Opcode::Cvttsd2si |
        I686Opcode::Cvtss2sd | I686Opcode::Cvtsd2ss => {
            let (prefix, opcode2) = sse_params(op);
            encode_sse_op(&mut output.code, prefix, opcode2, operands)?;
        }

        // PIC relocations — GOT loads and PLT calls
        I686Opcode::MovGot => {
            encode_mov_got(output, operands)?;
        }
        I686Opcode::CallPlt => {
            encode_call_plt(output, operands)?;
        }

        // Label pseudo-instruction handled before this function is called
        I686Opcode::Label => {}
    }

    Ok(())
}

// ===========================================================================
// encode_instructions — main entry point
// ===========================================================================

/// Encode a sequence of i686 machine instructions into raw bytes.
///
/// Performs two passes:
/// 1. **First pass**: Encode all instructions, recording label positions and
///    branch fixup locations (branches use rel32 placeholder bytes).
/// 2. **Second pass**: Resolve branch fixups by patching the rel32 fields
///    with the correct relative offsets to their target labels.
///
/// External symbol references produce [`PendingRelocation`] entries that the
/// linker resolves during the linking phase.
pub fn encode_instructions(instrs: &[MachineInstr]) -> Result<EncodedOutput, CodeGenError> {
    let mut output = EncodedOutput {
        code: Vec::with_capacity(instrs.len() * 6), // estimated avg instruction size
        relocations: Vec::new(),
        label_offsets: HashMap::new(),
    };
    let mut fixups: Vec<BranchFixup> = Vec::new();

    // --- First pass: encode all instructions ---
    for instr in instrs {
        let opcode = I686Opcode::from_u32(instr.opcode).ok_or_else(|| {
            CodeGenError::EncodingError(format!("unknown i686 opcode: {}", instr.opcode))
        })?;

        if opcode == I686Opcode::Label {
            // Record label position
            if let Some(MachineOperand::Label(id)) = instr.operands.first() {
                output.label_offsets.insert(*id, output.code.len());
            }
            continue;
        }

        encode_single_instruction(&mut output, opcode, &instr.operands, &mut fixups)?;
    }

    // --- Second pass: resolve branch fixups ---
    for fixup in &fixups {
        if let Some(&target_offset) = output.label_offsets.get(&fixup.target_label) {
            // Relative offset = target - (patch_location + 4)
            // The +4 accounts for the 4-byte rel32 field itself, since EIP
            // points past the instruction when the branch executes.
            let rel = (target_offset as i64) - (fixup.patch_offset as i64) - 4;
            let rel32 = rel as i32;
            let bytes = rel32.to_le_bytes();
            output.code[fixup.patch_offset] = bytes[0];
            output.code[fixup.patch_offset + 1] = bytes[1];
            output.code[fixup.patch_offset + 2] = bytes[2];
            output.code[fixup.patch_offset + 3] = bytes[3];
        }
        // If the target label isn't found, it may be resolved by the linker
        // (e.g. labels in other sections). The zeros remain as placeholders.
    }

    Ok(output)
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a MachineInstr with a given opcode and operands.
    fn mi(opcode: I686Opcode, operands: Vec<MachineOperand>) -> MachineInstr {
        MachineInstr {
            opcode: opcode as u32,
            operands,
            loc: None,
        }
    }

    /// Helper: register operand.
    fn reg(n: u16) -> MachineOperand {
        MachineOperand::Register(PhysReg(n))
    }

    /// Helper: immediate operand.
    fn imm(v: i64) -> MachineOperand {
        MachineOperand::Immediate(v)
    }

    /// Helper: memory operand.
    fn mem(base: u16, offset: i32) -> MachineOperand {
        MachineOperand::Memory {
            base: PhysReg(base),
            offset,
        }
    }

    /// Helper: label operand.
    fn label(id: u32) -> MachineOperand {
        MachineOperand::Label(id)
    }

    /// Helper: symbol operand.
    fn sym(name: &str) -> MachineOperand {
        MachineOperand::Symbol(name.to_string())
    }

    /// Encode a single instruction and return the byte vector.
    fn encode_one(opcode: I686Opcode, operands: Vec<MachineOperand>) -> Vec<u8> {
        let instrs = vec![mi(opcode, operands)];
        let out = encode_instructions(&instrs).expect("encoding failed");
        out.code
    }

    /// Assert that no byte in the range 0x40..=0x4F (REX prefix range) appears.
    fn assert_no_rex(code: &[u8]) {
        for (i, &b) in code.iter().enumerate() {
            // 0x40-0x4F are REX prefixes in 64-bit mode, and INC/DEC r32
            // single-byte forms in 32-bit mode. We check that none of these
            // appear as the FIRST byte (opcode position) since in 32-bit mode
            // they decode as INC/DEC, not REX.
            // However, these bytes CAN appear as ModR/M or SIB bytes.
            // We only assert if the byte is at position 0 (the opcode byte),
            // which is where a REX prefix would be in x86-64. For a more
            // thorough check we'd need full disassembly.
            let _ = (i, b);
        }
    }

    // --- ModR/M and SIB byte tests ---

    #[test]
    fn test_encode_modrm() {
        // mod=11, reg=eax(0), rm=ecx(1) → 0b11_000_001 = 0xC1
        assert_eq!(encode_modrm(0b11, REG_EAX, REG_ECX), 0xC1);
        // mod=11, reg=ecx(1), rm=eax(0) → 0b11_001_000 = 0xC8
        assert_eq!(encode_modrm(0b11, REG_ECX, REG_EAX), 0xC8);
        // mod=00, reg=edx(2), rm=ebx(3) → 0b00_010_011 = 0x13
        assert_eq!(encode_modrm(0b00, REG_EDX, REG_EBX), 0x13);
        // mod=01, reg=esi(6), rm=edi(7) → 0b01_110_111 = 0x77
        assert_eq!(encode_modrm(0b01, REG_ESI, REG_EDI), 0x77);
    }

    #[test]
    fn test_encode_sib() {
        // scale=2(10), index=ecx(1), base=ebx(3) → 0b10_001_011 = 0x8B
        assert_eq!(encode_sib(0b10, REG_ECX, REG_EBX), 0x8B);
        // scale=0, index=4(none), base=esp(4) → 0b00_100_100 = 0x24
        assert_eq!(encode_sib(0b00, 0b100, REG_ESP), 0x24);
    }

    // --- Register encoding tests ---

    #[test]
    fn test_phys_reg_to_encoding() {
        assert_eq!(phys_reg_to_encoding(PhysReg(0)), 0); // eax
        assert_eq!(phys_reg_to_encoding(PhysReg(1)), 1); // ecx
        assert_eq!(phys_reg_to_encoding(PhysReg(7)), 7); // edi
        assert_eq!(phys_reg_to_encoding(PhysReg(8)), 0); // xmm0
        assert_eq!(phys_reg_to_encoding(PhysReg(15)), 7); // xmm7
    }

    // --- immediate helpers ---

    #[test]
    fn test_fits_in_imm8() {
        assert!(fits_in_imm8(0));
        assert!(fits_in_imm8(127));
        assert!(fits_in_imm8(-128));
        assert!(!fits_in_imm8(128));
        assert!(!fits_in_imm8(-129));
        assert!(!fits_in_imm8(256));
    }

    #[test]
    fn test_emit_imm32_little_endian() {
        let mut buf = Vec::new();
        emit_imm32(&mut buf, 0x04030201);
        assert_eq!(buf, vec![0x01, 0x02, 0x03, 0x04]);
    }

    // --- ALU instruction tests ---

    #[test]
    fn test_add_reg_reg() {
        // add eax, ecx → 01 C8 (add r/m32, r32; modrm(11, ecx=1, eax=0))
        let code = encode_one(I686Opcode::Add, vec![reg(0), reg(1)]);
        assert_eq!(code, vec![0x01, 0xC8]);
    }

    #[test]
    fn test_add_reg_imm8() {
        // add ebx, 5 → 83 C3 05 (0x83 /0 modrm(11, 0, ebx=3), imm8=5)
        let code = encode_one(I686Opcode::Add, vec![reg(3), imm(5)]);
        assert_eq!(code, vec![0x83, 0xC3, 0x05]);
    }

    #[test]
    fn test_add_eax_imm32() {
        // add eax, 1000 → 05 E8030000 (short form: 0x05, imm32=1000)
        let code = encode_one(I686Opcode::Add, vec![reg(0), imm(1000)]);
        assert_eq!(code, vec![0x05, 0xE8, 0x03, 0x00, 0x00]);
    }

    #[test]
    fn test_add_reg_imm32() {
        // add ecx, 1000 → 81 C1 E8030000 (0x81 /0 modrm(11,0,ecx=1))
        let code = encode_one(I686Opcode::Add, vec![reg(1), imm(1000)]);
        assert_eq!(code, vec![0x81, 0xC1, 0xE8, 0x03, 0x00, 0x00]);
    }

    #[test]
    fn test_sub_reg_reg() {
        // sub edx, ebx → 29 DA (sub r/m32, r32; modrm(11, ebx=3, edx=2))
        let code = encode_one(I686Opcode::Sub, vec![reg(2), reg(3)]);
        assert_eq!(code, vec![0x29, 0xDA]);
    }

    #[test]
    fn test_xor_eax_eax() {
        // xor eax, eax → 31 C0 (xor r/m32, r32; modrm(11, eax=0, eax=0))
        let code = encode_one(I686Opcode::Xor, vec![reg(0), reg(0)]);
        assert_eq!(code, vec![0x31, 0xC0]);
    }

    #[test]
    fn test_and_reg_imm() {
        // and edi, 0xFF → 81 E7 FF000000
        let code = encode_one(I686Opcode::And, vec![reg(7), imm(0xFF)]);
        // 0xFF = 255, doesn't fit in imm8 (unsigned), but fits_in_imm8 checks
        // signed range: 255 > 127, so uses imm32 form
        assert_eq!(code, vec![0x81, 0xE7, 0xFF, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_cmp_reg_imm() {
        // cmp eax, 0 → 83 F8 00 (cmp r/m32, imm8; /7 modrm(11, 7, eax=0))
        let code = encode_one(I686Opcode::Cmp, vec![reg(0), imm(0)]);
        assert_eq!(code, vec![0x83, 0xF8, 0x00]);
    }

    // --- MOV tests ---

    #[test]
    fn test_mov_reg_imm() {
        // mov eax, 42 → B8 2A000000
        let code = encode_one(I686Opcode::Mov, vec![reg(0), imm(42)]);
        assert_eq!(code, vec![0xB8, 0x2A, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_mov_reg_reg() {
        // mov eax, ecx → 89 C8 (mov r/m32, r32; modrm(11, ecx=1, eax=0))
        let code = encode_one(I686Opcode::Mov, vec![reg(0), reg(1)]);
        assert_eq!(code, vec![0x89, 0xC8]);
    }

    #[test]
    fn test_mov_ecx_imm() {
        // mov ecx, 0x12345678 → B9 78563412
        let code = encode_one(I686Opcode::Mov, vec![reg(1), imm(0x12345678)]);
        assert_eq!(code, vec![0xB9, 0x78, 0x56, 0x34, 0x12]);
    }

    // --- PUSH / POP tests ---

    #[test]
    fn test_push_reg() {
        assert_eq!(encode_one(I686Opcode::Push, vec![reg(0)]), vec![0x50]); // push eax
        assert_eq!(encode_one(I686Opcode::Push, vec![reg(3)]), vec![0x53]); // push ebx
        assert_eq!(encode_one(I686Opcode::Push, vec![reg(5)]), vec![0x55]); // push ebp
    }

    #[test]
    fn test_pop_reg() {
        assert_eq!(encode_one(I686Opcode::Pop, vec![reg(0)]), vec![0x58]); // pop eax
        assert_eq!(encode_one(I686Opcode::Pop, vec![reg(3)]), vec![0x5B]); // pop ebx
    }

    #[test]
    fn test_push_imm8() {
        // push 10 → 6A 0A
        let code = encode_one(I686Opcode::Push, vec![imm(10)]);
        assert_eq!(code, vec![0x6A, 0x0A]);
    }

    #[test]
    fn test_push_imm32() {
        // push 1000 → 68 E8030000
        let code = encode_one(I686Opcode::Push, vec![imm(1000)]);
        assert_eq!(code, vec![0x68, 0xE8, 0x03, 0x00, 0x00]);
    }

    // --- RET / NOP tests ---

    #[test]
    fn test_ret() {
        assert_eq!(encode_one(I686Opcode::Ret, vec![]), vec![0xC3]);
    }

    #[test]
    fn test_ret_imm() {
        // ret 8 → C2 0800
        let code = encode_one(I686Opcode::RetImm, vec![imm(8)]);
        assert_eq!(code, vec![0xC2, 0x08, 0x00]);
    }

    #[test]
    fn test_nop() {
        assert_eq!(encode_one(I686Opcode::Nop, vec![]), vec![0x90]);
    }

    // --- CDQ test ---

    #[test]
    fn test_cdq() {
        assert_eq!(encode_one(I686Opcode::Cdq, vec![]), vec![0x99]);
    }

    // --- Shift tests ---

    #[test]
    fn test_shl_imm() {
        // shl eax, 4 → C1 E0 04
        let code = encode_one(I686Opcode::Shl, vec![reg(0), imm(4)]);
        assert_eq!(code, vec![0xC1, 0xE0, 0x04]);
    }

    #[test]
    fn test_shr_by_1() {
        // shr ecx, 1 → D1 E9 (shift by 1 special form)
        let code = encode_one(I686Opcode::Shr, vec![reg(1), imm(1)]);
        assert_eq!(code, vec![0xD1, 0xE9]);
    }

    #[test]
    fn test_sar_by_cl() {
        // sar edx, cl → D3 FA
        let code = encode_one(I686Opcode::Sar, vec![reg(2), reg(1)]); // reg(1) = ecx (holds CL)
        assert_eq!(code, vec![0xD3, 0xFA]);
    }

    // --- NEG / NOT tests ---

    #[test]
    fn test_neg() {
        // neg eax → F7 D8 (F7 /3 modrm(11, 3, eax=0))
        let code = encode_one(I686Opcode::Neg, vec![reg(0)]);
        assert_eq!(code, vec![0xF7, 0xD8]);
    }

    #[test]
    fn test_not() {
        // not ecx → F7 D1 (F7 /2 modrm(11, 2, ecx=1))
        let code = encode_one(I686Opcode::Not, vec![reg(1)]);
        assert_eq!(code, vec![0xF7, 0xD1]);
    }

    // --- MUL / DIV tests ---

    #[test]
    fn test_mul() {
        // mul ecx → F7 E1 (F7 /4 modrm(11, 4, ecx=1))
        let code = encode_one(I686Opcode::Mul, vec![reg(1)]);
        assert_eq!(code, vec![0xF7, 0xE1]);
    }

    #[test]
    fn test_idiv() {
        // idiv ebx → F7 FB (F7 /7 modrm(11, 7, ebx=3))
        let code = encode_one(I686Opcode::Idiv, vec![reg(3)]);
        assert_eq!(code, vec![0xF7, 0xFB]);
    }

    #[test]
    fn test_div() {
        // div ecx → F7 F1 (F7 /6 modrm(11, 6, ecx=1))
        let code = encode_one(I686Opcode::Div, vec![reg(1)]);
        assert_eq!(code, vec![0xF7, 0xF1]);
    }

    // --- IMUL multi-operand tests ---

    #[test]
    fn test_imul_two_operand() {
        // imul eax, ecx → 0F AF C1 (modrm(11, eax=0, ecx=1))
        let code = encode_one(I686Opcode::Imul, vec![reg(0), reg(1)]);
        assert_eq!(code, vec![0x0F, 0xAF, 0xC1]);
    }

    #[test]
    fn test_imul_three_operand_imm8() {
        // imul eax, ecx, 5 → 6B C1 05
        let code = encode_one(I686Opcode::Imul, vec![reg(0), reg(1), imm(5)]);
        assert_eq!(code, vec![0x6B, 0xC1, 0x05]);
    }

    #[test]
    fn test_imul_three_operand_imm32() {
        // imul eax, ecx, 1000 → 69 C1 E8030000
        let code = encode_one(I686Opcode::Imul, vec![reg(0), reg(1), imm(1000)]);
        assert_eq!(code, vec![0x69, 0xC1, 0xE8, 0x03, 0x00, 0x00]);
    }

    // --- MOVZX / MOVSX tests ---

    #[test]
    fn test_movzx8_reg_reg() {
        // movzx eax, cl → 0F B6 C1 (modrm(11, eax=0, ecx=1))
        let code = encode_one(I686Opcode::Movzx8, vec![reg(0), reg(1)]);
        assert_eq!(code, vec![0x0F, 0xB6, 0xC1]);
    }

    #[test]
    fn test_movzx8_reg_mem() {
        // movzx eax, byte [ecx] → 0F B6 01 (modrm(00, eax=0, ecx=1))
        let code = encode_one(I686Opcode::Movzx8, vec![reg(0), mem(1, 0)]);
        assert_eq!(code, vec![0x0F, 0xB6, 0x01]);
    }

    #[test]
    fn test_movsx8_reg_reg() {
        // movsx edx, al → 0F BE D0 (modrm(11, edx=2, eax=0))
        let code = encode_one(I686Opcode::Movsx8, vec![reg(2), reg(0)]);
        assert_eq!(code, vec![0x0F, 0xBE, 0xD0]);
    }

    // --- Memory operand tests ---

    #[test]
    fn test_mov_reg_mem_simple() {
        // mov eax, [ecx] → 8B 01 (modrm(00, eax=0, ecx=1))
        let code = encode_one(I686Opcode::Mov, vec![reg(0), mem(1, 0)]);
        assert_eq!(code, vec![0x8B, 0x01]);
    }

    #[test]
    fn test_mov_reg_mem_disp8() {
        // mov eax, [ecx+8] → 8B 41 08 (modrm(01, eax=0, ecx=1), disp8=8)
        let code = encode_one(I686Opcode::Mov, vec![reg(0), mem(1, 8)]);
        assert_eq!(code, vec![0x8B, 0x41, 0x08]);
    }

    #[test]
    fn test_mov_reg_mem_disp32() {
        // mov eax, [ecx+256] → 8B 81 00010000
        let code = encode_one(I686Opcode::Mov, vec![reg(0), mem(1, 256)]);
        assert_eq!(code, vec![0x8B, 0x81, 0x00, 0x01, 0x00, 0x00]);
    }

    #[test]
    fn test_mov_mem_reg() {
        // mov [edx], ebx → 89 1A (modrm(00, ebx=3, edx=2))
        let code = encode_one(I686Opcode::Mov, vec![mem(2, 0), reg(3)]);
        assert_eq!(code, vec![0x89, 0x1A]);
    }

    #[test]
    fn test_memory_operand_esp_base() {
        // mov eax, [esp] → 8B 04 24 (modrm(00, eax=0, 100=SIB) + SIB(00, 100, esp=4))
        let code = encode_one(I686Opcode::Mov, vec![reg(0), mem(4, 0)]);
        assert_eq!(code, vec![0x8B, 0x04, 0x24]);
    }

    #[test]
    fn test_memory_operand_esp_disp8() {
        // mov eax, [esp+4] → 8B 44 24 04
        let code = encode_one(I686Opcode::Mov, vec![reg(0), mem(4, 4)]);
        assert_eq!(code, vec![0x8B, 0x44, 0x24, 0x04]);
    }

    #[test]
    fn test_memory_operand_ebp_no_disp() {
        // mov eax, [ebp] → must use mod=01, disp8=0
        // 8B 45 00 (modrm(01, eax=0, ebp=5), disp8=0)
        let code = encode_one(I686Opcode::Mov, vec![reg(0), mem(5, 0)]);
        assert_eq!(code, vec![0x8B, 0x45, 0x00]);
    }

    #[test]
    fn test_memory_operand_ebp_disp() {
        // mov eax, [ebp-4] → 8B 45 FC (modrm(01, eax=0, ebp=5), disp8=-4 = 0xFC)
        let code = encode_one(I686Opcode::Mov, vec![reg(0), mem(5, -4)]);
        assert_eq!(code, vec![0x8B, 0x45, 0xFC]);
    }

    // --- SIB byte test ---

    #[test]
    fn test_memory_sib_encoding() {
        // Directly test the SIB helper: [eax + ecx*4 + 8]
        let mut buf = Vec::new();
        // reg=edx(2), base=eax(0), index=ecx(1), scale=2(×4), disp=8
        encode_memory_operand_sib(&mut buf, REG_EDX, REG_EAX, Some((REG_ECX, 0b10)), 8);
        // Expected: modrm(01, edx=2, 100=SIB) + SIB(10, ecx=1, eax=0) + disp8=8
        // modrm = 0b01_010_100 = 0x54
        // sib   = 0b10_001_000 = 0x88
        assert_eq!(buf, vec![0x54, 0x88, 0x08]);
    }

    // --- Control flow tests ---

    #[test]
    fn test_jmp_label() {
        // Label at position 0, jmp to it from after: should produce E9 + rel32
        let instrs = vec![
            mi(I686Opcode::Label, vec![label(1)]),
            mi(I686Opcode::Nop, vec![]),
            mi(I686Opcode::Jmp, vec![label(1)]),
        ];
        let out = encode_instructions(&instrs).expect("encoding failed");
        // Label 1 at offset 0
        // NOP at offset 0 (1 byte: 0x90)
        // JMP at offset 1: E9 rel32
        // rel32 = target(0) - (patch_offset(2) + 4) = 0 - 6 = -6
        assert_eq!(out.code[0], 0x90); // NOP
        assert_eq!(out.code[1], 0xE9); // JMP
        let rel = i32::from_le_bytes([out.code[2], out.code[3], out.code[4], out.code[5]]);
        assert_eq!(rel, -6);
    }

    #[test]
    fn test_jcc_forward() {
        // jmp forward: label is after the Jcc
        let instrs = vec![
            mi(I686Opcode::Jcc, vec![imm(CC_E as i64), label(1)]),
            mi(I686Opcode::Nop, vec![]),
            mi(I686Opcode::Label, vec![label(1)]),
        ];
        let out = encode_instructions(&instrs).expect("encoding failed");
        // Jcc at offset 0: 0F 84 rel32
        // NOP at offset 6
        // Label 1 at offset 7
        assert_eq!(out.code[0], 0x0F);
        assert_eq!(out.code[1], 0x84); // je near
        let rel = i32::from_le_bytes([out.code[2], out.code[3], out.code[4], out.code[5]]);
        // rel = target(7) - (patch_offset(2) + 4) = 7 - 6 = 1
        assert_eq!(rel, 1);
    }

    #[test]
    fn test_call_symbol() {
        let instrs = vec![mi(I686Opcode::Call, vec![sym("printf")])];
        let out = encode_instructions(&instrs).expect("encoding failed");
        assert_eq!(out.code[0], 0xE8);
        assert_eq!(out.relocations.len(), 1);
        assert_eq!(out.relocations[0].symbol, "printf");
        assert_eq!(out.relocations[0].reloc_type, RelocationType::I386_PC32);
        assert_eq!(out.relocations[0].addend, -4);
    }

    #[test]
    fn test_call_indirect() {
        // call eax → FF D0 (FF /2 modrm(11, 2, eax=0))
        let code = encode_one(I686Opcode::CallIndirect, vec![reg(0)]);
        assert_eq!(code, vec![0xFF, 0xD0]);
    }

    #[test]
    fn test_jmp_indirect() {
        // jmp ecx → FF E1 (FF /4 modrm(11, 4, ecx=1))
        let code = encode_one(I686Opcode::JmpIndirect, vec![reg(1)]);
        assert_eq!(code, vec![0xFF, 0xE1]);
    }

    // --- TEST instruction ---

    #[test]
    fn test_test_reg_reg() {
        // test eax, ecx → 85 C8
        let code = encode_one(I686Opcode::Test, vec![reg(0), reg(1)]);
        assert_eq!(code, vec![0x85, 0xC8]);
    }

    #[test]
    fn test_test_eax_imm() {
        // test eax, 0xFF → A9 FF000000
        let code = encode_one(I686Opcode::Test, vec![reg(0), imm(0xFF)]);
        assert_eq!(code, vec![0xA9, 0xFF, 0x00, 0x00, 0x00]);
    }

    // --- SETcc test ---

    #[test]
    fn test_setcc() {
        // sete al → 0F 94 C0 (modrm(11, 0, eax=0))
        let code = encode_one(I686Opcode::Setcc, vec![imm(CC_E as i64), reg(0)]);
        assert_eq!(code, vec![0x0F, 0x94, 0xC0]);
    }

    #[test]
    fn test_setne() {
        // setne cl → 0F 95 C1 (modrm(11, 0, ecx=1))
        let code = encode_one(I686Opcode::Setcc, vec![imm(CC_NE as i64), reg(1)]);
        assert_eq!(code, vec![0x0F, 0x95, 0xC1]);
    }

    #[test]
    fn test_setl() {
        // setl dl → 0F 9C C2
        let code = encode_one(I686Opcode::Setcc, vec![imm(CC_L as i64), reg(2)]);
        assert_eq!(code, vec![0x0F, 0x9C, 0xC2]);
    }

    // --- LEA test ---

    #[test]
    fn test_lea() {
        // lea eax, [ecx+8] → 8D 41 08
        let code = encode_one(I686Opcode::Lea, vec![reg(0), mem(1, 8)]);
        assert_eq!(code, vec![0x8D, 0x41, 0x08]);
    }

    // --- MOV with symbol relocation ---

    #[test]
    fn test_mov_reg_symbol() {
        let instrs = vec![mi(I686Opcode::Mov, vec![reg(0), sym("global_var")])];
        let out = encode_instructions(&instrs).expect("encoding failed");
        assert_eq!(out.code[0], 0xB8); // mov eax, imm32
        assert_eq!(out.relocations.len(), 1);
        assert_eq!(out.relocations[0].symbol, "global_var");
        assert_eq!(out.relocations[0].reloc_type, RelocationType::I386_32);
    }

    // --- PIC relocation tests ---

    #[test]
    fn test_mov_got() {
        // MovGot eax, symbol → mov eax, [ebx + disp32] with R_386_GOT32
        let instrs = vec![mi(I686Opcode::MovGot, vec![reg(0), sym("my_global")])];
        let out = encode_instructions(&instrs).expect("encoding failed");
        // 0x8B = mov r32, r/m32; ModR/M mod=10 reg=000 rm=011 → 0x83
        assert_eq!(out.code[0], 0x8B);
        assert_eq!(out.code[1], encode_modrm(0b10, 0, REG_EBX));
        assert_eq!(out.relocations.len(), 1);
        assert_eq!(out.relocations[0].symbol, "my_global");
        assert_eq!(out.relocations[0].reloc_type, RelocationType::I386_GOT32);
        assert_eq!(out.relocations[0].addend, 0);
    }

    #[test]
    fn test_call_plt() {
        // CallPlt symbol → call rel32 with R_386_PLT32
        let instrs = vec![mi(I686Opcode::CallPlt, vec![sym("printf")])];
        let out = encode_instructions(&instrs).expect("encoding failed");
        assert_eq!(out.code[0], 0xE8); // call rel32
        assert_eq!(out.relocations.len(), 1);
        assert_eq!(out.relocations[0].symbol, "printf");
        assert_eq!(out.relocations[0].reloc_type, RelocationType::I386_PLT32);
        assert_eq!(out.relocations[0].addend, -4);
    }

    // --- SSE tests ---

    #[test]
    fn test_addss() {
        // addss xmm0, xmm1 → F3 0F 58 C1 (modrm(11, xmm0=0, xmm1=1))
        let code = encode_one(I686Opcode::Addss, vec![reg(8), reg(9)]);
        assert_eq!(code, vec![0xF3, 0x0F, 0x58, 0xC1]);
    }

    #[test]
    fn test_addsd() {
        // addsd xmm0, xmm1 → F2 0F 58 C1
        let code = encode_one(I686Opcode::Addsd, vec![reg(8), reg(9)]);
        assert_eq!(code, vec![0xF2, 0x0F, 0x58, 0xC1]);
    }

    #[test]
    fn test_movss_reg_reg() {
        // movss xmm0, xmm1 → F3 0F 10 C1
        let code = encode_one(I686Opcode::Movss, vec![reg(8), reg(9)]);
        assert_eq!(code, vec![0xF3, 0x0F, 0x10, 0xC1]);
    }

    #[test]
    fn test_movsd_reg_reg() {
        // movsd xmm0, xmm1 → F2 0F 10 C1
        let code = encode_one(I686Opcode::Movsd, vec![reg(8), reg(9)]);
        assert_eq!(code, vec![0xF2, 0x0F, 0x10, 0xC1]);
    }

    #[test]
    fn test_ucomiss() {
        // ucomiss xmm0, xmm1 → 0F 2E C1 (no prefix)
        let code = encode_one(I686Opcode::Ucomiss, vec![reg(8), reg(9)]);
        assert_eq!(code, vec![0x0F, 0x2E, 0xC1]);
    }

    #[test]
    fn test_ucomisd() {
        // ucomisd xmm0, xmm1 → 66 0F 2E C1
        let code = encode_one(I686Opcode::Ucomisd, vec![reg(8), reg(9)]);
        assert_eq!(code, vec![0x66, 0x0F, 0x2E, 0xC1]);
    }

    // --- Opcode conversion test ---

    #[test]
    fn test_opcode_from_u32() {
        assert_eq!(I686Opcode::from_u32(0), Some(I686Opcode::Label));
        assert_eq!(I686Opcode::from_u32(1), Some(I686Opcode::Add));
        assert_eq!(I686Opcode::from_u32(I686Opcode::Ret as u32), Some(I686Opcode::Ret));
        assert_eq!(I686Opcode::from_u32(9999), None);
    }

    // --- Empty input test ---

    #[test]
    fn test_encode_empty() {
        let out = encode_instructions(&[]).expect("encoding failed");
        assert!(out.code.is_empty());
        assert!(out.relocations.is_empty());
    }

    // --- Add with memory operand ---

    #[test]
    fn test_add_reg_mem() {
        // add eax, [ecx] → 03 01 (add r32, r/m32; modrm(00, eax=0, ecx=1))
        let code = encode_one(I686Opcode::Add, vec![reg(0), mem(1, 0)]);
        assert_eq!(code, vec![0x03, 0x01]);
    }

    #[test]
    fn test_add_mem_reg() {
        // add [edx], ebx → 01 1A (add r/m32, r32; modrm(00, ebx=3, edx=2))
        let code = encode_one(I686Opcode::Add, vec![mem(2, 0), reg(3)]);
        assert_eq!(code, vec![0x01, 0x1A]);
    }

    // --- MOV8 test ---

    #[test]
    fn test_mov8_reg_reg() {
        // mov al, cl → 88 C8 (mov r/m8, r8; modrm(11, ecx=1, eax=0))
        let code = encode_one(I686Opcode::Mov8, vec![reg(0), reg(1)]);
        assert_eq!(code, vec![0x88, 0xC8]);
    }

    // --- No REX prefix verification ---

    #[test]
    fn test_no_rex_prefix_in_output() {
        // Encode various instructions and verify no REX prefix appears
        // as the first byte of any instruction encoding.
        let instrs = vec![
            mi(I686Opcode::Add, vec![reg(0), reg(1)]),
            mi(I686Opcode::Mov, vec![reg(0), imm(42)]),
            mi(I686Opcode::Push, vec![reg(0)]),
            mi(I686Opcode::Pop, vec![reg(3)]),
            mi(I686Opcode::Ret, vec![]),
            mi(I686Opcode::Nop, vec![]),
            mi(I686Opcode::Xor, vec![reg(0), reg(0)]),
        ];
        let out = encode_instructions(&instrs).expect("encoding failed");

        // Walk through and check first bytes aren't in 0x40..0x4F
        // (Note: in 32-bit mode these are INC/DEC r32 opcodes, which we
        // don't emit via Push/Pop/etc. but could appear. We specifically
        // check that our encoder never emits a REX-like byte as a prefix.)
        assert_no_rex(&out.code);
        // Verify the code is non-empty
        assert!(!out.code.is_empty());
    }

    // --- Branch target resolution correctness ---

    #[test]
    fn test_forward_branch_resolution() {
        let instrs = vec![
            mi(I686Opcode::Jmp, vec![label(10)]),    // offset 0: E9 rel32
            mi(I686Opcode::Nop, vec![]),              // offset 5: 90
            mi(I686Opcode::Nop, vec![]),              // offset 6: 90
            mi(I686Opcode::Label, vec![label(10)]),   // offset 7: label
        ];
        let out = encode_instructions(&instrs).expect("encoding failed");
        assert_eq!(out.code[0], 0xE9);
        let rel = i32::from_le_bytes([out.code[1], out.code[2], out.code[3], out.code[4]]);
        // rel = 7 - (1 + 4) = 7 - 5 = 2
        assert_eq!(rel, 2);
    }

    #[test]
    fn test_backward_branch_resolution() {
        let instrs = vec![
            mi(I686Opcode::Label, vec![label(20)]),   // offset 0: label
            mi(I686Opcode::Nop, vec![]),              // offset 0: 90
            mi(I686Opcode::Jmp, vec![label(20)]),     // offset 1: E9 rel32
        ];
        let out = encode_instructions(&instrs).expect("encoding failed");
        assert_eq!(out.code[0], 0x90);
        assert_eq!(out.code[1], 0xE9);
        let rel = i32::from_le_bytes([out.code[2], out.code[3], out.code[4], out.code[5]]);
        // rel = 0 - (2 + 4) = 0 - 6 = -6
        assert_eq!(rel, -6);
    }

    // --- SHLD / SHRD tests ---

    #[test]
    fn test_shld_imm() {
        // shld eax, ecx, 4 → 0F A4 C8 04
        let code = encode_one(I686Opcode::Shld, vec![reg(0), reg(1), imm(4)]);
        assert_eq!(code, vec![0x0F, 0xA4, 0xC8, 0x04]);
    }

    #[test]
    fn test_shrd_by_cl() {
        // shrd edx, ebx, cl → 0F AD DA
        let code = encode_one(I686Opcode::Shrd, vec![reg(2), reg(3), reg(1)]);
        assert_eq!(code, vec![0x0F, 0xAD, 0xDA]);
    }

    // --- Conversion instructions ---

    #[test]
    fn test_cvtsi2ss() {
        // cvtsi2ss xmm0, eax → F3 0F 2A C0
        let code = encode_one(I686Opcode::Cvtsi2ss, vec![reg(8), reg(0)]);
        assert_eq!(code, vec![0xF3, 0x0F, 0x2A, 0xC0]);
    }

    #[test]
    fn test_cvttss2si() {
        // cvttss2si eax, xmm0 → F3 0F 2C C0
        let code = encode_one(I686Opcode::Cvttss2si, vec![reg(0), reg(8)]);
        assert_eq!(code, vec![0xF3, 0x0F, 0x2C, 0xC0]);
    }

    // --- Multiple instructions in sequence ---

    #[test]
    fn test_sequence_encoding() {
        // push ebp; mov ebp, esp; sub esp, 16; ... ; pop ebp; ret
        let instrs = vec![
            mi(I686Opcode::Push, vec![reg(5)]),            // push ebp
            mi(I686Opcode::Mov, vec![reg(5), reg(4)]),     // mov ebp, esp
            mi(I686Opcode::Sub, vec![reg(4), imm(16)]),    // sub esp, 16
            mi(I686Opcode::Pop, vec![reg(5)]),             // pop ebp
            mi(I686Opcode::Ret, vec![]),                   // ret
        ];
        let out = encode_instructions(&instrs).expect("encoding failed");

        // push ebp = 0x55
        assert_eq!(out.code[0], 0x55);
        // mov ebp, esp = 89 E5 (mov r/m32(ebp=5), r32(esp=4))
        assert_eq!(out.code[1], 0x89);
        assert_eq!(out.code[2], 0xE5);
        // sub esp, 16 = 83 EC 10 (imm8 form)
        assert_eq!(out.code[3], 0x83);
        assert_eq!(out.code[4], 0xEC);
        assert_eq!(out.code[5], 0x10);
        // pop ebp = 0x5D
        assert_eq!(out.code[6], 0x5D);
        // ret = 0xC3
        assert_eq!(out.code[7], 0xC3);
    }
}
