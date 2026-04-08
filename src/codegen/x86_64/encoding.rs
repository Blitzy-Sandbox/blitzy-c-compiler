//! # x86-64 Integrated Assembler — Machine Code Encoder
//!
//! This module implements the x86-64 machine code encoder — the integrated
//! assembler that encodes [`MachineInstr`] sequences (from instruction
//! selection) into raw machine code bytes (`Vec<u8>`). It is the final stage
//! before the linker, converting abstract machine instructions into the
//! binary representation defined by the Intel x86-64 ISA.
//!
//! ## x86-64 Instruction Encoding Format
//!
//! Each encoded instruction follows this general layout:
//!
//! ```text
//! [Legacy Prefixes] [REX] [Opcode 1-3B] [ModR/M] [SIB] [Disp] [Imm]
//! ```
//!
//! - **Legacy prefixes**: Operand size override (0x66), repeat (F2/F3)
//! - **REX prefix** (0x40-0x4F): 64-bit operand size (W), register extensions (R/X/B)
//! - **Opcode**: 1-3 bytes identifying the instruction
//! - **ModR/M**: Addressing mode (mod), register operand (reg), r/m operand
//! - **SIB**: Scale-Index-Base for complex addressing
//! - **Displacement**: 8-bit or 32-bit signed offset
//! - **Immediate**: 8/16/32/64-bit constant operand
//!
//! ## Zero External Dependencies
//!
//! Only imports from `std` and internal crate modules. No external crates.

use std::collections::HashMap;

use crate::codegen::regalloc::PhysReg;
use crate::codegen::x86_64::abi::{is_xmm_reg, xmm_encoding};
use crate::codegen::x86_64::isel::{opcodes, CondCode};
use crate::codegen::{CodeGenError, MachineInstr, MachineOperand, Relocation, RelocationType};

// =========================================================================
// Public Helper Functions — Register Encoding
// =========================================================================

/// Returns the 3-bit register encoding index for a physical GPR register.
///
/// For GPRs (PhysReg 0-15), the low 3 bits of the PhysReg value give
/// the encoding index used in the ModR/M reg and r/m fields:
///   RAX=0, RCX=1, RDX=2, RBX=3, RSP=4, RBP=5, RSI=6, RDI=7
///   R8=0+REX, R9=1+REX, ... R15=7+REX
///
/// For XMM registers, use [`xmm_encoding`] from the ABI module instead.
#[inline]
pub fn reg_encoding(reg: PhysReg) -> u8 {
    (reg.0 as u8) & 0x07
}

/// Returns `true` if the GPR requires a REX extension bit (R, X, or B)
/// because it is one of the extended registers R8-R15.
///
/// This function is for GPRs only (PhysReg 0-15). For XMM registers,
/// the REX extension is determined by whether `xmm_encoding(reg) >= 8`.
#[inline]
pub fn needs_rex_extension(reg: PhysReg) -> bool {
    reg.0 >= 8 && reg.0 <= 15
}

// =========================================================================
// Private Helper Functions
// =========================================================================

/// Constructs a ModR/M byte from its three fields.
///
/// - `mod_bits` (2 bits): 11=register, 00=memory, 01=mem+disp8, 10=mem+disp32
/// - `reg` (3 bits): Register operand or opcode extension (/digit)
/// - `rm` (3 bits): Register or memory operand
#[inline]
fn modrm(mod_bits: u8, reg: u8, rm: u8) -> u8 {
    ((mod_bits & 0x03) << 6) | ((reg & 0x07) << 3) | (rm & 0x07)
}

/// Constructs a SIB (Scale-Index-Base) byte.
///
/// - `scale` (2 bits): 0=x1, 1=x2, 2=x4, 3=x8
/// - `index` (3 bits): Index register (4=no index, i.e. RSP encoding)
/// - `base` (3 bits): Base register (5=disp32-only when mod=00)
#[inline]
fn sib(scale: u8, index: u8, base: u8) -> u8 {
    ((scale & 0x03) << 6) | ((index & 0x07) << 3) | (base & 0x07)
}

/// Returns `true` if the value fits in a signed 8-bit immediate.
#[inline]
fn fits_in_i8(v: i32) -> bool {
    v >= -128 && v <= 127
}

/// Returns `true` if the value fits in a signed 32-bit immediate.
#[inline]
fn fits_in_i32(v: i64) -> bool {
    v >= (i32::MIN as i64) && v <= (i32::MAX as i64)
}

/// Maps a [`CondCode`] enum discriminant (stored as i64 in MachineOperand)
/// to the 4-bit x86-64 condition code used in JCC and SETCC encodings.
fn condcode_to_cc_byte(cc: i64) -> u8 {
    match cc as u8 {
        0 => 0x04,  // E  (ZF=1)
        1 => 0x05,  // NE (ZF=0)
        2 => 0x0C,  // L  (SF!=OF)
        3 => 0x0E,  // LE (ZF=1 or SF!=OF)
        4 => 0x0F,  // G  (ZF=0 and SF=OF)
        5 => 0x0D,  // GE (SF=OF)
        6 => 0x02,  // B  (CF=1)
        7 => 0x06,  // BE (CF=1 or ZF=1)
        8 => 0x07,  // A  (CF=0 and ZF=0)
        9 => 0x03,  // AE (CF=0)
        10 => 0x08, // S  (SF=1)
        11 => 0x09, // NS (SF=0)
        12 => 0x0A, // P  (PF=1)
        13 => 0x0B, // NP (PF=0)
        _ => 0x04,  // Default to E
    }
}

/// Converts a [`CondCode`] enum value directly to the x86-64 4-bit cc nibble.
#[allow(dead_code)]
fn condcode_enum_to_cc(cc: &CondCode) -> u8 {
    match cc {
        CondCode::E => 0x04,
        CondCode::NE => 0x05,
        CondCode::L => 0x0C,
        CondCode::LE => 0x0E,
        CondCode::G => 0x0F,
        CondCode::GE => 0x0D,
        CondCode::B => 0x02,
        CondCode::BE => 0x06,
        CondCode::A => 0x07,
        CondCode::AE => 0x03,
        CondCode::S => 0x08,
        CondCode::NS => 0x09,
        CondCode::P => 0x0A,
        CondCode::NP => 0x0B,
    }
}

// =========================================================================
// RexPrefix
// =========================================================================

/// REX prefix byte (0x40-0x4F) builder for x86-64 instructions.
struct RexPrefix {
    w: bool, // 64-bit operand size
    r: bool, // Extension of ModR/M reg field
    x: bool, // Extension of SIB index field
    b: bool, // Extension of ModR/M r/m or SIB base field
}

impl RexPrefix {
    fn none() -> Self {
        Self {
            w: false,
            r: false,
            x: false,
            b: false,
        }
    }

    fn encode(&self) -> Option<u8> {
        if self.w || self.r || self.x || self.b {
            Some(
                0x40 | (if self.w { 0x08 } else { 0 })
                    | (if self.r { 0x04 } else { 0 })
                    | (if self.x { 0x02 } else { 0 })
                    | (if self.b { 0x01 } else { 0 }),
            )
        } else {
            None
        }
    }

    #[allow(dead_code)]
    fn needs_emit(&self) -> bool {
        self.w || self.r || self.x || self.b
    }
}

// =========================================================================
// LabelFixup
// =========================================================================

/// Forward label reference to be patched after all instructions are encoded.
struct LabelFixup {
    code_offset: usize,
    label_id: u32,
    disp_size: u8,
}

// =========================================================================
// Displacement kind (used internally by emit_modrm_mem)
// =========================================================================

enum DispKind {
    None,
    Disp8,
    Disp32,
}

// =========================================================================
// EncodedFunction — output of encoding
// =========================================================================

/// Result of encoding a function's machine instructions into bytes.
pub struct EncodedFunction {
    /// Raw machine code bytes.
    pub code: Vec<u8>,
    /// Relocations for the linker.
    pub relocations: Vec<Relocation>,
    /// Label ID to byte offset mapping.
    pub labels: HashMap<u32, usize>,
}

// =========================================================================
// X86_64Encoder
// =========================================================================

/// x86-64 machine code encoder (integrated assembler).
///
/// Encodes [`MachineInstr`] sequences into raw machine code bytes for the
/// x86-64 architecture. Handles REX prefix generation, ModR/M and SIB
/// byte construction, displacement and immediate encoding, SSE prefix
/// bytes, label fixup resolution, and relocation emission.
pub struct X86_64Encoder {
    code: Vec<u8>,
    labels: HashMap<u32, usize>,
    fixups: Vec<LabelFixup>,
    relocations: Vec<Relocation>,
}

impl X86_64Encoder {
    /// Creates a new encoder with empty state.
    pub fn new() -> Self {
        Self {
            code: Vec::with_capacity(4096),
            labels: HashMap::new(),
            fixups: Vec::new(),
            relocations: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.code.clear();
        self.labels.clear();
        self.fixups.clear();
        self.relocations.clear();
    }

    // ----- Byte emission helpers -----

    #[inline]
    fn emit_byte(&mut self, b: u8) {
        self.code.push(b);
    }

    #[inline]
    fn emit_bytes(&mut self, bytes: &[u8]) {
        self.code.extend_from_slice(bytes);
    }

    #[inline]
    fn emit_u16_le(&mut self, v: u16) {
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    fn emit_u32_le(&mut self, v: u32) {
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    fn emit_u64_le(&mut self, v: u64) {
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    fn emit_i8(&mut self, v: i8) {
        self.code.push(v as u8);
    }

    #[inline]
    fn emit_i32_le(&mut self, v: i32) {
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    fn emit_rex(&mut self, prefix: &RexPrefix) {
        if let Some(byte) = prefix.encode() {
            self.emit_byte(byte);
        }
    }

    // ----- ModR/M and SIB emission -----

    #[inline]
    fn emit_modrm_rr(&mut self, reg_bits: u8, rm_bits: u8) {
        self.emit_byte(modrm(0x03, reg_bits, rm_bits));
    }

    /// Emits ModR/M (and optional SIB + displacement) for register-memory mode.
    fn emit_modrm_mem(&mut self, reg_bits: u8, base: PhysReg, offset: i32) {
        let base_enc = reg_encoding(base);
        let (mod_bits, disp_kind) = if offset == 0 && base_enc != 5 {
            (0x00u8, DispKind::None)
        } else if fits_in_i8(offset) {
            (0x01, DispKind::Disp8)
        } else {
            (0x02, DispKind::Disp32)
        };

        if base_enc == 4 {
            // RSP/R12 base requires SIB byte
            self.emit_byte(modrm(mod_bits, reg_bits, 0x04));
            self.emit_byte(sib(0x00, 0x04, base_enc));
        } else {
            self.emit_byte(modrm(mod_bits, reg_bits, base_enc));
        }

        match disp_kind {
            DispKind::None => {}
            DispKind::Disp8 => self.emit_i8(offset as i8),
            DispKind::Disp32 => self.emit_i32_le(offset),
        }
    }

    // ----- REX prefix computation -----

    fn compute_rex_gpr(w: bool, reg_op: PhysReg, rm_op: PhysReg) -> RexPrefix {
        RexPrefix {
            w,
            r: needs_rex_extension(reg_op),
            x: false,
            b: needs_rex_extension(rm_op),
        }
    }

    fn compute_rex_gpr_mem(w: bool, reg_op: PhysReg, base: PhysReg) -> RexPrefix {
        RexPrefix {
            w,
            r: needs_rex_extension(reg_op),
            x: false,
            b: needs_rex_extension(base),
        }
    }

    fn compute_rex_single(w: bool, rm_op: PhysReg) -> RexPrefix {
        RexPrefix {
            w,
            r: false,
            x: false,
            b: needs_rex_extension(rm_op),
        }
    }

    fn xmm_enc_info(reg: PhysReg) -> (u8, bool) {
        if !is_xmm_reg(reg) {
            panic!(
                "xmm_enc_info called with non-XMM register: PhysReg({})",
                reg.0
            );
        }
        let enc = xmm_encoding(reg);
        (enc & 0x07, enc >= 8)
    }

    // ----- Instruction encoding helpers -----

    fn encode_alu_rr(&mut self, opcode_byte: u8, dst: PhysReg, src: PhysReg) {
        let rex = Self::compute_rex_gpr(true, src, dst);
        self.emit_rex(&rex);
        self.emit_byte(opcode_byte);
        self.emit_modrm_rr(reg_encoding(src), reg_encoding(dst));
    }

    fn encode_alu_rr_32(&mut self, opcode_byte: u8, dst: PhysReg, src: PhysReg) {
        let rex = Self::compute_rex_gpr(false, src, dst);
        self.emit_rex(&rex);
        self.emit_byte(opcode_byte);
        self.emit_modrm_rr(reg_encoding(src), reg_encoding(dst));
    }

    fn encode_alu_ri(&mut self, ext: u8, dst: PhysReg, imm: i64) {
        let rex = Self::compute_rex_single(true, dst);
        self.emit_rex(&rex);
        let imm32 = imm as i32;
        if fits_in_i8(imm32) {
            self.emit_byte(0x83);
            self.emit_modrm_rr(ext, reg_encoding(dst));
            self.emit_i8(imm32 as i8);
        } else {
            self.emit_byte(0x81);
            self.emit_modrm_rr(ext, reg_encoding(dst));
            self.emit_i32_le(imm32);
        }
    }

    fn encode_unary_r(&mut self, opcode_byte: u8, ext: u8, reg: PhysReg) {
        let rex = Self::compute_rex_single(true, reg);
        self.emit_rex(&rex);
        self.emit_byte(opcode_byte);
        self.emit_modrm_rr(ext, reg_encoding(reg));
    }

    fn encode_sse_rr(&mut self, prefix: u8, opcode2: u8, dst: PhysReg, src: PhysReg) {
        let (dst_enc, dst_rex) = Self::xmm_enc_info(dst);
        let (src_enc, src_rex) = Self::xmm_enc_info(src);
        if prefix != 0 {
            self.emit_byte(prefix);
        }
        let rex = RexPrefix {
            w: false,
            r: dst_rex,
            x: false,
            b: src_rex,
        };
        self.emit_rex(&rex);
        self.emit_byte(0x0F);
        self.emit_byte(opcode2);
        self.emit_modrm_rr(dst_enc, src_enc);
    }

    fn encode_sse_gpr_to_xmm(
        &mut self,
        prefix: u8,
        opcode2: u8,
        dst_xmm: PhysReg,
        src_gpr: PhysReg,
    ) {
        let (dst_enc, dst_rex) = Self::xmm_enc_info(dst_xmm);
        self.emit_byte(prefix);
        let rex = RexPrefix {
            w: true,
            r: dst_rex,
            x: false,
            b: needs_rex_extension(src_gpr),
        };
        self.emit_rex(&rex);
        self.emit_byte(0x0F);
        self.emit_byte(opcode2);
        self.emit_modrm_rr(dst_enc, reg_encoding(src_gpr));
    }

    fn encode_sse_xmm_to_gpr(
        &mut self,
        prefix: u8,
        opcode2: u8,
        dst_gpr: PhysReg,
        src_xmm: PhysReg,
    ) {
        let (src_enc, src_rex) = Self::xmm_enc_info(src_xmm);
        self.emit_byte(prefix);
        let rex = RexPrefix {
            w: true,
            r: needs_rex_extension(dst_gpr),
            x: false,
            b: src_rex,
        };
        self.emit_rex(&rex);
        self.emit_byte(0x0F);
        self.emit_byte(opcode2);
        self.emit_modrm_rr(reg_encoding(dst_gpr), src_enc);
    }

    // ----- Label management -----

    fn define_label(&mut self, label_id: u32) {
        self.labels.insert(label_id, self.code.len());
    }

    fn emit_label_ref(&mut self, label_id: u32, disp_size: u8) {
        let code_offset = self.code.len();
        for _ in 0..disp_size {
            self.emit_byte(0x00);
        }
        self.fixups.push(LabelFixup {
            code_offset,
            label_id,
            disp_size,
        });
    }

    fn resolve_fixups(&mut self) -> Result<(), CodeGenError> {
        for fixup in &self.fixups {
            let target = self.labels.get(&fixup.label_id).copied().ok_or_else(|| {
                CodeGenError::EncodingError(format!("unresolved label: {}", fixup.label_id))
            })?;
            let rel = target as i64 - (fixup.code_offset as i64 + fixup.disp_size as i64);
            match fixup.disp_size {
                1 => {
                    if rel < -128 || rel > 127 {
                        return Err(CodeGenError::EncodingError(format!(
                            "label {} rel8 overflow: {}",
                            fixup.label_id, rel
                        )));
                    }
                    self.code[fixup.code_offset] = rel as i8 as u8;
                }
                4 => {
                    if rel < (i32::MIN as i64) || rel > (i32::MAX as i64) {
                        return Err(CodeGenError::EncodingError(format!(
                            "label {} rel32 overflow: {}",
                            fixup.label_id, rel
                        )));
                    }
                    let bytes = (rel as i32).to_le_bytes();
                    self.code[fixup.code_offset..fixup.code_offset + 4].copy_from_slice(&bytes);
                }
                _ => {
                    return Err(CodeGenError::EncodingError(format!(
                        "invalid fixup disp_size: {}",
                        fixup.disp_size
                    )))
                }
            }
        }
        Ok(())
    }

    fn emit_relocation(&mut self, symbol: &str, reloc_type: RelocationType, addend: i64) {
        self.relocations.push(Relocation {
            offset: self.code.len() as u64,
            symbol: symbol.to_string(),
            reloc_type,
            addend,
            section_index: 0,
        });
    }

    pub fn get_relocations(&self) -> &[Relocation] {
        &self.relocations
    }

    // ----- Operand extraction -----

    fn expect_reg(op: &MachineOperand, ctx: &str) -> Result<PhysReg, CodeGenError> {
        match op {
            MachineOperand::Register(r) => Ok(*r),
            _ => Err(CodeGenError::EncodingError(format!(
                "expected register for {}",
                ctx
            ))),
        }
    }
    fn expect_imm(op: &MachineOperand, ctx: &str) -> Result<i64, CodeGenError> {
        match op {
            MachineOperand::Immediate(v) => Ok(*v),
            _ => Err(CodeGenError::EncodingError(format!(
                "expected immediate for {}",
                ctx
            ))),
        }
    }
    fn expect_mem(op: &MachineOperand, ctx: &str) -> Result<(PhysReg, i32), CodeGenError> {
        match op {
            MachineOperand::Memory { base, offset } => Ok((*base, *offset)),
            _ => Err(CodeGenError::EncodingError(format!(
                "expected memory for {}",
                ctx
            ))),
        }
    }
    fn expect_label(op: &MachineOperand, ctx: &str) -> Result<u32, CodeGenError> {
        match op {
            MachineOperand::Label(id) => Ok(*id),
            _ => Err(CodeGenError::EncodingError(format!(
                "expected label for {}",
                ctx
            ))),
        }
    }
    fn expect_symbol(op: &MachineOperand, ctx: &str) -> Result<String, CodeGenError> {
        match op {
            MachineOperand::Symbol(s) => Ok(s.clone()),
            _ => Err(CodeGenError::EncodingError(format!(
                "expected symbol for {}",
                ctx
            ))),
        }
    }

    // =====================================================================
    // Main Instruction Encoding Dispatch
    // =====================================================================

    fn encode_instr(&mut self, instr: &MachineInstr) -> Result<(), CodeGenError> {
        let op = instr.opcode;
        let ops = &instr.operands;

        match op {
            // --- Integer Arithmetic ---
            opcodes::ADD_RR => {
                let dst = Self::expect_reg(&ops[0], "ADD_RR dst")?;
                let src = Self::expect_reg(&ops[1], "ADD_RR src")?;
                self.encode_alu_rr(0x01, dst, src);
            }
            opcodes::ADD_RI => {
                let dst = Self::expect_reg(&ops[0], "ADD_RI dst")?;
                let imm = Self::expect_imm(&ops[1], "ADD_RI imm")?;
                self.encode_alu_ri(0, dst, imm);
            }
            opcodes::ADD_RM => {
                let dst = Self::expect_reg(&ops[0], "ADD_RM dst")?;
                let (base, offset) = Self::expect_mem(&ops[1], "ADD_RM src")?;
                let rex = Self::compute_rex_gpr_mem(true, dst, base);
                self.emit_rex(&rex);
                self.emit_byte(0x03);
                self.emit_modrm_mem(reg_encoding(dst), base, offset);
            }
            opcodes::SUB_RR => {
                let dst = Self::expect_reg(&ops[0], "SUB_RR dst")?;
                let src = Self::expect_reg(&ops[1], "SUB_RR src")?;
                self.encode_alu_rr(0x29, dst, src);
            }
            opcodes::SUB_RI => {
                let dst = Self::expect_reg(&ops[0], "SUB_RI dst")?;
                let imm = Self::expect_imm(&ops[1], "SUB_RI imm")?;
                self.encode_alu_ri(5, dst, imm);
            }
            opcodes::SUB_RM => {
                let dst = Self::expect_reg(&ops[0], "SUB_RM dst")?;
                let (base, offset) = Self::expect_mem(&ops[1], "SUB_RM src")?;
                let rex = Self::compute_rex_gpr_mem(true, dst, base);
                self.emit_rex(&rex);
                self.emit_byte(0x2B);
                self.emit_modrm_mem(reg_encoding(dst), base, offset);
            }
            opcodes::IMUL_RR => {
                let dst = Self::expect_reg(&ops[0], "IMUL_RR dst")?;
                let src = Self::expect_reg(&ops[1], "IMUL_RR src")?;
                let rex = Self::compute_rex_gpr(true, dst, src);
                self.emit_rex(&rex);
                self.emit_bytes(&[0x0F, 0xAF]);
                self.emit_modrm_rr(reg_encoding(dst), reg_encoding(src));
            }
            opcodes::IMUL_RI => {
                let dst = Self::expect_reg(&ops[0], "IMUL_RI dst")?;
                let (src, imm_idx) = if ops.len() > 2 {
                    (Self::expect_reg(&ops[1], "IMUL_RI src")?, 2)
                } else {
                    (dst, 1)
                };
                let imm = Self::expect_imm(&ops[imm_idx], "IMUL_RI imm")?;
                let rex = Self::compute_rex_gpr(true, dst, src);
                self.emit_rex(&rex);
                let imm32 = imm as i32;
                if fits_in_i8(imm32) {
                    self.emit_byte(0x6B);
                    self.emit_modrm_rr(reg_encoding(dst), reg_encoding(src));
                    self.emit_i8(imm32 as i8);
                } else {
                    self.emit_byte(0x69);
                    self.emit_modrm_rr(reg_encoding(dst), reg_encoding(src));
                    self.emit_i32_le(imm32);
                }
            }
            opcodes::IDIV_R => {
                let reg = Self::expect_reg(&ops[0], "IDIV_R")?;
                self.encode_unary_r(0xF7, 7, reg);
            }
            opcodes::DIV_R => {
                let reg = Self::expect_reg(&ops[0], "DIV_R")?;
                self.encode_unary_r(0xF7, 6, reg);
            }

            // --- Bitwise ---
            opcodes::AND_RR => {
                let dst = Self::expect_reg(&ops[0], "AND_RR dst")?;
                let src = Self::expect_reg(&ops[1], "AND_RR src")?;
                self.encode_alu_rr(0x21, dst, src);
            }
            opcodes::AND_RI => {
                let dst = Self::expect_reg(&ops[0], "AND_RI dst")?;
                let imm = Self::expect_imm(&ops[1], "AND_RI imm")?;
                self.encode_alu_ri(4, dst, imm);
            }
            opcodes::OR_RR => {
                let dst = Self::expect_reg(&ops[0], "OR_RR dst")?;
                let src = Self::expect_reg(&ops[1], "OR_RR src")?;
                self.encode_alu_rr(0x09, dst, src);
            }
            opcodes::OR_RI => {
                let dst = Self::expect_reg(&ops[0], "OR_RI dst")?;
                let imm = Self::expect_imm(&ops[1], "OR_RI imm")?;
                self.encode_alu_ri(1, dst, imm);
            }
            opcodes::XOR_RR => {
                let dst = Self::expect_reg(&ops[0], "XOR_RR dst")?;
                let src = Self::expect_reg(&ops[1], "XOR_RR src")?;
                if dst == src {
                    self.encode_alu_rr_32(0x31, dst, src);
                } else {
                    self.encode_alu_rr(0x31, dst, src);
                }
            }
            opcodes::XOR_RI => {
                let dst = Self::expect_reg(&ops[0], "XOR_RI dst")?;
                let imm = Self::expect_imm(&ops[1], "XOR_RI imm")?;
                self.encode_alu_ri(6, dst, imm);
            }
            opcodes::SHL_RI => {
                let dst = Self::expect_reg(&ops[0], "SHL_RI")?;
                let imm = Self::expect_imm(&ops[1], "SHL_RI imm")?;
                let rex = Self::compute_rex_single(true, dst);
                self.emit_rex(&rex);
                self.emit_byte(0xC1);
                self.emit_modrm_rr(4, reg_encoding(dst));
                self.emit_i8(imm as i8);
            }
            opcodes::SHR_RI => {
                let dst = Self::expect_reg(&ops[0], "SHR_RI")?;
                let imm = Self::expect_imm(&ops[1], "SHR_RI imm")?;
                let rex = Self::compute_rex_single(true, dst);
                self.emit_rex(&rex);
                self.emit_byte(0xC1);
                self.emit_modrm_rr(5, reg_encoding(dst));
                self.emit_i8(imm as i8);
            }
            opcodes::SAR_RI => {
                let dst = Self::expect_reg(&ops[0], "SAR_RI")?;
                let imm = Self::expect_imm(&ops[1], "SAR_RI imm")?;
                let rex = Self::compute_rex_single(true, dst);
                self.emit_rex(&rex);
                self.emit_byte(0xC1);
                self.emit_modrm_rr(7, reg_encoding(dst));
                self.emit_i8(imm as i8);
            }
            opcodes::SHL_RCL => {
                let d = Self::expect_reg(&ops[0], "SHL_RCL")?;
                self.encode_unary_r(0xD3, 4, d);
            }
            opcodes::SHR_RCL => {
                let d = Self::expect_reg(&ops[0], "SHR_RCL")?;
                self.encode_unary_r(0xD3, 5, d);
            }
            opcodes::SAR_RCL => {
                let d = Self::expect_reg(&ops[0], "SAR_RCL")?;
                self.encode_unary_r(0xD3, 7, d);
            }
            opcodes::NOT_R => {
                let r = Self::expect_reg(&ops[0], "NOT_R")?;
                self.encode_unary_r(0xF7, 2, r);
            }
            opcodes::NEG_R => {
                let r = Self::expect_reg(&ops[0], "NEG_R")?;
                self.encode_unary_r(0xF7, 3, r);
            }

            // --- Data Movement ---
            opcodes::MOV_RR => {
                let dst = Self::expect_reg(&ops[0], "MOV_RR dst")?;
                // Handle the case where ISel emits MOV_RR with a non-register
                // source operand — transparently redirect to the appropriate
                // encoding for the actual operand type.
                match &ops[1] {
                    MachineOperand::Register(src) => {
                        self.encode_alu_rr(0x89, dst, *src);
                    }
                    MachineOperand::Immediate(imm) => {
                        // Redirect to MOV_RI encoding: mov reg, imm
                        let imm = *imm;
                        if imm >= 0 && imm <= 0x7FFFFFFF {
                            let rex = Self::compute_rex_single(false, dst);
                            self.emit_rex(&rex);
                            self.emit_byte(0xB8 + reg_encoding(dst));
                            self.emit_i32_le(imm as i32);
                        } else if fits_in_i32(imm) {
                            let rex = Self::compute_rex_single(true, dst);
                            self.emit_rex(&rex);
                            self.emit_byte(0xC7);
                            self.emit_modrm_rr(0, reg_encoding(dst));
                            self.emit_i32_le(imm as i32);
                        } else {
                            let rex = Self::compute_rex_single(true, dst);
                            self.emit_rex(&rex);
                            self.emit_byte(0xB8 + reg_encoding(dst));
                            self.emit_u64_le(imm as u64);
                        }
                    }
                    MachineOperand::Memory { base, offset, .. } => {
                        // Memory operand → emit MOV reg, [base+offset]
                        let base_reg = *base;
                        let off = *offset;
                        let rex = Self::compute_rex_gpr_mem(true, dst, base_reg);
                        self.emit_rex(&rex);
                        self.emit_byte(0x8B);
                        self.emit_modrm_mem(reg_encoding(dst), base_reg, off);
                    }
                    MachineOperand::Symbol(sym) => {
                        // Symbol → movabs reg, imm64 with relocation
                        let rex = Self::compute_rex_single(true, dst);
                        self.emit_rex(&rex);
                        self.emit_byte(0xB8 + reg_encoding(dst));
                        self.emit_relocation(sym, RelocationType::X86_64_64, 0);
                        self.emit_u64_le(0);
                    }
                    MachineOperand::Label(_) => {
                        return Err(CodeGenError::EncodingError(
                            "MOV_RR: label operand not supported".into(),
                        ));
                    }
                }
            }
            opcodes::MOV_RI => {
                let dst = Self::expect_reg(&ops[0], "MOV_RI dst")?;
                match &ops[1] {
                    MachineOperand::Immediate(imm) => {
                        let imm = *imm;
                        if imm >= 0 && imm <= 0x7FFFFFFF {
                            // Non-negative value fits in 32 bits: use mov r32, imm32
                            // which zero-extends to 64-bit (canonical, shorter encoding).
                            let rex = Self::compute_rex_single(false, dst);
                            self.emit_rex(&rex);
                            self.emit_byte(0xB8 + reg_encoding(dst));
                            self.emit_i32_le(imm as i32);
                        } else if fits_in_i32(imm) {
                            // Negative 32-bit: use mov r64, sign-extended-imm32 (REX.W + C7 /0)
                            let rex = Self::compute_rex_single(true, dst);
                            self.emit_rex(&rex);
                            self.emit_byte(0xC7);
                            self.emit_modrm_rr(0, reg_encoding(dst));
                            self.emit_i32_le(imm as i32);
                        } else {
                            // Full 64-bit immediate: movabs r64, imm64
                            let rex = Self::compute_rex_single(true, dst);
                            self.emit_rex(&rex);
                            self.emit_byte(0xB8 + reg_encoding(dst));
                            self.emit_u64_le(imm as u64);
                        }
                    }
                    MachineOperand::Symbol(sym) => {
                        // movabs reg, imm64 with R_X86_64_64 absolute relocation
                        let rex = Self::compute_rex_single(true, dst);
                        self.emit_rex(&rex);
                        self.emit_byte(0xB8 + reg_encoding(dst));
                        self.emit_relocation(sym, RelocationType::X86_64_64, 0);
                        self.emit_u64_le(0); // placeholder for linker
                    }
                    _ => {
                        return Err(CodeGenError::EncodingError(
                            "MOV_RI: second operand must be immediate or symbol".to_string(),
                        ));
                    }
                }
            }
            opcodes::MOV_RM => {
                let dst = Self::expect_reg(&ops[0], "MOV_RM dst")?;
                match &ops[1] {
                    MachineOperand::Memory { base, offset } => {
                        let rex = Self::compute_rex_gpr_mem(true, dst, *base);
                        self.emit_rex(&rex);
                        self.emit_byte(0x8B);
                        self.emit_modrm_mem(reg_encoding(dst), *base, *offset);
                    }
                    MachineOperand::Symbol(sym) => {
                        // RIP-relative MOV: mov reg, [rip + disp32] with GOTPCREL relocation
                        let rex = RexPrefix {
                            w: true,
                            r: needs_rex_extension(dst),
                            x: false,
                            b: false,
                        };
                        self.emit_rex(&rex);
                        self.emit_byte(0x8B);
                        // ModRM: mod=00, reg=dst, rm=5 (RIP-relative)
                        self.emit_byte(modrm(0x00, reg_encoding(dst), 0x05));
                        let reloc_type = if sym.contains("@GOTPCREL") {
                            RelocationType::X86_64_GOTPCREL
                        } else {
                            RelocationType::X86_64_PC32
                        };
                        self.emit_relocation(sym, reloc_type, -4);
                        self.emit_i32_le(0); // placeholder for linker
                    }
                    _ => {
                        return Err(CodeGenError::EncodingError(
                            "MOV_RM: second operand must be memory or symbol".to_string(),
                        ))
                    }
                }
            }
            opcodes::MOV_MR => {
                let (base, offset) = Self::expect_mem(&ops[0], "MOV_MR dst")?;
                let src = Self::expect_reg(&ops[1], "MOV_MR src")?;
                let rex = Self::compute_rex_gpr_mem(true, src, base);
                self.emit_rex(&rex);
                self.emit_byte(0x89);
                self.emit_modrm_mem(reg_encoding(src), base, offset);
            }
            opcodes::MOV_MI => {
                let (base, offset) = Self::expect_mem(&ops[0], "MOV_MI dst")?;
                let imm = Self::expect_imm(&ops[1], "MOV_MI imm")?;
                let rex = RexPrefix {
                    w: true,
                    r: false,
                    x: false,
                    b: needs_rex_extension(base),
                };
                self.emit_rex(&rex);
                self.emit_byte(0xC7);
                self.emit_modrm_mem(0, base, offset);
                self.emit_i32_le(imm as i32);
            }
            opcodes::MOVSX => {
                let dst = Self::expect_reg(&ops[0], "MOVSX dst")?;
                let src = Self::expect_reg(&ops[1], "MOVSX src")?;
                let size = if ops.len() > 2 {
                    Self::expect_imm(&ops[2], "MOVSX sz")? as u8
                } else {
                    4
                };
                match size {
                    1 => {
                        let rex = Self::compute_rex_gpr(true, dst, src);
                        self.emit_rex(&rex);
                        self.emit_bytes(&[0x0F, 0xBE]);
                        self.emit_modrm_rr(reg_encoding(dst), reg_encoding(src));
                    }
                    2 => {
                        let rex = Self::compute_rex_gpr(true, dst, src);
                        self.emit_rex(&rex);
                        self.emit_bytes(&[0x0F, 0xBF]);
                        self.emit_modrm_rr(reg_encoding(dst), reg_encoding(src));
                    }
                    4 => {
                        let rex = Self::compute_rex_gpr(true, dst, src);
                        self.emit_rex(&rex);
                        self.emit_byte(0x63);
                        self.emit_modrm_rr(reg_encoding(dst), reg_encoding(src));
                    }
                    _ => {
                        return Err(CodeGenError::EncodingError(format!(
                            "unsupported MOVSX size: {}",
                            size
                        )))
                    }
                }
            }
            opcodes::MOVZX => {
                let dst = Self::expect_reg(&ops[0], "MOVZX dst")?;
                let src = Self::expect_reg(&ops[1], "MOVZX src")?;
                let size = if ops.len() > 2 {
                    Self::expect_imm(&ops[2], "MOVZX sz")? as u8
                } else {
                    1
                };
                match size {
                    1 => {
                        // For byte source operand: encodings 4-7 without REX
                        // map to AH/CH/DH/BH (high bytes). With REX prefix,
                        // they map to SPL/BPL/SIL/DIL (low bytes). Force REX
                        // when source encoding >= 4 to ensure correct byte
                        // register interpretation (same issue as SETCC).
                        let mut rex = Self::compute_rex_gpr(false, dst, src);
                        if reg_encoding(src) >= 4 && !rex.w && !rex.r && !rex.x && !rex.b {
                            // Force a bare REX prefix (0x40) to switch byte
                            // register interpretation to the uniform low-byte set
                            self.emit_byte(0x40);
                        } else {
                            self.emit_rex(&rex);
                        }
                        self.emit_bytes(&[0x0F, 0xB6]);
                        self.emit_modrm_rr(reg_encoding(dst), reg_encoding(src));
                    }
                    2 => {
                        let rex = Self::compute_rex_gpr(false, dst, src);
                        self.emit_rex(&rex);
                        self.emit_bytes(&[0x0F, 0xB7]);
                        self.emit_modrm_rr(reg_encoding(dst), reg_encoding(src));
                    }
                    4 => {
                        let rex = Self::compute_rex_gpr(false, dst, src);
                        self.emit_rex(&rex);
                        self.emit_byte(0x8B);
                        self.emit_modrm_rr(reg_encoding(dst), reg_encoding(src));
                    }
                    _ => {
                        return Err(CodeGenError::EncodingError(format!(
                            "unsupported MOVZX size: {}",
                            size
                        )))
                    }
                }
            }
            opcodes::LEA => {
                let dst = Self::expect_reg(&ops[0], "LEA dst")?;
                match &ops[1] {
                    MachineOperand::Memory { base, offset } => {
                        let rex = Self::compute_rex_gpr_mem(true, dst, *base);
                        self.emit_rex(&rex);
                        self.emit_byte(0x8D);
                        self.emit_modrm_mem(reg_encoding(dst), *base, *offset);
                    }
                    MachineOperand::Symbol(sym) => {
                        // RIP-relative LEA: lea reg, [rip + disp32]
                        // Encoding: REX.W 8D /r with ModRM mod=00 rm=101 (RIP-relative)
                        let rex = RexPrefix {
                            w: true,
                            r: needs_rex_extension(dst),
                            x: false,
                            b: false,
                        };
                        self.emit_rex(&rex);
                        self.emit_byte(0x8D);
                        // ModRM: mod=00, reg=dst, rm=5 (RIP-relative)
                        self.emit_byte(modrm(0x00, reg_encoding(dst), 0x05));
                        self.emit_relocation(sym, RelocationType::X86_64_PC32, -4);
                        self.emit_i32_le(0); // placeholder for linker
                    }
                    _ => {
                        return Err(CodeGenError::EncodingError(
                            "LEA src must be memory or symbol".to_string(),
                        ))
                    }
                }
            }

            // --- Comparisons and Conditions ---
            opcodes::CMP_RR => {
                let l = Self::expect_reg(&ops[0], "CMP_RR")?;
                let r = Self::expect_reg(&ops[1], "CMP_RR")?;
                self.encode_alu_rr(0x39, l, r);
            }
            opcodes::CMP_RI => {
                let l = Self::expect_reg(&ops[0], "CMP_RI")?;
                let imm = Self::expect_imm(&ops[1], "CMP_RI")?;
                self.encode_alu_ri(7, l, imm);
            }
            opcodes::TEST_RR => {
                let l = Self::expect_reg(&ops[0], "TEST_RR")?;
                let r = Self::expect_reg(&ops[1], "TEST_RR")?;
                self.encode_alu_rr(0x85, l, r);
            }
            opcodes::TEST_RI => {
                let reg = Self::expect_reg(&ops[0], "TEST_RI")?;
                let imm = Self::expect_imm(&ops[1], "TEST_RI")?;
                let rex = Self::compute_rex_single(true, reg);
                self.emit_rex(&rex);
                self.emit_byte(0xF7);
                self.emit_modrm_rr(0, reg_encoding(reg));
                self.emit_i32_le(imm as i32);
            }
            opcodes::SETCC => {
                let cc = Self::expect_imm(&ops[0], "SETCC cc")?;
                let dst = Self::expect_reg(&ops[1], "SETCC dst")?;
                let cc_byte = condcode_to_cc_byte(cc);
                // SAFETY NOTE: Without a REX prefix, byte register encodings
                // 4-7 map to AH/CH/DH/BH (high bytes of AX/CX/DX/BX), which
                // would corrupt callee-saved registers like RBX. With a REX
                // prefix (even 0x40), encodings 4-7 map to SPL/BPL/SIL/DIL
                // (low bytes of RSP/RBP/RSI/RDI). We must force a REX prefix
                // whenever the register encoding is >= 4 to ensure correct
                // byte register addressing.
                let needs_rex_b = needs_rex_extension(dst);
                let force_rex = reg_encoding(dst) >= 4;
                if needs_rex_b || force_rex {
                    let rex_byte = 0x40u8 | (if needs_rex_b { 0x01 } else { 0 });
                    self.emit_byte(rex_byte);
                }
                self.emit_byte(0x0F);
                self.emit_byte(0x90 + cc_byte);
                self.emit_modrm_rr(0, reg_encoding(dst));
            }
            opcodes::CMOVCC => {
                let cc = Self::expect_imm(&ops[0], "CMOVCC cc")?;
                let dst = Self::expect_reg(&ops[1], "CMOVCC dst")?;
                let src = Self::expect_reg(&ops[2], "CMOVCC src")?;
                let cc_byte = condcode_to_cc_byte(cc);
                let rex = Self::compute_rex_gpr(true, dst, src);
                self.emit_rex(&rex);
                self.emit_byte(0x0F);
                self.emit_byte(0x40 + cc_byte);
                self.emit_modrm_rr(reg_encoding(dst), reg_encoding(src));
            }

            // --- Control Flow ---
            opcodes::JMP => {
                let label = Self::expect_label(&ops[0], "JMP")?;
                self.emit_byte(0xE9);
                self.emit_label_ref(label, 4);
            }
            opcodes::JCC => {
                let cc = Self::expect_imm(&ops[0], "JCC cc")?;
                let label = Self::expect_label(&ops[1], "JCC label")?;
                let cc_byte = condcode_to_cc_byte(cc);
                self.emit_byte(0x0F);
                self.emit_byte(0x80 + cc_byte);
                self.emit_label_ref(label, 4);
            }
            opcodes::CALL => {
                let sym = Self::expect_symbol(&ops[0], "CALL")?;
                self.emit_byte(0xE8);
                self.emit_relocation(&sym, RelocationType::X86_64_PLT32, -4);
                self.emit_i32_le(0);
            }
            opcodes::CALL_R => {
                let reg = Self::expect_reg(&ops[0], "CALL_R")?;
                let rex = Self::compute_rex_single(false, reg);
                self.emit_rex(&rex);
                self.emit_byte(0xFF);
                self.emit_modrm_rr(2, reg_encoding(reg));
            }
            opcodes::RET => {
                self.emit_byte(0xC3);
            }

            // --- Stack ---
            opcodes::PUSH => {
                match &ops[0] {
                    MachineOperand::Register(reg) => {
                        let reg = *reg;
                        if needs_rex_extension(reg) {
                            self.emit_byte(0x41);
                        }
                        self.emit_byte(0x50 + reg_encoding(reg));
                    }
                    MachineOperand::Memory { base, offset } => {
                        // PUSH r/m64: FF /6
                        let base = *base;
                        let offset = *offset;
                        let need_rex = needs_rex_extension(base);
                        if need_rex {
                            self.emit_byte(0x41);
                        }
                        self.emit_byte(0xFF);
                        self.emit_modrm_mem(6, base, offset);
                    }
                    MachineOperand::Immediate(val) => {
                        let val = *val;
                        if val >= -128 && val <= 127 {
                            // PUSH imm8: 6A ib
                            self.emit_byte(0x6A);
                            self.emit_byte(val as u8);
                        } else {
                            // PUSH imm32: 68 id
                            self.emit_byte(0x68);
                            let bytes = (val as u32).to_le_bytes();
                            for b in &bytes {
                                self.emit_byte(*b);
                            }
                        }
                    }
                    other => {
                        return Err(CodeGenError::EncodingError(format!(
                            "unsupported operand for PUSH: {:?}",
                            other
                        )));
                    }
                }
            }
            opcodes::POP => {
                let reg = Self::expect_reg(&ops[0], "POP")?;
                if needs_rex_extension(reg) {
                    self.emit_byte(0x41);
                }
                self.emit_byte(0x58 + reg_encoding(reg));
            }

            // --- Sized Memory Ops ---
            opcodes::LOAD8 => {
                let dst = Self::expect_reg(&ops[0], "LOAD8")?;
                let (base, offset) = Self::expect_mem(&ops[1], "LOAD8")?;
                let rex = Self::compute_rex_gpr_mem(false, dst, base);
                self.emit_rex(&rex);
                self.emit_bytes(&[0x0F, 0xB6]);
                self.emit_modrm_mem(reg_encoding(dst), base, offset);
            }
            opcodes::LOAD16 => {
                let dst = Self::expect_reg(&ops[0], "LOAD16")?;
                let (base, offset) = Self::expect_mem(&ops[1], "LOAD16")?;
                let rex = Self::compute_rex_gpr_mem(false, dst, base);
                self.emit_rex(&rex);
                self.emit_bytes(&[0x0F, 0xB7]);
                self.emit_modrm_mem(reg_encoding(dst), base, offset);
            }
            opcodes::LOAD32 => {
                // Use MOVSXD (sign-extend 32→64) instead of plain MOV r32,m32.
                // This ensures signed 32-bit values like `int x = -5;` are
                // correctly represented in 64-bit registers, so subsequent
                // 64-bit CMP/arithmetic instructions produce correct signed
                // results.  MOVSXD preserves both signed and unsigned ordering
                // when values are later compared as 64-bit quantities.
                let dst = Self::expect_reg(&ops[0], "LOAD32")?;
                let (base, offset) = Self::expect_mem(&ops[1], "LOAD32")?;
                let rex = Self::compute_rex_gpr_mem(true, dst, base);
                self.emit_rex(&rex);
                self.emit_byte(0x63);
                self.emit_modrm_mem(reg_encoding(dst), base, offset);
            }
            opcodes::LOAD64 => {
                let dst = Self::expect_reg(&ops[0], "LOAD64")?;
                let (base, offset) = Self::expect_mem(&ops[1], "LOAD64")?;
                let rex = Self::compute_rex_gpr_mem(true, dst, base);
                self.emit_rex(&rex);
                self.emit_byte(0x8B);
                self.emit_modrm_mem(reg_encoding(dst), base, offset);
            }
            opcodes::STORE8 => {
                let (base, offset) = Self::expect_mem(&ops[0], "STORE8")?;
                let src = Self::expect_reg(&ops[1], "STORE8")?;
                let rex = RexPrefix {
                    w: false,
                    r: needs_rex_extension(src),
                    x: false,
                    b: needs_rex_extension(base),
                };
                self.emit_rex(&rex);
                self.emit_byte(0x88);
                self.emit_modrm_mem(reg_encoding(src), base, offset);
            }
            opcodes::STORE16 => {
                let (base, offset) = Self::expect_mem(&ops[0], "STORE16")?;
                let src = Self::expect_reg(&ops[1], "STORE16")?;
                self.emit_byte(0x66);
                let rex = Self::compute_rex_gpr_mem(false, src, base);
                self.emit_rex(&rex);
                self.emit_byte(0x89);
                self.emit_modrm_mem(reg_encoding(src), base, offset);
            }
            opcodes::STORE32 => {
                let (base, offset) = Self::expect_mem(&ops[0], "STORE32")?;
                let src = Self::expect_reg(&ops[1], "STORE32")?;
                let rex = Self::compute_rex_gpr_mem(false, src, base);
                self.emit_rex(&rex);
                self.emit_byte(0x89);
                self.emit_modrm_mem(reg_encoding(src), base, offset);
            }
            opcodes::STORE64 => {
                let (base, offset) = Self::expect_mem(&ops[0], "STORE64")?;
                let src = Self::expect_reg(&ops[1], "STORE64")?;
                let rex = Self::compute_rex_gpr_mem(true, src, base);
                self.emit_rex(&rex);
                self.emit_byte(0x89);
                self.emit_modrm_mem(reg_encoding(src), base, offset);
            }

            // --- SSE Floating-Point ---
            opcodes::ADDSS => {
                let d = Self::expect_reg(&ops[0], "ADDSS")?;
                let s = Self::expect_reg(&ops[1], "ADDSS")?;
                self.encode_sse_rr(0xF3, 0x58, d, s);
            }
            opcodes::ADDSD => {
                let d = Self::expect_reg(&ops[0], "ADDSD")?;
                let s = Self::expect_reg(&ops[1], "ADDSD")?;
                self.encode_sse_rr(0xF2, 0x58, d, s);
            }
            opcodes::SUBSS => {
                let d = Self::expect_reg(&ops[0], "SUBSS")?;
                let s = Self::expect_reg(&ops[1], "SUBSS")?;
                self.encode_sse_rr(0xF3, 0x5C, d, s);
            }
            opcodes::SUBSD => {
                let d = Self::expect_reg(&ops[0], "SUBSD")?;
                let s = Self::expect_reg(&ops[1], "SUBSD")?;
                self.encode_sse_rr(0xF2, 0x5C, d, s);
            }
            opcodes::MULSS => {
                let d = Self::expect_reg(&ops[0], "MULSS")?;
                let s = Self::expect_reg(&ops[1], "MULSS")?;
                self.encode_sse_rr(0xF3, 0x59, d, s);
            }
            opcodes::MULSD => {
                let d = Self::expect_reg(&ops[0], "MULSD")?;
                let s = Self::expect_reg(&ops[1], "MULSD")?;
                self.encode_sse_rr(0xF2, 0x59, d, s);
            }
            opcodes::DIVSS => {
                let d = Self::expect_reg(&ops[0], "DIVSS")?;
                let s = Self::expect_reg(&ops[1], "DIVSS")?;
                self.encode_sse_rr(0xF3, 0x5E, d, s);
            }
            opcodes::DIVSD => {
                let d = Self::expect_reg(&ops[0], "DIVSD")?;
                let s = Self::expect_reg(&ops[1], "DIVSD")?;
                self.encode_sse_rr(0xF2, 0x5E, d, s);
            }
            opcodes::MOVSS => {
                // MOVSS has three forms:
                //   xmm, xmm -> F3 0F 10 /r
                //   xmm, mem -> F3 0F 10 /r (modrm memory)
                //   mem, xmm -> F3 0F 11 /r (modrm memory)
                match (&ops[0], &ops[1]) {
                    (MachineOperand::Register(d), MachineOperand::Register(s)) => {
                        self.encode_sse_rr(0xF3, 0x10, *d, *s);
                    }
                    (MachineOperand::Register(d), MachineOperand::Memory { base, offset }) => {
                        // MOVSS xmm, [base+offset]
                        self.emit_byte(0xF3);
                        let need_rex = needs_rex_extension(*d) || needs_rex_extension(*base);
                        if need_rex {
                            let rex = RexPrefix {
                                w: false,
                                r: needs_rex_extension(*d),
                                x: false,
                                b: needs_rex_extension(*base),
                            };
                            self.emit_rex(&rex);
                        }
                        self.emit_byte(0x0F);
                        self.emit_byte(0x10);
                        self.emit_modrm_mem(reg_encoding(*d), *base, *offset);
                    }
                    (MachineOperand::Memory { base, offset }, MachineOperand::Register(s)) => {
                        // MOVSS [base+offset], xmm
                        self.emit_byte(0xF3);
                        let need_rex = needs_rex_extension(*s) || needs_rex_extension(*base);
                        if need_rex {
                            let rex = RexPrefix {
                                w: false,
                                r: needs_rex_extension(*s),
                                x: false,
                                b: needs_rex_extension(*base),
                            };
                            self.emit_rex(&rex);
                        }
                        self.emit_byte(0x0F);
                        self.emit_byte(0x11);
                        self.emit_modrm_mem(reg_encoding(*s), *base, *offset);
                    }
                    _ => {
                        return Err(CodeGenError::EncodingError(
                            "MOVSS: unsupported operand combination".to_string(),
                        ))
                    }
                }
            }
            opcodes::MOVSD => {
                // MOVSD has three forms (same structure as MOVSS but prefix F2):
                //   xmm, xmm -> F2 0F 10 /r
                //   xmm, mem -> F2 0F 10 /r (modrm memory)
                //   mem, xmm -> F2 0F 11 /r (modrm memory)
                match (&ops[0], &ops[1]) {
                    (MachineOperand::Register(d), MachineOperand::Register(s)) => {
                        self.encode_sse_rr(0xF2, 0x10, *d, *s);
                    }
                    (MachineOperand::Register(d), MachineOperand::Memory { base, offset }) => {
                        // MOVSD xmm, [base+offset]
                        self.emit_byte(0xF2);
                        let need_rex = needs_rex_extension(*d) || needs_rex_extension(*base);
                        if need_rex {
                            let rex = RexPrefix {
                                w: false,
                                r: needs_rex_extension(*d),
                                x: false,
                                b: needs_rex_extension(*base),
                            };
                            self.emit_rex(&rex);
                        }
                        self.emit_byte(0x0F);
                        self.emit_byte(0x10);
                        self.emit_modrm_mem(reg_encoding(*d), *base, *offset);
                    }
                    (MachineOperand::Memory { base, offset }, MachineOperand::Register(s)) => {
                        // MOVSD [base+offset], xmm
                        self.emit_byte(0xF2);
                        let need_rex = needs_rex_extension(*s) || needs_rex_extension(*base);
                        if need_rex {
                            let rex = RexPrefix {
                                w: false,
                                r: needs_rex_extension(*s),
                                x: false,
                                b: needs_rex_extension(*base),
                            };
                            self.emit_rex(&rex);
                        }
                        self.emit_byte(0x0F);
                        self.emit_byte(0x11);
                        self.emit_modrm_mem(reg_encoding(*s), *base, *offset);
                    }
                    _ => {
                        return Err(CodeGenError::EncodingError(
                            "MOVSD: unsupported operand combination".to_string(),
                        ))
                    }
                }
            }
            opcodes::UCOMISS => {
                let d = Self::expect_reg(&ops[0], "UCOMISS")?;
                let s = Self::expect_reg(&ops[1], "UCOMISS")?;
                self.encode_sse_rr(0x00, 0x2E, d, s);
            }
            opcodes::UCOMISD => {
                let d = Self::expect_reg(&ops[0], "UCOMISD")?;
                let s = Self::expect_reg(&ops[1], "UCOMISD")?;
                self.encode_sse_rr(0x66, 0x2E, d, s);
            }
            opcodes::CVTSS2SD => {
                let d = Self::expect_reg(&ops[0], "CVTSS2SD")?;
                let s = Self::expect_reg(&ops[1], "CVTSS2SD")?;
                self.encode_sse_rr(0xF3, 0x5A, d, s);
            }
            opcodes::CVTSD2SS => {
                let d = Self::expect_reg(&ops[0], "CVTSD2SS")?;
                let s = Self::expect_reg(&ops[1], "CVTSD2SS")?;
                self.encode_sse_rr(0xF2, 0x5A, d, s);
            }
            opcodes::CVTSI2SS => {
                let d = Self::expect_reg(&ops[0], "CVTSI2SS")?;
                let s = Self::expect_reg(&ops[1], "CVTSI2SS")?;
                self.encode_sse_gpr_to_xmm(0xF3, 0x2A, d, s);
            }
            opcodes::CVTSI2SD => {
                let d = Self::expect_reg(&ops[0], "CVTSI2SD")?;
                let s = Self::expect_reg(&ops[1], "CVTSI2SD")?;
                self.encode_sse_gpr_to_xmm(0xF2, 0x2A, d, s);
            }
            opcodes::CVTTSS2SI => {
                let d = Self::expect_reg(&ops[0], "CVTTSS2SI")?;
                let s = Self::expect_reg(&ops[1], "CVTTSS2SI")?;
                self.encode_sse_xmm_to_gpr(0xF3, 0x2C, d, s);
            }
            opcodes::CVTTSD2SI => {
                let d = Self::expect_reg(&ops[0], "CVTTSD2SI")?;
                let s = Self::expect_reg(&ops[1], "CVTTSD2SI")?;
                self.encode_sse_xmm_to_gpr(0xF2, 0x2C, d, s);
            }

            // --- Extension / Conversion ---
            opcodes::CDQ => {
                self.emit_byte(0x99);
            }
            opcodes::CQO => {
                self.emit_byte(0x48);
                self.emit_byte(0x99);
            }

            // --- Special ---
            opcodes::NOP => {
                // NOP with a Label operand is a pseudo-instruction that defines
                // the label (basic block entry point) without emitting any bytes.
                // Plain NOP (no operands) emits 0x90.
                if !ops.is_empty() {
                    if let MachineOperand::Label(id) = &ops[0] {
                        self.define_label(*id);
                        return Ok(());
                    }
                }
                self.emit_byte(0x90);
            }
            opcodes::ENDBR64 => {
                self.emit_bytes(&[0xF3, 0x0F, 0x1E, 0xFA]);
            }
            opcodes::PAUSE => {
                self.emit_bytes(&[0xF3, 0x90]);
            }
            opcodes::LFENCE => {
                self.emit_bytes(&[0x0F, 0xAE, 0xE8]);
            }
            opcodes::UD2 => {
                self.emit_bytes(&[0x0F, 0x0B]);
            }

            // --- Label pseudo-instruction ---
            _ => {
                if !ops.is_empty() {
                    if let MachineOperand::Label(id) = &ops[0] {
                        self.define_label(*id);
                        return Ok(());
                    }
                }
                return Err(CodeGenError::EncodingError(format!(
                    "unknown opcode: 0x{:04X}",
                    op
                )));
            }
        }
        Ok(())
    }

    /// Encode a sequence of machine instructions into a complete function body.
    /// Handles label pseudo-instructions and resolves all internal fixups.
    pub fn encode_function(&mut self, instrs: &[MachineInstr]) -> Result<Vec<u8>, CodeGenError> {
        self.reset();
        for instr in instrs {
            self.encode_instr(instr)?;
        }
        self.resolve_fixups()?;
        Ok(self.code.clone())
    }

    /// Encode all instructions into an EncodedFunction with code, relocations,
    /// and label mappings. This is the primary entry point for the encoder.
    pub fn encode_all(&mut self, instrs: &[MachineInstr]) -> Result<EncodedFunction, CodeGenError> {
        self.reset();
        for instr in instrs {
            self.encode_instr(instr)?;
        }
        self.resolve_fixups()?;
        Ok(EncodedFunction {
            code: std::mem::take(&mut self.code),
            relocations: std::mem::take(&mut self.relocations),
            labels: std::mem::take(&mut self.labels),
        })
    }
}

// =============================================================================
// Unit Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::regalloc::PhysReg;
    use crate::codegen::x86_64::isel::{opcodes, CondCode};
    use crate::codegen::{MachineInstr, MachineOperand};

    // --- Helper constructors ---
    fn reg(id: u16) -> MachineOperand {
        MachineOperand::Register(PhysReg(id))
    }
    fn imm(v: i64) -> MachineOperand {
        MachineOperand::Immediate(v)
    }
    fn mem(base: u16, off: i32) -> MachineOperand {
        MachineOperand::Memory {
            base: PhysReg(base),
            offset: off,
        }
    }
    fn sym(name: &str) -> MachineOperand {
        MachineOperand::Symbol(name.to_string())
    }
    fn label(id: u32) -> MachineOperand {
        MachineOperand::Label(id)
    }

    fn instr(opcode: u32, operands: Vec<MachineOperand>) -> MachineInstr {
        MachineInstr {
            opcode,
            operands,
            loc: None,
        }
    }

    fn encode_single(opcode: u32, operands: Vec<MachineOperand>) -> Vec<u8> {
        let mut enc = X86_64Encoder::new();
        enc.encode_function(&[instr(opcode, operands)])
            .expect("encoding failed")
    }

    // ====================================================================
    // REX prefix tests
    // ====================================================================
    #[test]
    fn test_rex_w_bit() {
        // add rax, rcx  => REX.W(0x48) + 0x01 + ModR/M(C8)
        let bytes = encode_single(opcodes::ADD_RR, vec![reg(0), reg(1)]);
        assert_eq!(bytes[0], 0x48, "REX.W bit for 64-bit");
    }

    #[test]
    fn test_rex_r_bit() {
        // IMUL r8, rax  =>  REX.WR(0x4C) + 0F AF + ModR/M
        let bytes = encode_single(opcodes::IMUL_RR, vec![reg(8), reg(0)]);
        assert!(bytes[0] & 0x44 == 0x44, "REX.R should be set for r8 dest");
    }

    #[test]
    fn test_rex_b_bit() {
        // add r9, rax  =>  dst=r9 in rm field needs REX.B, src=rax in reg field
        // REX = 0x40 | W(0x08) | B(0x01) = 0x49
        let bytes = encode_single(opcodes::ADD_RR, vec![reg(9), reg(0)]);
        assert_eq!(
            bytes[0] & 0x49,
            0x49,
            "REX.WB should be set for r9 dst in rm field"
        );
    }

    #[test]
    fn test_rex_wrb() {
        // add r8, r9  => REX.WRB needed
        let bytes = encode_single(opcodes::ADD_RR, vec![reg(8), reg(9)]);
        let rex = bytes[0];
        assert!(rex & 0x48 == 0x48, "REX.W set");
        // In ALU RR: src goes in reg field, dst in rm. src=r9→REX.R, dst=r8→REX.B
        assert!(rex & 0x05 == 0x05, "Both REX.R and REX.B should be set");
    }

    #[test]
    fn test_no_rex_for_xor_self() {
        // xor eax, eax => 31 C0 (no REX needed — 32-bit form)
        let bytes = encode_single(opcodes::XOR_RR, vec![reg(0), reg(0)]);
        assert_eq!(bytes, vec![0x31, 0xC0]);
    }

    // ====================================================================
    // ModR/M encoding tests
    // ====================================================================
    #[test]
    fn test_modrm_reg_reg() {
        // modrm(11, rax=0, rcx=1) => 0xC1
        assert_eq!(modrm(0b11, 0, 1), 0xC1);
    }

    #[test]
    fn test_modrm_reg_mem_simple() {
        // mov rax, [rcx]  => REX.W + 0x8B + ModR/M(00,000,001) = 0x01
        let bytes = encode_single(opcodes::MOV_RM, vec![reg(0), mem(1, 0)]);
        // REX.W=0x48, opcode=0x8B, ModR/M=modrm(00,0,1)=0x01
        assert_eq!(bytes, vec![0x48, 0x8B, 0x01]);
    }

    #[test]
    fn test_modrm_with_disp8() {
        // mov rax, [rcx + 8]  => 0x48 0x8B 0x41 0x08
        let bytes = encode_single(opcodes::MOV_RM, vec![reg(0), mem(1, 8)]);
        assert_eq!(bytes, vec![0x48, 0x8B, 0x41, 0x08]);
    }

    #[test]
    fn test_modrm_with_disp32() {
        // mov rax, [rcx + 0x200]  => 0x48 0x8B 0x81 0x00 0x02 0x00 0x00
        let bytes = encode_single(opcodes::MOV_RM, vec![reg(0), mem(1, 0x200)]);
        assert_eq!(bytes, vec![0x48, 0x8B, 0x81, 0x00, 0x02, 0x00, 0x00]);
    }

    #[test]
    fn test_modrm_rsp_requires_sib() {
        // mov rax, [rsp]  => needs SIB byte (RSP encoding 4 triggers SIB)
        let bytes = encode_single(opcodes::MOV_RM, vec![reg(0), mem(4, 0)]);
        // REX.W=0x48, opcode=0x8B, ModR/M(00,000,100)=0x04, SIB(00,100,100)=0x24
        assert_eq!(bytes, vec![0x48, 0x8B, 0x04, 0x24]);
    }

    #[test]
    fn test_modrm_rbp_requires_disp() {
        // mov rax, [rbp]  => uses disp8=0 (RBP with mod=00 means RIP-relative)
        let bytes = encode_single(opcodes::MOV_RM, vec![reg(0), mem(5, 0)]);
        // REX.W=0x48, opcode=0x8B, ModR/M(01,000,101)=0x45, disp8=0x00
        assert_eq!(bytes, vec![0x48, 0x8B, 0x45, 0x00]);
    }

    // ====================================================================
    // SIB byte tests
    // ====================================================================
    #[test]
    fn test_sib_simple_base() {
        assert_eq!(sib(0, 4, 4), 0x24); // scale=1, no index (RSP=4), base=RSP
    }

    #[test]
    fn test_sib_construction() {
        // scale=2 (01), index=rcx(1), base=rax(0) => 0b01_001_000 = 0x48
        assert_eq!(sib(1, 1, 0), 0x48);
    }

    // ====================================================================
    // Full instruction encoding tests
    // ====================================================================
    #[test]
    fn test_encode_add_rr() {
        // add rax, rcx  => 48 01 C8
        // src(rcx=1) in reg field, dst(rax=0) in rm field
        let bytes = encode_single(opcodes::ADD_RR, vec![reg(0), reg(1)]);
        assert_eq!(bytes, vec![0x48, 0x01, 0xC8]);
    }

    #[test]
    fn test_encode_add_ri_small() {
        // add rax, 5  => REX.W + 0x83 + ModR/M(/0, rax) + imm8(5)
        let bytes = encode_single(opcodes::ADD_RI, vec![reg(0), imm(5)]);
        assert_eq!(bytes, vec![0x48, 0x83, 0xC0, 0x05]);
    }

    #[test]
    fn test_encode_add_ri_large() {
        // add rax, 0x12345  => REX.W + 0x81 + ModR/M(/0, rax) + imm32
        let bytes = encode_single(opcodes::ADD_RI, vec![reg(0), imm(0x12345)]);
        assert_eq!(bytes, vec![0x48, 0x81, 0xC0, 0x45, 0x23, 0x01, 0x00]);
    }

    #[test]
    fn test_encode_sub_rr() {
        // sub rdx, rbx => 48 29 DA
        let bytes = encode_single(opcodes::SUB_RR, vec![reg(2), reg(3)]);
        assert_eq!(bytes, vec![0x48, 0x29, 0xDA]);
    }

    #[test]
    fn test_encode_mov_rr() {
        // mov rdi, rsi => 48 89 F7
        let bytes = encode_single(opcodes::MOV_RR, vec![reg(7), reg(6)]);
        assert_eq!(bytes, vec![0x48, 0x89, 0xF7]);
    }

    #[test]
    fn test_encode_mov_ri_i32() {
        // mov eax, 0x12345678 => B8 + 4-byte imm (32-bit, zero-extends to rax)
        // Uses optimized encoding: non-negative imm32 avoids REX.W prefix
        let bytes = encode_single(opcodes::MOV_RI, vec![reg(0), imm(0x12345678)]);
        assert_eq!(bytes, vec![0xB8, 0x78, 0x56, 0x34, 0x12]);
    }

    #[test]
    fn test_encode_mov_ri64() {
        // movabs rax, 0x123456789ABCDEF0 => 48 B8 F0 DE BC 9A 78 56 34 12
        let v: i64 = 0x123456789ABCDEF0u64 as i64;
        let bytes = encode_single(opcodes::MOV_RI, vec![reg(0), imm(v)]);
        assert_eq!(bytes[0], 0x48); // REX.W
        assert_eq!(bytes[1], 0xB8); // B8+rax
        assert_eq!(&bytes[2..10], &0x123456789ABCDEF0u64.to_le_bytes());
    }

    #[test]
    fn test_encode_push_pop() {
        let push_bytes = encode_single(opcodes::PUSH, vec![reg(3)]); // push rbx => 53
        assert_eq!(push_bytes, vec![0x53]);
        let pop_bytes = encode_single(opcodes::POP, vec![reg(3)]); // pop rbx => 5B
        assert_eq!(pop_bytes, vec![0x5B]);
    }

    #[test]
    fn test_encode_push_r8() {
        // push r8 => 41 50
        let bytes = encode_single(opcodes::PUSH, vec![reg(8)]);
        assert_eq!(bytes, vec![0x41, 0x50]);
    }

    #[test]
    fn test_encode_ret() {
        let bytes = encode_single(opcodes::RET, vec![]);
        assert_eq!(bytes, vec![0xC3]);
    }

    #[test]
    fn test_encode_call_symbol() {
        // call <sym> => E8 xx xx xx xx (with PLT32 relocation)
        let mut enc = X86_64Encoder::new();
        let result = enc
            .encode_all(&[instr(opcodes::CALL, vec![sym("printf")])])
            .unwrap();
        assert_eq!(result.code[0], 0xE8);
        assert_eq!(result.code.len(), 5);
        assert_eq!(result.relocations.len(), 1);
        assert_eq!(result.relocations[0].symbol, "printf");
        assert_eq!(result.relocations[0].addend, -4);
    }

    #[test]
    fn test_encode_call_r() {
        // call rax => FF D0
        let bytes = encode_single(opcodes::CALL_R, vec![reg(0)]);
        assert_eq!(bytes, vec![0xFF, 0xD0]);
    }

    #[test]
    fn test_encode_nop() {
        assert_eq!(encode_single(opcodes::NOP, vec![]), vec![0x90]);
    }

    #[test]
    fn test_encode_endbr64() {
        assert_eq!(
            encode_single(opcodes::ENDBR64, vec![]),
            vec![0xF3, 0x0F, 0x1E, 0xFA]
        );
    }

    #[test]
    fn test_encode_pause() {
        assert_eq!(encode_single(opcodes::PAUSE, vec![]), vec![0xF3, 0x90]);
    }

    #[test]
    fn test_encode_lfence() {
        assert_eq!(
            encode_single(opcodes::LFENCE, vec![]),
            vec![0x0F, 0xAE, 0xE8]
        );
    }

    #[test]
    fn test_encode_cdq() {
        assert_eq!(encode_single(opcodes::CDQ, vec![]), vec![0x99]);
    }

    #[test]
    fn test_encode_cqo() {
        assert_eq!(encode_single(opcodes::CQO, vec![]), vec![0x48, 0x99]);
    }

    // ====================================================================
    // SSE instruction tests
    // ====================================================================
    #[test]
    fn test_encode_addsd() {
        // addsd xmm0, xmm1 => F2 0F 58 C1
        let bytes = encode_single(opcodes::ADDSD, vec![reg(16), reg(17)]);
        assert_eq!(bytes, vec![0xF2, 0x0F, 0x58, 0xC1]);
    }

    #[test]
    fn test_encode_movsd() {
        // movsd xmm0, xmm1 => F2 0F 10 C1
        let bytes = encode_single(opcodes::MOVSD, vec![reg(16), reg(17)]);
        assert_eq!(bytes, vec![0xF2, 0x0F, 0x10, 0xC1]);
    }

    #[test]
    fn test_encode_addss() {
        // addss xmm2, xmm3 => F3 0F 58 D3
        let bytes = encode_single(opcodes::ADDSS, vec![reg(18), reg(19)]);
        assert_eq!(bytes, vec![0xF3, 0x0F, 0x58, 0xD3]);
    }

    #[test]
    fn test_encode_mulsd() {
        // mulsd xmm0, xmm1 => F2 0F 59 C1
        let bytes = encode_single(opcodes::MULSD, vec![reg(16), reg(17)]);
        assert_eq!(bytes, vec![0xF2, 0x0F, 0x59, 0xC1]);
    }

    #[test]
    fn test_encode_ucomisd() {
        // ucomisd xmm0, xmm1 => 66 0F 2E C1
        let bytes = encode_single(opcodes::UCOMISD, vec![reg(16), reg(17)]);
        assert_eq!(bytes, vec![0x66, 0x0F, 0x2E, 0xC1]);
    }

    #[test]
    fn test_encode_ucomiss() {
        // ucomiss xmm0, xmm1 => 0F 2E C1 (no prefix byte for NP)
        let bytes = encode_single(opcodes::UCOMISS, vec![reg(16), reg(17)]);
        assert_eq!(bytes, vec![0x0F, 0x2E, 0xC1]);
    }

    // ====================================================================
    // Label resolution tests
    // ====================================================================
    #[test]
    fn test_forward_jump_fixup() {
        // JMP label(1), NOP, label(1):
        let mut enc = X86_64Encoder::new();
        let result = enc
            .encode_all(&[
                instr(opcodes::JMP, vec![label(1)]),
                instr(opcodes::NOP, vec![]),
                instr(0xFFFF, vec![label(1)]), // label pseudo-instruction
            ])
            .unwrap();
        // JMP is E9 + rel32. JMP occupies 5 bytes (0..4), NOP at offset 5.
        // Label at offset 6. rel32 = 6 - (0 + 4+1) = 6-5 = 1
        assert_eq!(result.code[0], 0xE9);
        let disp = i32::from_le_bytes([
            result.code[1],
            result.code[2],
            result.code[3],
            result.code[4],
        ]);
        assert_eq!(disp, 1, "forward jump should skip the NOP");
    }

    #[test]
    fn test_backward_jump() {
        // label(1):, NOP, JMP label(1)
        let mut enc = X86_64Encoder::new();
        let result = enc
            .encode_all(&[
                instr(0xFFFF, vec![label(1)]),       // label at offset 0
                instr(opcodes::NOP, vec![]),         // NOP at offset 0 (label emits no bytes)
                instr(opcodes::JMP, vec![label(1)]), // JMP at offset 1
            ])
            .unwrap();
        // NOP=1 byte at offset 0, JMP at offset 1 (E9 + 4 bytes)
        // rel32 = 0 - (1 + 4 + 1) = 0 - 6 = -6
        assert_eq!(result.code[1], 0xE9);
        let disp = i32::from_le_bytes([
            result.code[2],
            result.code[3],
            result.code[4],
            result.code[5],
        ]);
        assert_eq!(disp, -6, "backward jump should have negative displacement");
    }

    // ====================================================================
    // Immediate size tests
    // ====================================================================
    #[test]
    fn test_imm8_fits() {
        assert!(fits_in_i8(127));
        assert!(fits_in_i8(-128));
        assert!(!fits_in_i8(128));
        assert!(!fits_in_i8(-129));
    }

    #[test]
    fn test_imm32_fits() {
        assert!(fits_in_i32(0x7FFFFFFF));
        assert!(fits_in_i32(-0x80000000));
        assert!(!fits_in_i32(0x80000000));
    }

    // ====================================================================
    // Register encoding tests
    // ====================================================================
    #[test]
    fn test_reg_encoding_low() {
        assert_eq!(reg_encoding(PhysReg(0)), 0); // rax
        assert_eq!(reg_encoding(PhysReg(7)), 7); // rdi
    }

    #[test]
    fn test_reg_encoding_high() {
        assert_eq!(reg_encoding(PhysReg(8)), 0); // r8 encodes as 0 with REX.B
        assert_eq!(reg_encoding(PhysReg(15)), 7); // r15 encodes as 7 with REX.B
    }

    #[test]
    fn test_needs_rex_extension() {
        assert!(!needs_rex_extension(PhysReg(0)));
        assert!(!needs_rex_extension(PhysReg(7)));
        assert!(needs_rex_extension(PhysReg(8)));
        assert!(needs_rex_extension(PhysReg(15)));
        // XMM registers (>= 16) should not trigger GPR REX extension
        assert!(!needs_rex_extension(PhysReg(16)));
    }

    // ====================================================================
    // Condition code mapping tests
    // ====================================================================
    #[test]
    fn test_condcode_mapping() {
        assert_eq!(condcode_enum_to_cc(&CondCode::E), 0x04);
        assert_eq!(condcode_enum_to_cc(&CondCode::NE), 0x05);
        assert_eq!(condcode_enum_to_cc(&CondCode::L), 0x0C);
        assert_eq!(condcode_enum_to_cc(&CondCode::GE), 0x0D);
        assert_eq!(condcode_enum_to_cc(&CondCode::B), 0x02);
        assert_eq!(condcode_enum_to_cc(&CondCode::AE), 0x03);
    }

    // ====================================================================
    // SETCC and JCC tests
    // ====================================================================
    #[test]
    fn test_encode_setcc() {
        // sete al => 0F 94 C0
        let cc_val = CondCode::E as i64;
        let bytes = encode_single(opcodes::SETCC, vec![imm(cc_val), reg(0)]);
        assert_eq!(bytes, vec![0x0F, 0x94, 0xC0]);
    }

    #[test]
    fn test_encode_jcc_forward() {
        let cc_val = CondCode::NE as i64;
        let mut enc = X86_64Encoder::new();
        let result = enc
            .encode_all(&[
                instr(opcodes::JCC, vec![imm(cc_val), label(1)]),
                instr(0xFFFF, vec![label(1)]),
            ])
            .unwrap();
        // JCC = 0F 85 + rel32 (6 bytes)
        assert_eq!(result.code[0], 0x0F);
        assert_eq!(result.code[1], 0x85);
    }

    // ====================================================================
    // Shift encoding tests
    // ====================================================================
    #[test]
    fn test_encode_shl_ri() {
        // shl rax, 4 => 48 C1 E0 04
        let bytes = encode_single(opcodes::SHL_RI, vec![reg(0), imm(4)]);
        assert_eq!(bytes, vec![0x48, 0xC1, 0xE0, 0x04]);
    }

    #[test]
    fn test_encode_shr_ri() {
        // shr rax, 1 => 48 C1 E8 01
        let bytes = encode_single(opcodes::SHR_RI, vec![reg(0), imm(1)]);
        assert_eq!(bytes, vec![0x48, 0xC1, 0xE8, 0x01]);
    }

    // ====================================================================
    // NOT/NEG encoding tests
    // ====================================================================
    #[test]
    fn test_encode_not() {
        // not rax => 48 F7 D0
        let bytes = encode_single(opcodes::NOT_R, vec![reg(0)]);
        assert_eq!(bytes, vec![0x48, 0xF7, 0xD0]);
    }

    #[test]
    fn test_encode_neg() {
        // neg rax => 48 F7 D8
        let bytes = encode_single(opcodes::NEG_R, vec![reg(0)]);
        assert_eq!(bytes, vec![0x48, 0xF7, 0xD8]);
    }

    // ====================================================================
    // Memory operation tests
    // ====================================================================
    #[test]
    fn test_encode_store_load_64() {
        // STORE64 [rbx+8], rax => 48 89 43 08
        let store = encode_single(opcodes::STORE64, vec![mem(3, 8), reg(0)]);
        assert_eq!(store, vec![0x48, 0x89, 0x43, 0x08]);
        // LOAD64 rax, [rbx+8] => 48 8B 43 08
        let load = encode_single(opcodes::LOAD64, vec![reg(0), mem(3, 8)]);
        assert_eq!(load, vec![0x48, 0x8B, 0x43, 0x08]);
    }

    // ====================================================================
    // LEA test
    // ====================================================================
    #[test]
    fn test_encode_lea() {
        // lea rax, [rbx + 16] => 48 8D 43 10
        let bytes = encode_single(opcodes::LEA, vec![reg(0), mem(3, 16)]);
        assert_eq!(bytes, vec![0x48, 0x8D, 0x43, 0x10]);
    }

    // ====================================================================
    // MOVSX / MOVZX tests
    // ====================================================================
    #[test]
    fn test_encode_movsx_8_to_64() {
        // movsx rax, cl => REX.W + 0F BE C1
        let bytes = encode_single(opcodes::MOVSX, vec![reg(0), reg(1), imm(1)]);
        assert_eq!(bytes, vec![0x48, 0x0F, 0xBE, 0xC1]);
    }

    #[test]
    fn test_encode_movzx_8() {
        // movzx eax, cl => 0F B6 C1
        let bytes = encode_single(opcodes::MOVZX, vec![reg(0), reg(1), imm(1)]);
        assert_eq!(bytes, vec![0x0F, 0xB6, 0xC1]);
    }

    // ====================================================================
    // Encode-all returns correct structure
    // ====================================================================
    #[test]
    fn test_encode_all_structure() {
        let mut enc = X86_64Encoder::new();
        let result = enc
            .encode_all(&[
                instr(0xFFFF, vec![label(0)]),
                instr(opcodes::NOP, vec![]),
                instr(opcodes::RET, vec![]),
            ])
            .unwrap();
        assert_eq!(result.code.len(), 2); // NOP + RET
        assert_eq!(result.code, vec![0x90, 0xC3]);
        assert!(result.labels.contains_key(&0));
        assert_eq!(*result.labels.get(&0).unwrap(), 0);
    }

    // ====================================================================
    // IMUL tests
    // ====================================================================
    #[test]
    fn test_encode_imul_rr() {
        // imul rax, rcx => 48 0F AF C1
        let bytes = encode_single(opcodes::IMUL_RR, vec![reg(0), reg(1)]);
        assert_eq!(bytes, vec![0x48, 0x0F, 0xAF, 0xC1]);
    }

    #[test]
    fn test_encode_imul_ri_small() {
        // imul rax, rax, 5 => 48 6B C0 05
        let bytes = encode_single(opcodes::IMUL_RI, vec![reg(0), imm(5)]);
        assert_eq!(bytes, vec![0x48, 0x6B, 0xC0, 0x05]);
    }

    // ====================================================================
    // TEST with immediate
    // ====================================================================
    #[test]
    fn test_encode_test_ri() {
        // test rax, 0xFF => 48 F7 C0 FF 00 00 00
        let bytes = encode_single(opcodes::TEST_RI, vec![reg(0), imm(0xFF)]);
        assert_eq!(bytes, vec![0x48, 0xF7, 0xC0, 0xFF, 0x00, 0x00, 0x00]);
    }

    // ====================================================================
    // Extended register instruction test
    // ====================================================================
    #[test]
    fn test_encode_mov_r12_r13() {
        // mov r12, r13 => 4D 89 EC
        // src=r13 goes in reg field (needs REX.R), dst=r12 goes in rm (needs REX.B)
        let bytes = encode_single(opcodes::MOV_RR, vec![reg(12), reg(13)]);
        assert_eq!(bytes[0], 0x4D); // REX.WRB
    }

    // ====================================================================
    // Conversion instruction tests
    // ====================================================================
    #[test]
    fn test_encode_cvtsi2sd() {
        // cvtsi2sd xmm0, rax => F2 48 0F 2A C0
        let bytes = encode_single(opcodes::CVTSI2SD, vec![reg(16), reg(0)]);
        assert_eq!(bytes, vec![0xF2, 0x48, 0x0F, 0x2A, 0xC0]);
    }

    #[test]
    fn test_encode_cvttsd2si() {
        // cvttsd2si rax, xmm0 => F2 48 0F 2C C0
        let bytes = encode_single(opcodes::CVTTSD2SI, vec![reg(0), reg(16)]);
        assert_eq!(bytes, vec![0xF2, 0x48, 0x0F, 0x2C, 0xC0]);
    }
}
