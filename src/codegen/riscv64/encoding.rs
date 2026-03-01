//! # RISC-V 64 Integrated Assembler — Machine Code Encoder
//!
//! Encodes RV64GC machine instructions into binary byte sequences for ELF64
//! `.text` section emission. Implements all six 32-bit RISC-V base instruction
//! formats (R, I, S, B, U, J).
//!
//! ## Zero External Dependencies
//!
//! Only `std` and internal crate modules. No external crates.

use std::collections::HashMap;

use crate::codegen::{
    MachineInstr, MachineOperand, PhysReg, Relocation, RelocationType,
};
use super::isel::Riscv64Opcode;

// ====================== RISC-V Base Opcodes (bits [6:0]) ======================
const OP_LUI: u32 = 0b0110111;
const OP_AUIPC: u32 = 0b0010111;
const OP_JAL: u32 = 0b1101111;
const OP_JALR: u32 = 0b1100111;
const OP_BRANCH: u32 = 0b1100011;
const OP_LOAD: u32 = 0b0000011;
const OP_STORE: u32 = 0b0100011;
const OP_OP_IMM: u32 = 0b0010011;
const OP_OP: u32 = 0b0110011;
const OP_OP_IMM_32: u32 = 0b0011011;
const OP_OP_32: u32 = 0b0111011;
const OP_LOAD_FP: u32 = 0b0000111;
const OP_STORE_FP: u32 = 0b0100111;
const OP_OP_FP: u32 = 0b1010011;
const OP_AMO: u32 = 0b0101111;
const OP_FENCE: u32 = 0b0001111;
const OP_SYSTEM: u32 = 0b1110011;

// ====================== funct3 ======================
const FUNCT3_BEQ: u32 = 0b000;
const FUNCT3_BNE: u32 = 0b001;
const FUNCT3_BLT: u32 = 0b100;
const FUNCT3_BGE: u32 = 0b101;
const FUNCT3_BLTU: u32 = 0b110;
const FUNCT3_BGEU: u32 = 0b111;
const FUNCT3_LB: u32 = 0b000;
const FUNCT3_LH: u32 = 0b001;
const FUNCT3_LW: u32 = 0b010;
const FUNCT3_LD: u32 = 0b011;
const FUNCT3_LBU: u32 = 0b100;
const FUNCT3_LHU: u32 = 0b101;
const FUNCT3_LWU: u32 = 0b110;
const FUNCT3_SB: u32 = 0b000;
const FUNCT3_SH: u32 = 0b001;
const FUNCT3_SW: u32 = 0b010;
const FUNCT3_SD: u32 = 0b011;
const FUNCT3_ADDI: u32 = 0b000;
const FUNCT3_SLTI: u32 = 0b010;
const FUNCT3_SLTIU: u32 = 0b011;
const FUNCT3_XORI: u32 = 0b100;
const FUNCT3_ORI: u32 = 0b110;
const FUNCT3_ANDI: u32 = 0b111;
const FUNCT3_SLLI: u32 = 0b001;
const FUNCT3_SRLI_SRAI: u32 = 0b101;
const FUNCT3_ADD_SUB: u32 = 0b000;
const FUNCT3_SLL: u32 = 0b001;
const FUNCT3_SLT: u32 = 0b010;
const FUNCT3_SLTU: u32 = 0b011;
const FUNCT3_XOR: u32 = 0b100;
const FUNCT3_SRL_SRA: u32 = 0b101;
const FUNCT3_OR: u32 = 0b110;
const FUNCT3_AND: u32 = 0b111;
const FUNCT3_MUL: u32 = 0b000;
const FUNCT3_MULH: u32 = 0b001;
const FUNCT3_MULHSU: u32 = 0b010;
const FUNCT3_MULHU: u32 = 0b011;
const FUNCT3_DIV: u32 = 0b100;
const FUNCT3_DIVU: u32 = 0b101;
const FUNCT3_REM: u32 = 0b110;
const FUNCT3_REMU: u32 = 0b111;
const FUNCT3_FLW: u32 = 0b010;
const FUNCT3_FLD: u32 = 0b011;
const FUNCT3_JALR: u32 = 0b000;
const FUNCT3_FENCE: u32 = 0b000;
const FUNCT3_AMO_D: u32 = 0b011;

// ====================== funct7 ======================
const FUNCT7_NORMAL: u32 = 0b0000000;
const FUNCT7_ALT: u32 = 0b0100000;
const FUNCT7_MULDIV: u32 = 0b0000001;

// ====================== FP format / operation ======================
const FMT_S: u32 = 0b00;
const FMT_D: u32 = 0b01;
const FP_ADD: u32 = 0b00000;
const FP_SUB: u32 = 0b00001;
const FP_MUL: u32 = 0b00010;
const FP_DIV: u32 = 0b00011;
const FP_SQRT: u32 = 0b01011;
const FP_MINMAX: u32 = 0b00101;
const FP_CMP: u32 = 0b10100;
const FP_CVT_FROM_FP: u32 = 0b11000;
const FP_CVT_TO_FP: u32 = 0b11010;
const FP_CVT_FP_FP: u32 = 0b01000;
const FP_MV_TO_INT: u32 = 0b11100;
const FP_MV_TO_FP: u32 = 0b11110;
const RM_DYN: u32 = 0b111;

// ====================== AMO operations ======================
const AMO_LR: u32 = 0b00010;
const AMO_SC: u32 = 0b00011;
const AMO_SWAP: u32 = 0b00001;
const AMO_ADD: u32 = 0b00000;
const AMO_AND: u32 = 0b01100;
const AMO_OR: u32 = 0b01000;
const AMO_XOR: u32 = 0b00100;
const AMO_MAX: u32 = 0b10100;
const AMO_MIN: u32 = 0b10000;
const AMO_MAXU: u32 = 0b11100;
const AMO_MINU: u32 = 0b11000;

// ====================== Fixup types ======================
struct BranchFixup {
    label: u32,
    code_offset: usize,
    fixup_type: FixupType,
}

#[derive(Clone, Copy)]
enum FixupType {
    BranchB,
    JumpJ,
    #[allow(dead_code)]
    CallPair,
}

// ====================== Format encoding ======================

#[inline]
fn encode_r_type(opcode: u32, rd: u32, funct3: u32, rs1: u32, rs2: u32, funct7: u32) -> u32 {
    (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}

#[inline]
fn encode_i_type(opcode: u32, rd: u32, funct3: u32, rs1: u32, imm: i32) -> u32 {
    let imm = (imm as u32) & 0xFFF;
    (imm << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}

#[inline]
fn encode_s_type(opcode: u32, funct3: u32, rs1: u32, rs2: u32, imm: i32) -> u32 {
    let imm = (imm as u32) & 0xFFF;
    ((imm >> 5) & 0x7F) << 25 | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | ((imm & 0x1F) << 7) | opcode
}

/// B-type: non-contiguous immediate bit shuffling.
#[inline]
fn encode_b_type(opcode: u32, funct3: u32, rs1: u32, rs2: u32, imm: i32) -> u32 {
    let imm = imm as u32;
    ((imm >> 12) & 0x1) << 31
        | ((imm >> 5) & 0x3F) << 25
        | (rs2 << 20) | (rs1 << 15) | (funct3 << 12)
        | ((imm >> 1) & 0xF) << 8
        | ((imm >> 11) & 0x1) << 7
        | opcode
}

#[inline]
fn encode_u_type(opcode: u32, rd: u32, imm: u32) -> u32 {
    (imm & 0xFFFFF000) | (rd << 7) | opcode
}

/// J-type: non-contiguous immediate bit shuffling.
#[inline]
fn encode_j_type(opcode: u32, rd: u32, imm: i32) -> u32 {
    let imm = imm as u32;
    ((imm >> 20) & 0x1) << 31
        | ((imm >> 1) & 0x3FF) << 21
        | ((imm >> 11) & 0x1) << 20
        | ((imm >> 12) & 0xFF) << 12
        | (rd << 7) | opcode
}

#[inline]
fn encode_fp_r_type(fp_op: u32, fmt: u32, rd: u32, rs1: u32, rs2: u32, rm: u32) -> u32 {
    encode_r_type(OP_OP_FP, rd, rm, rs1, rs2, (fp_op << 2) | fmt)
}

#[inline]
fn encode_amo(amo_op: u32, rd: u32, rs1: u32, rs2: u32, aq: u32, rl: u32) -> u32 {
    encode_r_type(OP_AMO, rd, FUNCT3_AMO_D, rs1, rs2, (amo_op << 2) | (aq << 1) | rl)
}

// ====================== Riscv64Encoder ======================

/// RISC-V 64-bit machine code encoder (integrated assembler).
pub struct Riscv64Encoder {
    code: Vec<u8>,
    offset: usize,
    relocations: Vec<Relocation>,
    label_offsets: HashMap<u32, usize>,
    fixups: Vec<BranchFixup>,
}

impl Riscv64Encoder {
    /// Create a new, empty encoder.
    pub fn new() -> Self {
        Self {
            code: Vec::with_capacity(4096),
            offset: 0,
            relocations: Vec::new(),
            label_offsets: HashMap::new(),
            fixups: Vec::new(),
        }
    }

    /// Encode a function's instructions, resolve branches, return code bytes.
    pub fn encode_function(&mut self, instrs: &[MachineInstr]) -> Vec<u8> {
        self.code.clear();
        self.offset = 0;
        self.relocations.clear();
        self.label_offsets.clear();
        self.fixups.clear();
        for instr in instrs {
            self.encode_instruction(instr);
        }
        self.resolve_labels();
        self.code.clone()
    }

    /// Returns the relocations emitted during the most recent encoding.
    pub fn get_relocations(&self) -> &[Relocation] {
        &self.relocations
    }

    /// Returns the current encoded size in bytes.
    pub fn code_size(&self) -> usize {
        self.offset
    }

    // ---- byte emission ----

    fn emit_u32(&mut self, value: u32) {
        self.code.extend_from_slice(&value.to_le_bytes());
        self.offset += 4;
    }

    #[allow(dead_code)]
    fn emit_u16(&mut self, value: u16) {
        self.code.extend_from_slice(&value.to_le_bytes());
        self.offset += 2;
    }

    fn patch_u32(&mut self, pos: usize, value: u32) {
        let bytes = value.to_le_bytes();
        self.code[pos..pos + 4].copy_from_slice(&bytes);
    }

    // ---- operand extraction ----

    fn get_reg(op: &MachineOperand) -> u32 {
        match op {
            MachineOperand::Register(r) => {
                if r.0 >= 32 { (r.0 - 32) as u32 } else { r.0 as u32 }
            }
            _ => 0,
        }
    }

    fn get_imm(op: &MachineOperand) -> i32 {
        match op { MachineOperand::Immediate(v) => *v as i32, _ => 0 }
    }

    fn get_imm64(op: &MachineOperand) -> i64 {
        match op { MachineOperand::Immediate(v) => *v, _ => 0 }
    }

    fn get_symbol(op: &MachineOperand) -> &str {
        match op { MachineOperand::Symbol(s) => s.as_str(), _ => "" }
    }

    fn get_label(op: &MachineOperand) -> u32 {
        match op { MachineOperand::Label(id) => *id, _ => 0 }
    }

    fn get_mem(op: &MachineOperand) -> (u32, i32) {
        match op {
            MachineOperand::Memory { base, offset } => {
                let b = if base.0 >= 32 { (base.0 - 32) as u32 } else { base.0 as u32 };
                (b, *offset)
            }
            _ => (0, 0),
        }
    }

    fn opcode_from_u32(val: u32) -> Riscv64Opcode {
        // Bounds check: verify val is within the valid discriminant range
        // before transmuting. Riscv64Opcode is #[repr(u32)] with contiguous
        // variants numbered 0..=(LA as u32). Any out-of-range value would
        // cause undefined behavior in the transmute.
        const MAX_DISCRIMINANT: u32 = Riscv64Opcode::LA as u32;
        assert!(
            val <= MAX_DISCRIMINANT,
            "invalid Riscv64Opcode discriminant: 0x{:X} (valid range: 0..=0x{:X})",
            val,
            MAX_DISCRIMINANT,
        );
        // SAFETY: Riscv64Opcode is #[repr(u32)] with contiguous variants
        // numbered 0..N. The assert! above guarantees `val` is within the
        // valid discriminant range, so the transmute cannot produce an
        // invalid enum value. A safe match on 130+ literals is functionally
        // identical but impractical. Scope: single transmute of a u32 that
        // has been validated to be a legal discriminant.
        unsafe { std::mem::transmute(val) }
    }

    fn record_label(&mut self, label_id: u32) {
        self.label_offsets.insert(label_id, self.offset);
    }

    // ---- constant materialisation (LI pseudo) ----

    fn materialize_constant(&mut self, rd: u32, value: i64) {
        if value >= -2048 && value <= 2047 {
            self.emit_u32(encode_i_type(OP_OP_IMM, rd, FUNCT3_ADDI, 0, value as i32));
            return;
        }
        let v32 = value as i32;
        if value == v32 as i64 {
            let lo = ((v32 as u32) & 0xFFF) as i32;
            let hi = if lo >= 0x800_i32 {
                (v32.wrapping_add(0x1000)) & !0xFFF
            } else {
                v32 & !0xFFF
            };
            if hi != 0 {
                self.emit_u32(encode_u_type(OP_LUI, rd, hi as u32));
                if lo != 0 { self.emit_u32(encode_i_type(OP_OP_IMM, rd, FUNCT3_ADDI, rd, lo)); }
            } else {
                self.emit_u32(encode_i_type(OP_OP_IMM, rd, FUNCT3_ADDI, 0, lo));
            }
            return;
        }
        // 64-bit: load upper 32 bits, shift, add lower 32
        let upper = ((value as u64) >> 32) as i32;
        let lower = value as i32;
        self.materialize_constant(rd, upper as i64);
        self.emit_u32(encode_i_type(OP_OP_IMM, rd, FUNCT3_SLLI, rd, 32));
        if lower != 0 {
            let lo12 = ((lower as u32) & 0xFFF) as i32;
            let hi20 = if lo12 >= 0x800_i32 {
                (lower.wrapping_add(0x1000)) & !0xFFF
            } else {
                lower & !0xFFF
            };
            let tmp = 31u32; // t6
            if hi20 != 0 {
                self.emit_u32(encode_u_type(OP_LUI, tmp, hi20 as u32));
                if lo12 != 0 { self.emit_u32(encode_i_type(OP_OP_IMM, tmp, FUNCT3_ADDI, tmp, lo12)); }
            } else {
                self.emit_u32(encode_i_type(OP_OP_IMM, tmp, FUNCT3_ADDI, 0, lo12));
            }
            self.emit_u32(encode_r_type(OP_OP, rd, FUNCT3_ADD_SUB, rd, tmp, FUNCT7_NORMAL));
        }
    }

    // ---- main instruction encoding dispatch ----

    fn encode_instruction(&mut self, instr: &MachineInstr) {
        use Riscv64Opcode::*;

        // Label marker: NOP with a single Label operand records the label.
        if instr.operands.len() == 1 {
            if let MachineOperand::Label(id) = &instr.operands[0] {
                if Self::opcode_from_u32(instr.opcode) == NOP {
                    self.record_label(*id);
                    return;
                }
            }
        }

        let op = Self::opcode_from_u32(instr.opcode);
        match op {
            // === R-type base ALU ===
            ADD  => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP, rd, FUNCT3_ADD_SUB, r1, r2, FUNCT7_NORMAL)); }
            SUB  => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP, rd, FUNCT3_ADD_SUB, r1, r2, FUNCT7_ALT)); }
            SLL  => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP, rd, FUNCT3_SLL, r1, r2, FUNCT7_NORMAL)); }
            SLT  => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP, rd, FUNCT3_SLT, r1, r2, FUNCT7_NORMAL)); }
            SLTU => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP, rd, FUNCT3_SLTU, r1, r2, FUNCT7_NORMAL)); }
            XOR  => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP, rd, FUNCT3_XOR, r1, r2, FUNCT7_NORMAL)); }
            SRL  => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP, rd, FUNCT3_SRL_SRA, r1, r2, FUNCT7_NORMAL)); }
            SRA  => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP, rd, FUNCT3_SRL_SRA, r1, r2, FUNCT7_ALT)); }
            OR   => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP, rd, FUNCT3_OR, r1, r2, FUNCT7_NORMAL)); }
            AND  => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP, rd, FUNCT3_AND, r1, r2, FUNCT7_NORMAL)); }
            // === R-type W-suffix ===
            ADDW => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP_32, rd, FUNCT3_ADD_SUB, r1, r2, FUNCT7_NORMAL)); }
            SUBW => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP_32, rd, FUNCT3_ADD_SUB, r1, r2, FUNCT7_ALT)); }
            SLLW => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP_32, rd, FUNCT3_SLL, r1, r2, FUNCT7_NORMAL)); }
            SRLW => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP_32, rd, FUNCT3_SRL_SRA, r1, r2, FUNCT7_NORMAL)); }
            SRAW => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP_32, rd, FUNCT3_SRL_SRA, r1, r2, FUNCT7_ALT)); }
            // === M extension ===
            MUL    => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP, rd, FUNCT3_MUL, r1, r2, FUNCT7_MULDIV)); }
            MULH   => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP, rd, FUNCT3_MULH, r1, r2, FUNCT7_MULDIV)); }
            MULHSU => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP, rd, FUNCT3_MULHSU, r1, r2, FUNCT7_MULDIV)); }
            MULHU  => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP, rd, FUNCT3_MULHU, r1, r2, FUNCT7_MULDIV)); }
            DIV    => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP, rd, FUNCT3_DIV, r1, r2, FUNCT7_MULDIV)); }
            DIVU   => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP, rd, FUNCT3_DIVU, r1, r2, FUNCT7_MULDIV)); }
            REM    => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP, rd, FUNCT3_REM, r1, r2, FUNCT7_MULDIV)); }
            REMU   => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP, rd, FUNCT3_REMU, r1, r2, FUNCT7_MULDIV)); }
            MULW   => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP_32, rd, FUNCT3_MUL, r1, r2, FUNCT7_MULDIV)); }
            DIVW   => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP_32, rd, FUNCT3_DIV, r1, r2, FUNCT7_MULDIV)); }
            DIVUW  => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP_32, rd, FUNCT3_DIVU, r1, r2, FUNCT7_MULDIV)); }
            REMW   => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP_32, rd, FUNCT3_REM, r1, r2, FUNCT7_MULDIV)); }
            REMUW  => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_r_type(OP_OP_32, rd, FUNCT3_REMU, r1, r2, FUNCT7_MULDIV)); }
            // === I-type ALU ===
            ADDI  => { let (rd,r1,im) = self.rri(instr); self.emit_u32(encode_i_type(OP_OP_IMM, rd, FUNCT3_ADDI, r1, im)); }
            SLTI  => { let (rd,r1,im) = self.rri(instr); self.emit_u32(encode_i_type(OP_OP_IMM, rd, FUNCT3_SLTI, r1, im)); }
            SLTIU => { let (rd,r1,im) = self.rri(instr); self.emit_u32(encode_i_type(OP_OP_IMM, rd, FUNCT3_SLTIU, r1, im)); }
            XORI  => { let (rd,r1,im) = self.rri(instr); self.emit_u32(encode_i_type(OP_OP_IMM, rd, FUNCT3_XORI, r1, im)); }
            ORI   => { let (rd,r1,im) = self.rri(instr); self.emit_u32(encode_i_type(OP_OP_IMM, rd, FUNCT3_ORI, r1, im)); }
            ANDI  => { let (rd,r1,im) = self.rri(instr); self.emit_u32(encode_i_type(OP_OP_IMM, rd, FUNCT3_ANDI, r1, im)); }
            // Shifts (RV64 6-bit shamt)
            SLLI  => { let (rd,r1,sh) = self.rri(instr); self.emit_u32(encode_i_type(OP_OP_IMM, rd, FUNCT3_SLLI, r1, sh & 0x3F)); }
            SRLI  => { let (rd,r1,sh) = self.rri(instr); self.emit_u32(encode_i_type(OP_OP_IMM, rd, FUNCT3_SRLI_SRAI, r1, sh & 0x3F)); }
            SRAI  => { let (rd,r1,sh) = self.rri(instr); self.emit_u32(encode_i_type(OP_OP_IMM, rd, FUNCT3_SRLI_SRAI, r1, 0x400 | (sh & 0x3F))); }
            // W-suffix immediate
            ADDIW => { let (rd,r1,im) = self.rri(instr); self.emit_u32(encode_i_type(OP_OP_IMM_32, rd, FUNCT3_ADDI, r1, im)); }
            SLLIW => { let (rd,r1,sh) = self.rri(instr); self.emit_u32(encode_i_type(OP_OP_IMM_32, rd, FUNCT3_SLLI, r1, sh & 0x1F)); }
            SRLIW => { let (rd,r1,sh) = self.rri(instr); self.emit_u32(encode_i_type(OP_OP_IMM_32, rd, FUNCT3_SRLI_SRAI, r1, sh & 0x1F)); }
            SRAIW => { let (rd,r1,sh) = self.rri(instr); self.emit_u32(encode_i_type(OP_OP_IMM_32, rd, FUNCT3_SRLI_SRAI, r1, 0x400 | (sh & 0x1F))); }
            // === Loads ===
            LB  => { let (rd,(b,o)) = self.rm(instr); self.emit_u32(encode_i_type(OP_LOAD, rd, FUNCT3_LB, b, o)); }
            LH  => { let (rd,(b,o)) = self.rm(instr); self.emit_u32(encode_i_type(OP_LOAD, rd, FUNCT3_LH, b, o)); }
            LW  => { let (rd,(b,o)) = self.rm(instr); self.emit_u32(encode_i_type(OP_LOAD, rd, FUNCT3_LW, b, o)); }
            LD  => { let (rd,(b,o)) = self.rm(instr); self.emit_u32(encode_i_type(OP_LOAD, rd, FUNCT3_LD, b, o)); }
            LBU => { let (rd,(b,o)) = self.rm(instr); self.emit_u32(encode_i_type(OP_LOAD, rd, FUNCT3_LBU, b, o)); }
            LHU => { let (rd,(b,o)) = self.rm(instr); self.emit_u32(encode_i_type(OP_LOAD, rd, FUNCT3_LHU, b, o)); }
            LWU => { let (rd,(b,o)) = self.rm(instr); self.emit_u32(encode_i_type(OP_LOAD, rd, FUNCT3_LWU, b, o)); }
            // === Stores ===
            SB => { let (s,(b,o)) = self.rm(instr); self.emit_u32(encode_s_type(OP_STORE, FUNCT3_SB, b, s, o)); }
            SH => { let (s,(b,o)) = self.rm(instr); self.emit_u32(encode_s_type(OP_STORE, FUNCT3_SH, b, s, o)); }
            SW => { let (s,(b,o)) = self.rm(instr); self.emit_u32(encode_s_type(OP_STORE, FUNCT3_SW, b, s, o)); }
            SD => { let (s,(b,o)) = self.rm(instr); self.emit_u32(encode_s_type(OP_STORE, FUNCT3_SD, b, s, o)); }
            // === Branches ===
            BEQ | BNE | BLT | BGE | BLTU | BGEU => {
                let rs1 = Self::get_reg(&instr.operands[0]);
                let rs2 = Self::get_reg(&instr.operands[1]);
                let f3 = match op { BEQ=>FUNCT3_BEQ, BNE=>FUNCT3_BNE, BLT=>FUNCT3_BLT,
                    BGE=>FUNCT3_BGE, BLTU=>FUNCT3_BLTU, _=>FUNCT3_BGEU };
                match &instr.operands[2] {
                    MachineOperand::Label(id) => {
                        self.fixups.push(BranchFixup { label: *id, code_offset: self.offset, fixup_type: FixupType::BranchB });
                        self.emit_u32(encode_b_type(OP_BRANCH, f3, rs1, rs2, 0));
                    }
                    MachineOperand::Immediate(off) => { self.emit_u32(encode_b_type(OP_BRANCH, f3, rs1, rs2, *off as i32)); }
                    _ => { self.emit_u32(encode_b_type(OP_BRANCH, f3, rs1, rs2, 0)); }
                }
            }
            // === LUI / AUIPC ===
            LUI => {
                let rd = Self::get_reg(&instr.operands[0]);
                let imm = Self::get_imm(&instr.operands[1]) as u32;
                let v = if imm & 0xFFF == 0 { imm } else { imm << 12 };
                self.emit_u32(encode_u_type(OP_LUI, rd, v));
            }
            AUIPC => {
                let rd = Self::get_reg(&instr.operands[0]);
                let imm = Self::get_imm(&instr.operands[1]) as u32;
                let v = if imm & 0xFFF == 0 { imm } else { imm << 12 };
                self.emit_u32(encode_u_type(OP_AUIPC, rd, v));
            }
            // === JAL ===
            JAL => {
                let rd = Self::get_reg(&instr.operands[0]);
                match &instr.operands[1] {
                    MachineOperand::Label(id) => {
                        self.fixups.push(BranchFixup { label: *id, code_offset: self.offset, fixup_type: FixupType::JumpJ });
                        self.emit_u32(encode_j_type(OP_JAL, rd, 0));
                    }
                    MachineOperand::Immediate(off) => { self.emit_u32(encode_j_type(OP_JAL, rd, *off as i32)); }
                    MachineOperand::Symbol(sym) => {
                        self.relocations.push(Relocation { offset: self.offset as u64, symbol: sym.clone(), reloc_type: RelocationType::Riscv_Jal, addend: 0, section_index: 0 });
                        self.emit_u32(encode_j_type(OP_JAL, rd, 0));
                    }
                    _ => { self.emit_u32(encode_j_type(OP_JAL, rd, 0)); }
                }
            }
            // === JALR ===
            JALR => {
                let rd = Self::get_reg(&instr.operands[0]);
                let rs1 = Self::get_reg(&instr.operands[1]);
                let imm = if instr.operands.len() > 2 { Self::get_imm(&instr.operands[2]) } else { 0 };
                self.emit_u32(encode_i_type(OP_JALR, rd, FUNCT3_JALR, rs1, imm));
            }
            // === A extension ===
            LR_D      => { let rd = Self::get_reg(&instr.operands[0]); let rs1 = Self::get_reg(&instr.operands[1]); self.emit_u32(encode_amo(AMO_LR, rd, rs1, 0, 0, 0)); }
            SC_D      => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_amo(AMO_SC, rd, r1, r2, 0, 0)); }
            AMOSWAP_D => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_amo(AMO_SWAP, rd, r1, r2, 0, 0)); }
            AMOADD_D  => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_amo(AMO_ADD, rd, r1, r2, 0, 0)); }
            AMOAND_D  => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_amo(AMO_AND, rd, r1, r2, 0, 0)); }
            AMOOR_D   => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_amo(AMO_OR, rd, r1, r2, 0, 0)); }
            AMOXOR_D  => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_amo(AMO_XOR, rd, r1, r2, 0, 0)); }
            AMOMAX_D  => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_amo(AMO_MAX, rd, r1, r2, 0, 0)); }
            AMOMIN_D  => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_amo(AMO_MIN, rd, r1, r2, 0, 0)); }
            AMOMAXU_D => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_amo(AMO_MAXU, rd, r1, r2, 0, 0)); }
            AMOMINU_D => { let (rd,r1,r2) = self.rrr(instr); self.emit_u32(encode_amo(AMO_MINU, rd, r1, r2, 0, 0)); }
            // === F extension ===
            FLW  => { let (rd,(b,o)) = self.rm(instr); self.emit_u32(encode_i_type(OP_LOAD_FP, rd, FUNCT3_FLW, b, o)); }
            FSW  => { let (s,(b,o)) = self.rm(instr); self.emit_u32(encode_s_type(OP_STORE_FP, FUNCT3_FLW, b, s, o)); }
            FADD_S => { let (d,s1,s2) = self.rrr(instr); self.emit_u32(encode_fp_r_type(FP_ADD, FMT_S, d, s1, s2, RM_DYN)); }
            FSUB_S => { let (d,s1,s2) = self.rrr(instr); self.emit_u32(encode_fp_r_type(FP_SUB, FMT_S, d, s1, s2, RM_DYN)); }
            FMUL_S => { let (d,s1,s2) = self.rrr(instr); self.emit_u32(encode_fp_r_type(FP_MUL, FMT_S, d, s1, s2, RM_DYN)); }
            FDIV_S => { let (d,s1,s2) = self.rrr(instr); self.emit_u32(encode_fp_r_type(FP_DIV, FMT_S, d, s1, s2, RM_DYN)); }
            FSQRT_S => { let (d,s1) = self.rr(instr); self.emit_u32(encode_fp_r_type(FP_SQRT, FMT_S, d, s1, 0, RM_DYN)); }
            FMIN_S => { let (d,s1,s2) = self.rrr(instr); self.emit_u32(encode_fp_r_type(FP_MINMAX, FMT_S, d, s1, s2, 0b000)); }
            FMAX_S => { let (d,s1,s2) = self.rrr(instr); self.emit_u32(encode_fp_r_type(FP_MINMAX, FMT_S, d, s1, s2, 0b001)); }
            FEQ_S  => { let (d,s1,s2) = self.rrr(instr); self.emit_u32(encode_fp_r_type(FP_CMP, FMT_S, d, s1, s2, 0b010)); }
            FLT_S  => { let (d,s1,s2) = self.rrr(instr); self.emit_u32(encode_fp_r_type(FP_CMP, FMT_S, d, s1, s2, 0b001)); }
            FLE_S  => { let (d,s1,s2) = self.rrr(instr); self.emit_u32(encode_fp_r_type(FP_CMP, FMT_S, d, s1, s2, 0b000)); }
            FCVT_W_S  => { let (d,s) = self.rr(instr); self.emit_u32(encode_fp_r_type(FP_CVT_FROM_FP, FMT_S, d, s, 0, RM_DYN)); }
            FCVT_WU_S => { let (d,s) = self.rr(instr); self.emit_u32(encode_fp_r_type(FP_CVT_FROM_FP, FMT_S, d, s, 1, RM_DYN)); }
            FCVT_L_S  => { let (d,s) = self.rr(instr); self.emit_u32(encode_fp_r_type(FP_CVT_FROM_FP, FMT_S, d, s, 2, RM_DYN)); }
            FCVT_LU_S => { let (d,s) = self.rr(instr); self.emit_u32(encode_fp_r_type(FP_CVT_FROM_FP, FMT_S, d, s, 3, RM_DYN)); }
            FCVT_S_W  => { let (d,s) = self.rr(instr); self.emit_u32(encode_fp_r_type(FP_CVT_TO_FP, FMT_S, d, s, 0, RM_DYN)); }
            FCVT_S_WU => { let (d,s) = self.rr(instr); self.emit_u32(encode_fp_r_type(FP_CVT_TO_FP, FMT_S, d, s, 1, RM_DYN)); }
            FCVT_S_L  => { let (d,s) = self.rr(instr); self.emit_u32(encode_fp_r_type(FP_CVT_TO_FP, FMT_S, d, s, 2, RM_DYN)); }
            FCVT_S_LU => { let (d,s) = self.rr(instr); self.emit_u32(encode_fp_r_type(FP_CVT_TO_FP, FMT_S, d, s, 3, RM_DYN)); }
            FMV_X_W   => { let (d,s) = self.rr(instr); self.emit_u32(encode_fp_r_type(FP_MV_TO_INT, FMT_S, d, s, 0, 0b000)); }
            FMV_W_X   => { let (d,s) = self.rr(instr); self.emit_u32(encode_fp_r_type(FP_MV_TO_FP, FMT_S, d, s, 0, 0b000)); }
            // === D extension ===
            FLD  => { let (rd,(b,o)) = self.rm(instr); self.emit_u32(encode_i_type(OP_LOAD_FP, rd, FUNCT3_FLD, b, o)); }
            FSD  => { let (s,(b,o)) = self.rm(instr); self.emit_u32(encode_s_type(OP_STORE_FP, FUNCT3_FLD, b, s, o)); }
            FADD_D => { let (d,s1,s2) = self.rrr(instr); self.emit_u32(encode_fp_r_type(FP_ADD, FMT_D, d, s1, s2, RM_DYN)); }
            FSUB_D => { let (d,s1,s2) = self.rrr(instr); self.emit_u32(encode_fp_r_type(FP_SUB, FMT_D, d, s1, s2, RM_DYN)); }
            FMUL_D => { let (d,s1,s2) = self.rrr(instr); self.emit_u32(encode_fp_r_type(FP_MUL, FMT_D, d, s1, s2, RM_DYN)); }
            FDIV_D => { let (d,s1,s2) = self.rrr(instr); self.emit_u32(encode_fp_r_type(FP_DIV, FMT_D, d, s1, s2, RM_DYN)); }
            FSQRT_D => { let (d,s) = self.rr(instr); self.emit_u32(encode_fp_r_type(FP_SQRT, FMT_D, d, s, 0, RM_DYN)); }
            FMIN_D => { let (d,s1,s2) = self.rrr(instr); self.emit_u32(encode_fp_r_type(FP_MINMAX, FMT_D, d, s1, s2, 0b000)); }
            FMAX_D => { let (d,s1,s2) = self.rrr(instr); self.emit_u32(encode_fp_r_type(FP_MINMAX, FMT_D, d, s1, s2, 0b001)); }
            FEQ_D  => { let (d,s1,s2) = self.rrr(instr); self.emit_u32(encode_fp_r_type(FP_CMP, FMT_D, d, s1, s2, 0b010)); }
            FLT_D  => { let (d,s1,s2) = self.rrr(instr); self.emit_u32(encode_fp_r_type(FP_CMP, FMT_D, d, s1, s2, 0b001)); }
            FLE_D  => { let (d,s1,s2) = self.rrr(instr); self.emit_u32(encode_fp_r_type(FP_CMP, FMT_D, d, s1, s2, 0b000)); }
            FCVT_W_D  => { let (d,s) = self.rr(instr); self.emit_u32(encode_fp_r_type(FP_CVT_FROM_FP, FMT_D, d, s, 0, RM_DYN)); }
            FCVT_WU_D => { let (d,s) = self.rr(instr); self.emit_u32(encode_fp_r_type(FP_CVT_FROM_FP, FMT_D, d, s, 1, RM_DYN)); }
            FCVT_L_D  => { let (d,s) = self.rr(instr); self.emit_u32(encode_fp_r_type(FP_CVT_FROM_FP, FMT_D, d, s, 2, RM_DYN)); }
            FCVT_LU_D => { let (d,s) = self.rr(instr); self.emit_u32(encode_fp_r_type(FP_CVT_FROM_FP, FMT_D, d, s, 3, RM_DYN)); }
            FCVT_D_W  => { let (d,s) = self.rr(instr); self.emit_u32(encode_fp_r_type(FP_CVT_TO_FP, FMT_D, d, s, 0, RM_DYN)); }
            FCVT_D_WU => { let (d,s) = self.rr(instr); self.emit_u32(encode_fp_r_type(FP_CVT_TO_FP, FMT_D, d, s, 1, RM_DYN)); }
            FCVT_D_L  => { let (d,s) = self.rr(instr); self.emit_u32(encode_fp_r_type(FP_CVT_TO_FP, FMT_D, d, s, 2, RM_DYN)); }
            FCVT_D_LU => { let (d,s) = self.rr(instr); self.emit_u32(encode_fp_r_type(FP_CVT_TO_FP, FMT_D, d, s, 3, RM_DYN)); }
            FCVT_D_S  => { let (d,s) = self.rr(instr); self.emit_u32(encode_fp_r_type(FP_CVT_FP_FP, FMT_D, d, s, 0, RM_DYN)); }
            FCVT_S_D  => { let (d,s) = self.rr(instr); self.emit_u32(encode_fp_r_type(FP_CVT_FP_FP, FMT_S, d, s, 1, RM_DYN)); }
            FMV_X_D   => { let (d,s) = self.rr(instr); self.emit_u32(encode_fp_r_type(FP_MV_TO_INT, FMT_D, d, s, 0, 0b000)); }
            FMV_D_X   => { let (d,s) = self.rr(instr); self.emit_u32(encode_fp_r_type(FP_MV_TO_FP, FMT_D, d, s, 0, 0b000)); }
            // === System ===
            FENCE  => { self.emit_u32(encode_i_type(OP_FENCE, 0, FUNCT3_FENCE, 0, 0x0FF)); }
            ECALL  => { self.emit_u32(encode_i_type(OP_SYSTEM, 0, 0b000, 0, 0)); }
            EBREAK => { self.emit_u32(encode_i_type(OP_SYSTEM, 0, 0b000, 0, 1)); }
            // === Pseudo-instructions ===
            NOP  => { self.emit_u32(encode_i_type(OP_OP_IMM, 0, FUNCT3_ADDI, 0, 0)); }
            MV   => { let (d,s) = self.rr(instr); self.emit_u32(encode_i_type(OP_OP_IMM, d, FUNCT3_ADDI, s, 0)); }
            NEG  => { let (d,s) = self.rr(instr); self.emit_u32(encode_r_type(OP_OP, d, FUNCT3_ADD_SUB, 0, s, FUNCT7_ALT)); }
            NOT  => { let (d,s) = self.rr(instr); self.emit_u32(encode_i_type(OP_OP_IMM, d, FUNCT3_XORI, s, -1)); }
            SEQZ => { let (d,s) = self.rr(instr); self.emit_u32(encode_i_type(OP_OP_IMM, d, FUNCT3_SLTIU, s, 1)); }
            SNEZ => { let (d,s) = self.rr(instr); self.emit_u32(encode_r_type(OP_OP, d, FUNCT3_SLTU, 0, s, FUNCT7_NORMAL)); }
            LI   => {
                let rd = Self::get_reg(&instr.operands[0]);
                let imm = Self::get_imm64(&instr.operands[1]);
                self.materialize_constant(rd, imm);
            }
            CALL => {
                let symbol = if !instr.operands.is_empty() { Self::get_symbol(&instr.operands[0]).to_string() } else { String::new() };
                self.relocations.push(Relocation { offset: self.offset as u64, symbol, reloc_type: RelocationType::Riscv_Call, addend: 0, section_index: 0 });
                self.emit_u32(encode_u_type(OP_AUIPC, 1, 0));
                self.emit_u32(encode_i_type(OP_JALR, 1, FUNCT3_JALR, 1, 0));
            }
            RET => { self.emit_u32(encode_i_type(OP_JALR, 0, FUNCT3_JALR, 1, 0)); }
            LA  => {
                let rd = Self::get_reg(&instr.operands[0]);
                let symbol = if instr.operands.len() > 1 { Self::get_symbol(&instr.operands[1]).to_string() } else { String::new() };
                self.relocations.push(Relocation { offset: self.offset as u64, symbol: symbol.clone(), reloc_type: RelocationType::Riscv_Pcrel_Hi20, addend: 0, section_index: 0 });
                self.emit_u32(encode_u_type(OP_AUIPC, rd, 0));
                self.relocations.push(Relocation { offset: self.offset as u64, symbol, reloc_type: RelocationType::Riscv_Pcrel_Lo12_I, addend: 0, section_index: 0 });
                self.emit_u32(encode_i_type(OP_OP_IMM, rd, FUNCT3_ADDI, rd, 0));
            }
        }
    }

    // ---- operand shorthand helpers ----

    fn rrr(&self, i: &MachineInstr) -> (u32, u32, u32) {
        (Self::get_reg(&i.operands[0]), Self::get_reg(&i.operands[1]), Self::get_reg(&i.operands[2]))
    }
    fn rri(&self, i: &MachineInstr) -> (u32, u32, i32) {
        (Self::get_reg(&i.operands[0]), Self::get_reg(&i.operands[1]), Self::get_imm(&i.operands[2]))
    }
    fn rr(&self, i: &MachineInstr) -> (u32, u32) {
        (Self::get_reg(&i.operands[0]), Self::get_reg(&i.operands[1]))
    }
    fn rm(&self, i: &MachineInstr) -> (u32, (u32, i32)) {
        (Self::get_reg(&i.operands[0]), Self::get_mem(&i.operands[1]))
    }

    // ---- branch resolution ----

    fn resolve_labels(&mut self) {
        // Collect all patches first to avoid overlapping borrows on self
        let patches: Vec<(usize, FixupType, u32, i64)> = self.fixups.iter().filter_map(|fixup| {
            let target = *self.label_offsets.get(&fixup.label)?;
            let pc = fixup.code_offset;
            let rel = (target as i64) - (pc as i64);
            let orig = u32::from_le_bytes([
                self.code[pc], self.code[pc + 1],
                self.code[pc + 2], self.code[pc + 3],
            ]);
            Some((pc, fixup.fixup_type, orig, rel))
        }).collect();

        for (pc, ft, orig, rel) in patches {
            match ft {
                FixupType::BranchB => {
                    let rs1 = (orig >> 15) & 0x1F;
                    let rs2 = (orig >> 20) & 0x1F;
                    let f3 = (orig >> 12) & 0x7;
                    self.patch_u32(pc, encode_b_type(OP_BRANCH, f3, rs1, rs2, rel as i32));
                }
                FixupType::JumpJ => {
                    let rd = (orig >> 7) & 0x1F;
                    self.patch_u32(pc, encode_j_type(OP_JAL, rd, rel as i32));
                }
                FixupType::CallPair => { /* Handled via R_RISCV_CALL relocation by the linker */ }
            }
        }
    }
}

// ====================== Unit Tests ======================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{MachineInstr, MachineOperand, PhysReg};

    fn rl(bytes: &[u8]) -> u32 { u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) }

    fn mk(op: Riscv64Opcode, operands: Vec<MachineOperand>) -> MachineInstr {
        MachineInstr { opcode: op.as_u32(), operands, loc: None }
    }
    fn r(n: u16) -> MachineOperand { MachineOperand::Register(PhysReg(n)) }
    fn im(v: i64) -> MachineOperand { MachineOperand::Immediate(v) }
    fn mem(b: u16, o: i32) -> MachineOperand { MachineOperand::Memory { base: PhysReg(b), offset: o } }

    // R-type
    #[test] fn test_add() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::ADD, vec![r(1),r(2),r(3)])]);
        let w = rl(&c);
        assert_eq!(w & 0x7F, OP_OP);
        assert_eq!((w>>7)&0x1F, 1);
        assert_eq!((w>>15)&0x1F, 2);
        assert_eq!((w>>20)&0x1F, 3);
        assert_eq!((w>>25)&0x7F, 0);
    }
    #[test] fn test_sub() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::SUB, vec![r(5),r(6),r(7)])]);
        assert_eq!((rl(&c)>>25)&0x7F, FUNCT7_ALT);
    }
    #[test] fn test_mul() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::MUL, vec![r(10),r(11),r(12)])]);
        let w = rl(&c);
        assert_eq!((w>>25)&0x7F, FUNCT7_MULDIV);
        assert_eq!((w>>12)&0x7, FUNCT3_MUL);
    }
    // I-type
    #[test] fn test_addi() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::ADDI, vec![r(1),r(2),im(100)])]);
        let w = rl(&c);
        assert_eq!((w as i32)>>20, 100);
    }
    #[test] fn test_addi_neg() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::ADDI, vec![r(1),r(2),im(-1)])]);
        assert_eq!((rl(&c) as i32)>>20, -1);
    }
    #[test] fn test_lw() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::LW, vec![r(1),mem(2,8)])]);
        let w = rl(&c);
        assert_eq!(w & 0x7F, OP_LOAD);
        assert_eq!((w>>12)&0x7, FUNCT3_LW);
        assert_eq!((w as i32)>>20, 8);
    }
    #[test] fn test_ld() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::LD, vec![r(1),mem(2,0)])]);
        assert_eq!((rl(&c)>>12)&0x7, FUNCT3_LD);
    }
    // S-type
    #[test] fn test_sw() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::SW, vec![r(3),mem(2,16)])]);
        let w = rl(&c);
        let lo = (w>>7)&0x1F;
        let hi = (w>>25)&0x7F;
        assert_eq!((hi<<5)|lo, 16);
    }
    // B-type
    #[test] fn test_beq_pos() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::BEQ, vec![r(1),r(2),im(4)])]);
        let w = rl(&c);
        let b = ((w>>31)&1)<<12 | ((w>>7)&1)<<11 | ((w>>25)&0x3F)<<5 | ((w>>8)&0xF)<<1;
        assert_eq!(b, 4);
    }
    #[test] fn test_bne_neg() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::BNE, vec![r(3),r(4),im(-8)])]);
        let w = rl(&c);
        let raw = ((w>>31)&1)<<12 | ((w>>7)&1)<<11 | ((w>>25)&0x3F)<<5 | ((w>>8)&0xF)<<1;
        let s = if raw & 0x1000 != 0 { (raw|0xFFFFE000) as i32 } else { raw as i32 };
        assert_eq!(s, -8);
    }
    // U-type
    #[test] fn test_lui() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::LUI, vec![r(1),im(0x12345)])]);
        let w = rl(&c);
        assert_eq!(w & 0x7F, OP_LUI);
        assert_eq!(w & 0xFFFFF000, 0x12345000);
    }
    #[test] fn test_auipc() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::AUIPC, vec![r(2),im(0xABCDE_i64)])]);
        assert_eq!(rl(&c) & 0xFFFFF000, 0xABCDE000);
    }
    // J-type
    #[test] fn test_jal_pos() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::JAL, vec![r(1),im(8)])]);
        let w = rl(&c);
        let j = ((w>>31)&1)<<20 | ((w>>12)&0xFF)<<12 | ((w>>20)&1)<<11 | ((w>>21)&0x3FF)<<1;
        assert_eq!(j, 8);
    }
    #[test] fn test_jal_neg() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::JAL, vec![r(0),im(-4)])]);
        let w = rl(&c);
        let raw = ((w>>31)&1)<<20 | ((w>>12)&0xFF)<<12 | ((w>>20)&1)<<11 | ((w>>21)&0x3FF)<<1;
        let s = if raw&0x100000 != 0 { (raw|0xFFE00000) as i32 } else { raw as i32 };
        assert_eq!(s, -4);
    }
    // Pseudo
    #[test] fn test_nop() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::NOP, vec![])]);
        assert_eq!(c, vec![0x13, 0x00, 0x00, 0x00]);
    }
    #[test] fn test_ret() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::RET, vec![])]);
        let w = rl(&c);
        assert_eq!(w & 0x7F, OP_JALR);
        assert_eq!((w>>7)&0x1F, 0);
        assert_eq!((w>>15)&0x1F, 1);
    }
    #[test] fn test_mv() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::MV, vec![r(5),r(6)])]);
        let w = rl(&c);
        assert_eq!(w&0x7F, OP_OP_IMM);
        assert_eq!((w>>7)&0x1F, 5);
        assert_eq!((w>>15)&0x1F, 6);
    }
    #[test] fn test_neg() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::NEG, vec![r(5),r(6)])]);
        let w = rl(&c);
        assert_eq!((w>>15)&0x1F, 0);
        assert_eq!((w>>20)&0x1F, 6);
        assert_eq!((w>>25)&0x7F, FUNCT7_ALT);
    }
    // FP
    #[test] fn test_fadd_d() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::FADD_D, vec![r(32),r(33),r(34)])]);
        let w = rl(&c);
        assert_eq!(w&0x7F, OP_OP_FP);
        assert_eq!((w>>7)&0x1F, 0);
        assert_eq!((w>>15)&0x1F, 1);
        assert_eq!((w>>20)&0x1F, 2);
        assert_eq!((w>>25)&0x7F, 0b0000001);
    }
    #[test] fn test_fld() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::FLD, vec![r(35),mem(4,0)])]);
        let w = rl(&c);
        assert_eq!(w&0x7F, OP_LOAD_FP);
        assert_eq!((w>>7)&0x1F, 3);
        assert_eq!((w>>12)&0x7, FUNCT3_FLD);
    }
    // Relocation
    #[test] fn test_call_reloc() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::CALL, vec![MachineOperand::Symbol("printf".into())])]);
        assert_eq!(c.len(), 8);
        assert_eq!(e.get_relocations().len(), 1);
        assert_eq!(e.get_relocations()[0].reloc_type, RelocationType::Riscv_Call);
    }
    #[test] fn test_la_reloc() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::LA, vec![r(5), MachineOperand::Symbol("g".into())])]);
        assert_eq!(c.len(), 8);
        let rl = e.get_relocations();
        assert_eq!(rl.len(), 2);
        assert_eq!(rl[0].reloc_type, RelocationType::Riscv_Pcrel_Hi20);
        assert_eq!(rl[1].reloc_type, RelocationType::Riscv_Pcrel_Lo12_I);
    }
    // Branch fixup
    #[test] fn test_branch_fixup() {
        let mut e = Riscv64Encoder::new();
        let instrs = vec![
            mk(Riscv64Opcode::BEQ, vec![r(1),r(2),MachineOperand::Label(0)]),
            mk(Riscv64Opcode::NOP, vec![]),
            MachineInstr { opcode: Riscv64Opcode::NOP.as_u32(), operands: vec![MachineOperand::Label(0)], loc: None },
        ];
        let c = e.encode_function(&instrs);
        assert_eq!(c.len(), 8);
        let w = rl(&c[0..4]);
        let raw = ((w>>31)&1)<<12 | ((w>>7)&1)<<11 | ((w>>25)&0x3F)<<5 | ((w>>8)&0xF)<<1;
        let s = if raw&0x1000!=0 { (raw|0xFFFFE000) as i32 } else { raw as i32 };
        assert_eq!(s, 8);
    }
    // LI
    #[test] fn test_li_small() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::LI, vec![r(5),im(42)])]);
        assert_eq!(c.len(), 4);
        let w = rl(&c);
        assert_eq!((w as i32)>>20, 42);
    }
    // Shifts
    #[test] fn test_slli() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::SLLI, vec![r(1),r(2),im(5)])]);
        let w = rl(&c);
        assert_eq!((w>>12)&0x7, FUNCT3_SLLI);
        assert_eq!((w>>20)&0x3F, 5);
    }
    #[test] fn test_srai() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::SRAI, vec![r(3),r(4),im(10)])]);
        let w = rl(&c);
        assert_eq!((w>>20)&0x3F, 10);
        assert_ne!(w & (1<<30), 0);
    }
    // W-suffix
    #[test] fn test_addw() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::ADDW, vec![r(1),r(2),r(3)])]);
        assert_eq!(rl(&c)&0x7F, OP_OP_32);
    }
    // System
    #[test] fn test_ecall() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::ECALL, vec![])]);
        let w = rl(&c);
        assert_eq!(w&0x7F, OP_SYSTEM);
        assert_eq!((w>>20)&0xFFF, 0);
    }
    #[test] fn test_ebreak() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::EBREAK, vec![])]);
        assert_eq!((rl(&c)>>20)&0xFFF, 1);
    }
    #[test] fn test_code_size() {
        let mut e = Riscv64Encoder::new();
        e.encode_function(&[mk(Riscv64Opcode::NOP,vec![]),mk(Riscv64Opcode::NOP,vec![]),mk(Riscv64Opcode::RET,vec![])]);
        assert_eq!(e.code_size(), 12);
    }
    #[test] fn test_little_endian() {
        let mut e = Riscv64Encoder::new();
        let c = e.encode_function(&[mk(Riscv64Opcode::NOP, vec![])]);
        assert_eq!(c, vec![0x13, 0x00, 0x00, 0x00]);
    }
}
