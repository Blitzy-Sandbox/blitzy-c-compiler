//! AArch64 integrated assembler — machine code encoder.
//!
//! This module encodes AArch64 (A64 / ARMv8-A 64-bit) machine instructions
//! into binary byte sequences.  Every AArch64 instruction is exactly 32 bits
//! (4 bytes) wide, stored in **little-endian** byte order for ELF64 emission.
//!
//! AArch64 instructions are organised into major encoding groups determined
//! by bits `[28:25]` of the 32-bit instruction word:
//!
//! | Bits `[28:25]` | Group                                |
//! |----------------|--------------------------------------|
//! | `100x`         | Data Processing — Immediate          |
//! | `101x`         | Branches / Exception / System        |
//! | `x1x0`         | Loads and Stores                     |
//! | `x101`         | Data Processing — Register           |
//! | `x111`         | SIMD and Floating-Point              |
//!
//! # Zero External Dependencies
//!
//! Only imports from `std` and internal crate modules.  No external crates.

use std::collections::HashMap;

#[allow(unused_imports)]
use crate::codegen::aarch64::isel::{Aarch64Condition, Aarch64Opcode, ShiftType};
use crate::codegen::regalloc::PhysReg;
use crate::codegen::{MachineInstr, MachineOperand, Relocation, RelocationType};

// ---------------------------------------------------------------------------
// Internal types for branch fixup resolution
// ---------------------------------------------------------------------------

/// Describes a forward/backward branch reference that needs patching once
/// all label offsets are known.
struct BranchFixup {
    /// The label ID of the target basic block.
    label: u32,
    /// Byte offset in the code buffer where the instruction was emitted.
    code_offset: usize,
    /// The kind of branch instruction, which determines the bit-field
    /// width and encoding position of the offset.
    fixup_type: AArch64FixupType,
}

/// Classifies the branch instruction so the fixup pass knows which bits
/// to patch and what range constraint applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AArch64FixupType {
    /// B / BL — 26-bit signed offset in 4-byte units (±128 MB).
    Branch26,
    /// B.cond / CBZ / CBNZ — 19-bit signed offset in 4-byte units (±1 MB).
    CondBranch19,
    /// TBZ / TBNZ — 14-bit signed offset in 4-byte units (±32 KB).
    TestBranch14,
    /// ADRP — 21-bit page offset (±4 GB).  Rarely fixed up locally
    /// (usually handled via relocations), but included for completeness.
    AdrpPage21,
}

// ---------------------------------------------------------------------------
// Aarch64Encoder — the main public type
// ---------------------------------------------------------------------------

/// AArch64 machine code encoder / integrated assembler.
///
/// The encoder converts a stream of [`MachineInstr`] values (produced by
/// instruction selection) into raw bytes.  It performs two passes:
///
/// 1. **Emission pass** — encodes each instruction, recording branch fixups
///    and label positions.
/// 2. **Fixup pass** — resolves forward branch references by patching the
///    offset fields of branch instructions.
///
/// Relocations for external symbols (function calls, global data) are
/// accumulated and can be retrieved via [`get_relocations`](Self::get_relocations).
pub struct Aarch64Encoder {
    /// Output byte buffer for encoded instructions.
    code: Vec<u8>,
    /// Current byte offset within the code buffer.
    offset: usize,
    /// Relocations emitted during encoding (for the linker).
    relocations: Vec<Relocation>,
    /// Maps label IDs to their resolved byte offsets in the code buffer.
    label_offsets: HashMap<u32, usize>,
    /// Branch instructions whose target label has not yet been seen.
    fixups: Vec<BranchFixup>,
}

impl Aarch64Encoder {
    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Create a new, empty encoder.
    pub fn new() -> Self {
        Self {
            code: Vec::new(),
            offset: 0,
            relocations: Vec::new(),
            label_offsets: HashMap::new(),
            fixups: Vec::new(),
        }
    }

    /// Encode an entire function's worth of instructions and return the
    /// machine code bytes.
    ///
    /// This resets the encoder state, performs the two-pass encode, and
    /// returns a freshly allocated `Vec<u8>`.
    pub fn encode_function(&mut self, instrs: &[MachineInstr]) -> Vec<u8> {
        // Reset state for a fresh function.
        self.code.clear();
        self.offset = 0;
        self.relocations.clear();
        self.label_offsets.clear();
        self.fixups.clear();

        // --- Pass 1: emit all instructions ---
        for instr in instrs {
            self.encode_instruction(instr);
        }

        // --- Pass 2: resolve branch fixups ---
        self.resolve_labels();

        self.code.clone()
    }

    /// Encode a single [`MachineInstr`] into the code buffer.
    ///
    /// The instruction's `opcode` field is decoded via
    /// [`Aarch64Opcode::from_u32`] and dispatched to the appropriate
    /// encoding-group function.
    pub fn encode_instruction(&mut self, instr: &MachineInstr) {
        let opcode = match Aarch64Opcode::from_u32(instr.opcode) {
            Some(op) => op,
            None => {
                // Unknown opcode — emit a NOP as a safe fallback and
                // continue.  The driver will have already validated
                // instruction selection, so this should not happen.
                self.emit_u32(0xD503201F); // NOP encoding
                return;
            }
        };

        match opcode {
            // =============================================================
            // Integer Arithmetic — ADD / ADDS / SUB / SUBS
            // =============================================================
            Aarch64Opcode::ADD | Aarch64Opcode::ADDS | Aarch64Opcode::SUB | Aarch64Opcode::SUBS => {
                self.encode_add_sub(opcode, &instr.operands);
            }

            // =============================================================
            // Multiply / Divide
            // =============================================================
            Aarch64Opcode::MUL => {
                // MUL Xd, Xn, Xm  ≡  MADD Xd, Xn, Xm, XZR
                let (rd, rn, rm) = self.extract_3reg(&instr.operands);
                let sf = self.sf_from_operands(&instr.operands);
                let enc = self.encode_dp_3src(sf, 0, 0b000, rm, 0, 31, rn, rd);
                self.emit_u32(enc);
            }
            Aarch64Opcode::MADD => {
                let (rd, rn, rm, ra) = self.extract_4reg(&instr.operands);
                let sf = self.sf_from_operands(&instr.operands);
                let enc = self.encode_dp_3src(sf, 0, 0b000, rm, 0, ra, rn, rd);
                self.emit_u32(enc);
            }
            Aarch64Opcode::MSUB => {
                let (rd, rn, rm, ra) = self.extract_4reg(&instr.operands);
                let sf = self.sf_from_operands(&instr.operands);
                let enc = self.encode_dp_3src(sf, 0, 0b000, rm, 1, ra, rn, rd);
                self.emit_u32(enc);
            }
            Aarch64Opcode::SDIV => {
                let (rd, rn, rm) = self.extract_3reg(&instr.operands);
                let sf = self.sf_from_operands(&instr.operands);
                let enc = self.encode_dp_2src_fn(sf, 0, rm, 0b000011, rn, rd);
                self.emit_u32(enc);
            }
            Aarch64Opcode::UDIV => {
                let (rd, rn, rm) = self.extract_3reg(&instr.operands);
                let sf = self.sf_from_operands(&instr.operands);
                let enc = self.encode_dp_2src_fn(sf, 0, rm, 0b000010, rn, rd);
                self.emit_u32(enc);
            }

            // =============================================================
            // Logical — AND / ANDS / ORR / EOR
            // =============================================================
            Aarch64Opcode::AND | Aarch64Opcode::ANDS | Aarch64Opcode::ORR | Aarch64Opcode::EOR => {
                self.encode_logical(opcode, &instr.operands);
            }

            // =============================================================
            // Shifts — LSL / LSR / ASR  (register and immediate forms)
            // =============================================================
            Aarch64Opcode::LSL | Aarch64Opcode::LSR | Aarch64Opcode::ASR => {
                self.encode_shift(opcode, &instr.operands);
            }

            // =============================================================
            // Move — MOV / MOVZ / MOVK / MOVN
            // =============================================================
            Aarch64Opcode::MOV => {
                self.encode_mov(&instr.operands);
            }
            Aarch64Opcode::MOVZ => {
                self.encode_movzk(opcode, &instr.operands);
            }
            Aarch64Opcode::MOVK => {
                self.encode_movzk(opcode, &instr.operands);
            }
            Aarch64Opcode::MOVN => {
                self.encode_movzk(opcode, &instr.operands);
            }

            // =============================================================
            // Conditional select — CSEL / CSINC / CSINV / CSNEG
            // =============================================================
            Aarch64Opcode::CSEL
            | Aarch64Opcode::CSINC
            | Aarch64Opcode::CSINV
            | Aarch64Opcode::CSNEG => {
                self.encode_cond_sel(opcode, &instr.operands);
            }

            // =============================================================
            // Load / Store — integer
            // =============================================================
            Aarch64Opcode::LDR | Aarch64Opcode::STR => {
                self.encode_ldst_64(&instr.operands, opcode == Aarch64Opcode::LDR);
            }
            Aarch64Opcode::LDRW | Aarch64Opcode::STRW => {
                self.encode_ldst_32(&instr.operands, opcode == Aarch64Opcode::LDRW);
            }
            Aarch64Opcode::LDRH | Aarch64Opcode::STRH => {
                self.encode_ldst_16(&instr.operands, opcode == Aarch64Opcode::LDRH);
            }
            Aarch64Opcode::LDRB | Aarch64Opcode::STRB => {
                self.encode_ldst_8(&instr.operands, opcode == Aarch64Opcode::LDRB);
            }
            Aarch64Opcode::LDRSW => {
                // size=10, opc=10 (sign-extend to 64)
                self.encode_ldst_sign_ext(&instr.operands, 0b10, 0b10);
            }
            Aarch64Opcode::LDRSH => {
                // size=01, opc=10 (sign-extend to 64)
                self.encode_ldst_sign_ext(&instr.operands, 0b01, 0b10);
            }
            Aarch64Opcode::LDRSB => {
                // size=00, opc=10 (sign-extend to 64)
                self.encode_ldst_sign_ext(&instr.operands, 0b00, 0b10);
            }

            // =============================================================
            // Load / Store pair — LDP / STP
            // =============================================================
            Aarch64Opcode::LDP => {
                self.encode_ldp_stp(&instr.operands, true);
            }
            Aarch64Opcode::STP => {
                self.encode_ldp_stp(&instr.operands, false);
            }

            // =============================================================
            // PC-relative addressing — ADRP / ADR
            // =============================================================
            Aarch64Opcode::ADRP => {
                self.encode_adrp(&instr.operands);
            }
            Aarch64Opcode::ADR => {
                self.encode_adr(&instr.operands);
            }

            // =============================================================
            // Branches — B / BL / BR / BLR / RET / B_cond / CBZ / CBNZ
            // =============================================================
            Aarch64Opcode::B => {
                self.encode_branch_b(&instr.operands);
            }
            Aarch64Opcode::BL => {
                self.encode_branch_bl(&instr.operands);
            }
            Aarch64Opcode::BR => {
                let rn = self.extract_reg(&instr.operands, 0);
                let enc = self.encode_branch_reg_fn(0b0000, rn);
                self.emit_u32(enc);
            }
            Aarch64Opcode::BLR => {
                let rn = self.extract_reg(&instr.operands, 0);
                let enc = self.encode_branch_reg_fn(0b0001, rn);
                self.emit_u32(enc);
            }
            Aarch64Opcode::RET => {
                // RET defaults to X30 (link register) if no operand given.
                let rn = if instr.operands.is_empty() {
                    30 // X30 = LR
                } else {
                    self.extract_reg(&instr.operands, 0)
                };
                let enc = self.encode_branch_reg_fn(0b0010, rn);
                self.emit_u32(enc);
            }
            Aarch64Opcode::B_cond => {
                self.encode_b_cond(&instr.operands);
            }
            Aarch64Opcode::CBZ => {
                self.encode_cbz_cbnz(&instr.operands, false);
            }
            Aarch64Opcode::CBNZ => {
                self.encode_cbz_cbnz(&instr.operands, true);
            }

            // =============================================================
            // NOP
            // =============================================================
            Aarch64Opcode::NOP => {
                self.emit_u32(0xD503201F);
            }

            // =============================================================
            // FP arithmetic — FADD / FSUB / FMUL / FDIV
            // =============================================================
            Aarch64Opcode::FADD_S => self.encode_fp_arith2(&instr.operands, 0b00, 0b0010),
            Aarch64Opcode::FADD_D => self.encode_fp_arith2(&instr.operands, 0b01, 0b0010),
            Aarch64Opcode::FSUB_S => self.encode_fp_arith2(&instr.operands, 0b00, 0b0011),
            Aarch64Opcode::FSUB_D => self.encode_fp_arith2(&instr.operands, 0b01, 0b0011),
            Aarch64Opcode::FMUL_S => self.encode_fp_arith2(&instr.operands, 0b00, 0b0000),
            Aarch64Opcode::FMUL_D => self.encode_fp_arith2(&instr.operands, 0b01, 0b0000),
            Aarch64Opcode::FDIV_S => self.encode_fp_arith2(&instr.operands, 0b00, 0b0001),
            Aarch64Opcode::FDIV_D => self.encode_fp_arith2(&instr.operands, 0b01, 0b0001),
            Aarch64Opcode::FMAX_S => self.encode_fp_arith2(&instr.operands, 0b00, 0b1000),
            Aarch64Opcode::FMAX_D => self.encode_fp_arith2(&instr.operands, 0b01, 0b1000),
            Aarch64Opcode::FMIN_S => self.encode_fp_arith2(&instr.operands, 0b00, 0b1001),
            Aarch64Opcode::FMIN_D => self.encode_fp_arith2(&instr.operands, 0b01, 0b1001),

            // FP unary
            Aarch64Opcode::FSQRT_S => self.encode_fp_unary(&instr.operands, 0b00, 0b000011),
            Aarch64Opcode::FSQRT_D => self.encode_fp_unary(&instr.operands, 0b01, 0b000011),
            Aarch64Opcode::FABS_S => self.encode_fp_unary(&instr.operands, 0b00, 0b000001),
            Aarch64Opcode::FABS_D => self.encode_fp_unary(&instr.operands, 0b01, 0b000001),
            Aarch64Opcode::FNEG_S => self.encode_fp_unary(&instr.operands, 0b00, 0b000010),
            Aarch64Opcode::FNEG_D => self.encode_fp_unary(&instr.operands, 0b01, 0b000010),

            // FP move
            Aarch64Opcode::FMOV => {
                self.encode_fmov(&instr.operands);
            }

            // FP compare
            Aarch64Opcode::FCMP_S => self.encode_fp_compare(&instr.operands, 0b00),
            Aarch64Opcode::FCMP_D => self.encode_fp_compare(&instr.operands, 0b01),

            // FP conditional select
            Aarch64Opcode::FCSEL_S => self.encode_fcsel(&instr.operands, 0b00),
            Aarch64Opcode::FCSEL_D => self.encode_fcsel(&instr.operands, 0b01),

            // FP conversion: int → float
            Aarch64Opcode::SCVTF_S => self.encode_fp_int_conv(&instr.operands, 0b00, 0b00, 0b010),
            Aarch64Opcode::SCVTF_D => self.encode_fp_int_conv(&instr.operands, 0b01, 0b00, 0b010),
            Aarch64Opcode::UCVTF_S => self.encode_fp_int_conv(&instr.operands, 0b00, 0b00, 0b011),
            Aarch64Opcode::UCVTF_D => self.encode_fp_int_conv(&instr.operands, 0b01, 0b00, 0b011),

            // FP conversion: float → int
            Aarch64Opcode::FCVTZS_S => self.encode_fp_int_conv(&instr.operands, 0b00, 0b11, 0b000),
            Aarch64Opcode::FCVTZS_D => self.encode_fp_int_conv(&instr.operands, 0b01, 0b11, 0b000),
            Aarch64Opcode::FCVTZU_S => self.encode_fp_int_conv(&instr.operands, 0b00, 0b11, 0b001),
            Aarch64Opcode::FCVTZU_D => self.encode_fp_int_conv(&instr.operands, 0b01, 0b11, 0b001),

            // FP conversion: float ↔ float
            Aarch64Opcode::FCVT_S_TO_D => self.encode_fp_cvt_precision(&instr.operands, 0b00, 0b01),
            Aarch64Opcode::FCVT_D_TO_S => self.encode_fp_cvt_precision(&instr.operands, 0b01, 0b00),

            // FP loads / stores
            Aarch64Opcode::LDR_S => self.encode_fp_ldst(&instr.operands, 0b10, 0b01),
            Aarch64Opcode::LDR_D => self.encode_fp_ldst(&instr.operands, 0b11, 0b01),
            Aarch64Opcode::STR_S => self.encode_fp_ldst(&instr.operands, 0b10, 0b00),
            Aarch64Opcode::STR_D => self.encode_fp_ldst(&instr.operands, 0b11, 0b00),
            Aarch64Opcode::LDP_D => self.encode_fp_ldp_stp(&instr.operands, true),
            Aarch64Opcode::STP_D => self.encode_fp_ldp_stp(&instr.operands, false),

            // =============================================================
            // System — SVC / barriers
            // =============================================================
            Aarch64Opcode::SVC => {
                let imm16 = self.extract_imm(&instr.operands, 0) as u32;
                // SVC: 11010100 000 imm16 000 01
                let enc = (0b11010100_000u32 << 21) | ((imm16 & 0xFFFF) << 5) | 0b00001;
                self.emit_u32(enc);
            }
            Aarch64Opcode::DMB => {
                // DMB ISH: 0xD5033BBF
                self.emit_u32(0xD5033BBF);
            }
            Aarch64Opcode::DSB => {
                // DSB ISH: 0xD5033B9F
                self.emit_u32(0xD5033B9F);
            }
            Aarch64Opcode::ISB => {
                // ISB: 0xD5033FDF
                self.emit_u32(0xD5033FDF);
            }

            // =============================================================
            // Remaining opcodes that dispatch via alias logic
            // =============================================================
            _ => {
                // Remaining opcodes handled by individual helpers.
                self.encode_remaining(opcode, &instr.operands);
            }
        }
    }

    /// Returns a slice of all relocations emitted so far.
    pub fn get_relocations(&self) -> &[Relocation] {
        &self.relocations
    }

    /// Returns the current total size of encoded machine code in bytes.
    pub fn code_size(&self) -> usize {
        self.offset
    }

    // -----------------------------------------------------------------------
    // Byte emission utilities
    // -----------------------------------------------------------------------

    /// Emit a 32-bit word in little-endian byte order.
    fn emit_u32(&mut self, value: u32) {
        self.code.push((value & 0xFF) as u8);
        self.code.push(((value >> 8) & 0xFF) as u8);
        self.code.push(((value >> 16) & 0xFF) as u8);
        self.code.push(((value >> 24) & 0xFF) as u8);
        self.offset += 4;
    }

    /// Overwrite 4 bytes at `offset` with `value` in little-endian order.
    /// Used for branch fixups after all labels are known.
    fn patch_u32(&mut self, offset: usize, value: u32) {
        if offset + 4 <= self.code.len() {
            self.code[offset] = (value & 0xFF) as u8;
            self.code[offset + 1] = ((value >> 8) & 0xFF) as u8;
            self.code[offset + 2] = ((value >> 16) & 0xFF) as u8;
            self.code[offset + 3] = ((value >> 24) & 0xFF) as u8;
        }
    }

    /// Read back a little-endian u32 from the code buffer at `offset`.
    fn read_u32(&self, offset: usize) -> u32 {
        if offset + 4 <= self.code.len() {
            (self.code[offset] as u32)
                | ((self.code[offset + 1] as u32) << 8)
                | ((self.code[offset + 2] as u32) << 16)
                | ((self.code[offset + 3] as u32) << 24)
        } else {
            0
        }
    }

    /// Record a label at the current code offset.
    fn record_label(&mut self, label: u32) {
        self.label_offsets.insert(label, self.offset);
    }

    // -----------------------------------------------------------------------
    // Register extraction helpers
    // -----------------------------------------------------------------------

    /// Map a [`PhysReg`] to a 5-bit register encoding (0–31).
    ///
    /// - GPR X0–X30 ⇒ PhysReg(0)–PhysReg(30) → encoding 0–30
    /// - SP / XZR   ⇒ PhysReg(31)             → encoding 31
    /// - SIMD V0–V31 ⇒ PhysReg(32)–PhysReg(63) → encoding 0–31
    fn reg_num(reg: &PhysReg) -> u32 {
        let raw = reg.0 as u32;
        if raw >= 32 {
            // SIMD/FP register: subtract 32 to get 0-31.
            raw - 32
        } else {
            raw
        }
    }

    /// Returns `true` if the physical register is a SIMD/FP register
    /// (PhysReg(32) through PhysReg(63)).
    fn is_fp_phys(reg: &PhysReg) -> bool {
        reg.0 >= 32
    }

    /// Determine the `sf` (size flag) bit from operands.
    /// sf=1 for 64-bit (X registers), sf=0 for 32-bit (W registers).
    /// We default to 64-bit (sf=1) unless the instruction selection
    /// explicitly provides 32-bit operand information.
    fn sf_from_operands(&self, operands: &[MachineOperand]) -> u32 {
        // Convention: if there is an operand with Immediate value that
        // encodes the size, we look for it.  Otherwise default to 64-bit.
        // The isel module can indicate 32-bit by setting a size marker
        // as the last operand (Immediate(32) vs Immediate(64)).
        // By default we assume 64-bit.
        for op in operands.iter().rev() {
            if let MachineOperand::Immediate(val) = op {
                if *val == 32 {
                    return 0; // W-register, 32-bit
                }
                if *val == 64 {
                    return 1; // X-register, 64-bit
                }
            }
        }
        1 // default to 64-bit
    }

    /// Extract register number from operand at index `idx`.
    /// Returns the 5-bit encoding (0–31).
    fn extract_reg(&self, operands: &[MachineOperand], idx: usize) -> u32 {
        match operands.get(idx) {
            Some(MachineOperand::Register(r)) => Self::reg_num(r),
            _ => 31, // Default to XZR/SP if missing.
        }
    }

    /// Extract an immediate value from operand at index `idx`.
    fn extract_imm(&self, operands: &[MachineOperand], idx: usize) -> i64 {
        match operands.get(idx) {
            Some(MachineOperand::Immediate(val)) => *val,
            _ => 0,
        }
    }

    /// Extract 3 register numbers: (Rd, Rn, Rm) from first 3 operands.
    fn extract_3reg(&self, operands: &[MachineOperand]) -> (u32, u32, u32) {
        (
            self.extract_reg(operands, 0),
            self.extract_reg(operands, 1),
            self.extract_reg(operands, 2),
        )
    }

    /// Extract 4 register numbers: (Rd, Rn, Rm, Ra) from first 4 operands.
    fn extract_4reg(&self, operands: &[MachineOperand]) -> (u32, u32, u32, u32) {
        (
            self.extract_reg(operands, 0),
            self.extract_reg(operands, 1),
            self.extract_reg(operands, 2),
            self.extract_reg(operands, 3),
        )
    }

    /// Extract a memory operand (base register + offset) at `idx`.
    fn extract_memory(&self, operands: &[MachineOperand], idx: usize) -> (u32, i32) {
        match operands.get(idx) {
            Some(MachineOperand::Memory { base, offset }) => (Self::reg_num(base), *offset),
            _ => (31, 0), // Default to [SP, #0]
        }
    }

    /// Extract a Label operand at `idx`.
    fn extract_label(&self, operands: &[MachineOperand], idx: usize) -> Option<u32> {
        match operands.get(idx) {
            Some(MachineOperand::Label(lbl)) => Some(*lbl),
            _ => None,
        }
    }

    /// Extract a Symbol operand at `idx`.
    fn extract_symbol<'a>(&self, operands: &'a [MachineOperand], idx: usize) -> Option<&'a str> {
        match operands.get(idx) {
            Some(MachineOperand::Symbol(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // Low-level encoding group functions
    // -----------------------------------------------------------------------
    // These produce a u32 word with the correct bit-field packing.
    // They do NOT write to the code buffer — the caller invokes emit_u32().

    /// ADD/SUB immediate:
    /// `[sf(1)][op(1)][S(1)][100010][sh(1)][imm12(12)][Rn(5)][Rd(5)]`
    fn encode_add_sub_imm_fn(
        sf: u32,
        op: u32,
        s: u32,
        sh: u32,
        imm12: u32,
        rn: u32,
        rd: u32,
    ) -> u32 {
        (sf << 31)
            | (op << 30)
            | (s << 29)
            | (0b100010 << 23)
            | ((sh & 1) << 22)
            | ((imm12 & 0xFFF) << 10)
            | ((rn & 0x1F) << 5)
            | (rd & 0x1F)
    }

    /// Logical immediate:
    /// `[sf(1)][opc(2)][100100][N(1)][immr(6)][imms(6)][Rn(5)][Rd(5)]`
    fn encode_logical_imm_fn(
        sf: u32,
        opc: u32,
        n: u32,
        immr: u32,
        imms: u32,
        rn: u32,
        rd: u32,
    ) -> u32 {
        (sf << 31)
            | (opc << 29)
            | (0b100100 << 23)
            | ((n & 1) << 22)
            | ((immr & 0x3F) << 16)
            | ((imms & 0x3F) << 10)
            | ((rn & 0x1F) << 5)
            | (rd & 0x1F)
    }

    /// Move wide (MOVZ/MOVK/MOVN):
    /// `[sf(1)][opc(2)][100101][hw(2)][imm16(16)][Rd(5)]`
    fn encode_move_wide_fn(sf: u32, opc: u32, hw: u32, imm16: u32, rd: u32) -> u32 {
        (sf << 31)
            | (opc << 29)
            | (0b100101 << 23)
            | ((hw & 0x3) << 21)
            | ((imm16 & 0xFFFF) << 5)
            | (rd & 0x1F)
    }

    /// Bitfield (SBFM/BFM/UBFM):
    /// `[sf(1)][opc(2)][100110][N(1)][immr(6)][imms(6)][Rn(5)][Rd(5)]`
    fn encode_bitfield_fn(
        sf: u32,
        opc: u32,
        n: u32,
        immr: u32,
        imms: u32,
        rn: u32,
        rd: u32,
    ) -> u32 {
        (sf << 31)
            | (opc << 29)
            | (0b100110 << 23)
            | ((n & 1) << 22)
            | ((immr & 0x3F) << 16)
            | ((imms & 0x3F) << 10)
            | ((rn & 0x1F) << 5)
            | (rd & 0x1F)
    }

    /// PC-relative (ADR/ADRP):
    /// `[op(1)][immlo(2)][10000][immhi(19)][Rd(5)]`
    fn encode_pc_rel_fn(op: u32, immlo: u32, immhi: u32, rd: u32) -> u32 {
        (op << 31)
            | ((immlo & 0x3) << 29)
            | (0b10000 << 24)
            | ((immhi & 0x7FFFF) << 5)
            | (rd & 0x1F)
    }

    /// Unconditional branch immediate (B/BL):
    /// `[op(1)][00101][imm26(26)]`
    fn encode_branch_imm_fn(op: u32, imm26: u32) -> u32 {
        (op << 31) | (0b00101 << 26) | (imm26 & 0x3FFFFFF)
    }

    /// Conditional branch (B.cond):
    /// `[01010100][imm19(19)][0][cond(4)]`
    fn encode_cond_branch_fn(imm19: u32, cond: u32) -> u32 {
        (0b01010100u32 << 24) | ((imm19 & 0x7FFFF) << 5) | (cond & 0xF)
    }

    /// Compare and branch (CBZ/CBNZ):
    /// `[sf(1)][011010][op(1)][imm19(19)][Rt(5)]`
    fn encode_compare_branch_fn(sf: u32, op: u32, imm19: u32, rt: u32) -> u32 {
        (sf << 31) | (0b011010 << 25) | (op << 24) | ((imm19 & 0x7FFFF) << 5) | (rt & 0x1F)
    }

    /// Test and branch (TBZ/TBNZ):
    /// `[b5(1)][011011][op(1)][b40(5)][imm14(14)][Rt(5)]`
    fn encode_test_branch_fn(b5: u32, op: u32, b40: u32, imm14: u32, rt: u32) -> u32 {
        (b5 << 31)
            | (0b011011 << 25)
            | (op << 24)
            | ((b40 & 0x1F) << 19)
            | ((imm14 & 0x3FFF) << 5)
            | (rt & 0x1F)
    }

    /// Unconditional branch register (BR/BLR/RET):
    /// `[1101011][opc(4)][11111][000000][Rn(5)][00000]`
    fn encode_branch_reg_fn(&self, opc: u32, rn: u32) -> u32 {
        (0b1101011u32 << 25) | ((opc & 0xF) << 21) | (0b11111 << 16) | ((rn & 0x1F) << 5)
    }

    /// Logical shifted register:
    /// `[sf(1)][opc(2)][01010][shift(2)][N(1)][Rm(5)][imm6(6)][Rn(5)][Rd(5)]`
    fn encode_logical_shifted_fn(
        sf: u32,
        opc: u32,
        shift: u32,
        n: u32,
        rm: u32,
        imm6: u32,
        rn: u32,
        rd: u32,
    ) -> u32 {
        (sf << 31)
            | (opc << 29)
            | (0b01010 << 24)
            | ((shift & 0x3) << 22)
            | ((n & 1) << 21)
            | ((rm & 0x1F) << 16)
            | ((imm6 & 0x3F) << 10)
            | ((rn & 0x1F) << 5)
            | (rd & 0x1F)
    }

    /// Add/subtract shifted register:
    /// `[sf(1)][op(1)][S(1)][01011][shift(2)][0][Rm(5)][imm6(6)][Rn(5)][Rd(5)]`
    fn encode_add_sub_shifted_fn(
        sf: u32,
        op: u32,
        s: u32,
        shift: u32,
        rm: u32,
        imm6: u32,
        rn: u32,
        rd: u32,
    ) -> u32 {
        (sf << 31) | (op << 30) | (s << 29)
            | (0b01011 << 24)
            | ((shift & 0x3) << 22)
            | (0 << 21) // bit 21 = 0 for shifted
            | ((rm & 0x1F) << 16)
            | ((imm6 & 0x3F) << 10)
            | ((rn & 0x1F) << 5)
            | (rd & 0x1F)
    }

    /// Conditional select (CSEL/CSINC/CSINV/CSNEG):
    /// `[sf(1)][op(1)][S(1)][11010100][Rm(5)][cond(4)][op2(1)][0][Rn(5)][Rd(5)]`
    fn encode_cond_select_fn(
        sf: u32,
        op: u32,
        s: u32,
        rm: u32,
        cond: u32,
        op2: u32,
        rn: u32,
        rd: u32,
    ) -> u32 {
        (sf << 31)
            | (op << 30)
            | (s << 29)
            | (0b11010100u32 << 21)
            | ((rm & 0x1F) << 16)
            | ((cond & 0xF) << 12)
            | ((op2 & 1) << 10)
            | ((rn & 0x1F) << 5)
            | (rd & 0x1F)
    }

    /// Data-processing 2-source:
    /// `[sf(1)][0][S(1)][11010110][Rm(5)][opcode(6)][Rn(5)][Rd(5)]`
    fn encode_dp_2src_fn(&self, sf: u32, s: u32, rm: u32, opcode: u32, rn: u32, rd: u32) -> u32 {
        (sf << 31)
            | (s << 29)
            | (0b11010110u32 << 21)
            | ((rm & 0x1F) << 16)
            | ((opcode & 0x3F) << 10)
            | ((rn & 0x1F) << 5)
            | (rd & 0x1F)
    }

    /// Data-processing 3-source:
    /// `[sf(1)][op54(2)][11011][op31(3)][Rm(5)][o0(1)][Ra(5)][Rn(5)][Rd(5)]`
    fn encode_dp_3src(
        &self,
        sf: u32,
        op54: u32,
        op31: u32,
        rm: u32,
        o0: u32,
        ra: u32,
        rn: u32,
        rd: u32,
    ) -> u32 {
        (sf << 31)
            | (op54 << 29)
            | (0b11011 << 24)
            | ((op31 & 0x7) << 21)
            | ((rm & 0x1F) << 16)
            | ((o0 & 1) << 15)
            | ((ra & 0x1F) << 10)
            | ((rn & 0x1F) << 5)
            | (rd & 0x1F)
    }

    /// Data-processing 1-source:
    /// `[sf(1)][1][S(1)][11010110][opcode2(5)][opcode(6)][Rn(5)][Rd(5)]`
    fn encode_dp_1src_fn(sf: u32, s: u32, opcode2: u32, opcode: u32, rn: u32, rd: u32) -> u32 {
        (sf << 31)
            | (1 << 30)
            | (s << 29)
            | (0b11010110u32 << 21)
            | ((opcode2 & 0x1F) << 16)
            | ((opcode & 0x3F) << 10)
            | ((rn & 0x1F) << 5)
            | (rd & 0x1F)
    }

    /// Load/store unsigned immediate offset:
    /// `[size(2)][111][V(1)][01][opc(2)][imm12(12)][Rn(5)][Rt(5)]`
    fn encode_ldst_unsigned_imm_fn(
        size: u32,
        v: u32,
        opc: u32,
        imm12: u32,
        rn: u32,
        rt: u32,
    ) -> u32 {
        (size << 30)
            | (0b111 << 27)
            | (v << 26)
            | (0b01 << 24)
            | ((opc & 0x3) << 22)
            | ((imm12 & 0xFFF) << 10)
            | ((rn & 0x1F) << 5)
            | (rt & 0x1F)
    }

    /// Load/store register pair (signed offset):
    /// `[opc(2)][101][V(1)][010][L(1)][imm7(7)][Rt2(5)][Rn(5)][Rt(5)]`
    fn encode_ldst_pair_fn(opc: u32, v: u32, l: u32, imm7: u32, rt2: u32, rn: u32, rt: u32) -> u32 {
        (opc << 30)
            | (0b101 << 27)
            | (v << 26)
            | (0b010 << 23)
            | ((l & 1) << 22)
            | ((imm7 & 0x7F) << 15)
            | ((rt2 & 0x1F) << 10)
            | ((rn & 0x1F) << 5)
            | (rt & 0x1F)
    }

    /// Load/store register pair (pre-indexed):
    /// `[opc(2)][101][V(1)][011][L(1)][imm7(7)][Rt2(5)][Rn(5)][Rt(5)]`
    fn encode_ldst_pair_pre_fn(
        opc: u32,
        v: u32,
        l: u32,
        imm7: u32,
        rt2: u32,
        rn: u32,
        rt: u32,
    ) -> u32 {
        (opc << 30)
            | (0b101 << 27)
            | (v << 26)
            | (0b011 << 23)
            | ((l & 1) << 22)
            | ((imm7 & 0x7F) << 15)
            | ((rt2 & 0x1F) << 10)
            | ((rn & 0x1F) << 5)
            | (rt & 0x1F)
    }

    /// FP two-operand data processing:
    /// `[M(1)][0][S(1)][11110][ftype(2)][1][Rm(5)][opcode(4)][10][Rn(5)][Rd(5)]`
    fn encode_fp_dp2_fn(ftype: u32, rm: u32, opcode: u32, rn: u32, rd: u32) -> u32 {
        (0b0001_1110u32 << 24)
            | ((ftype & 0x3) << 22)
            | (1 << 21)
            | ((rm & 0x1F) << 16)
            | ((opcode & 0xF) << 12)
            | (0b10 << 10)
            | ((rn & 0x1F) << 5)
            | (rd & 0x1F)
    }

    /// FP one-source data processing:
    /// `[M(1)][0][S(1)][11110][ftype(2)][1][0000][opcode(6)][10000][Rn(5)][Rd(5)]`
    fn encode_fp_dp1_fn(ftype: u32, opcode: u32, rn: u32, rd: u32) -> u32 {
        (0b0001_1110u32 << 24)
            | ((ftype & 0x3) << 22)
            | (1 << 21)
            | ((opcode & 0x3F) << 15)
            | (0b10000 << 10)
            | ((rn & 0x1F) << 5)
            | (rd & 0x1F)
    }

    /// FP compare:
    /// `[M(1)][0][S(1)][11110][ftype(2)][1][Rm(5)][001000][Rn(5)][opcode2(5)]`
    fn encode_fp_cmp_fn(ftype: u32, rm: u32, rn: u32, opcode2: u32) -> u32 {
        (0b0001_1110u32 << 24)
            | ((ftype & 0x3) << 22)
            | (1 << 21)
            | ((rm & 0x1F) << 16)
            | (0b001000 << 10)
            | ((rn & 0x1F) << 5)
            | (opcode2 & 0x1F)
    }

    /// FP integer conversion:
    /// `[sf(1)][0][0][11110][ftype(2)][1][rmode(2)][opcode(3)][000000][Rn(5)][Rd(5)]`
    fn encode_fp_int_conv_fn(
        sf: u32,
        ftype: u32,
        rmode: u32,
        opcode: u32,
        rn: u32,
        rd: u32,
    ) -> u32 {
        (sf << 31)
            | (0b0001_1110u32 << 24)
            | ((ftype & 0x3) << 22)
            | (1 << 21)
            | ((rmode & 0x3) << 19)
            | ((opcode & 0x7) << 16)
            | ((rn & 0x1F) << 5)
            | (rd & 0x1F)
    }

    /// FP precision conversion:
    /// `[0][0][0][11110][ftype(2)][1][0001][opc(2)][10000][Rn(5)][Rd(5)]`
    fn encode_fp_cvt_fn(ftype: u32, opc: u32, rn: u32, rd: u32) -> u32 {
        (0b0001_1110u32 << 24)
            | ((ftype & 0x3) << 22)
            | (1 << 21)
            | (0b0001 << 17)
            | ((opc & 0x3) << 15)
            | (0b10000 << 10)
            | ((rn & 0x1F) << 5)
            | (rd & 0x1F)
    }

    // -----------------------------------------------------------------------
    // High-level instruction encoding methods
    // -----------------------------------------------------------------------

    /// Encode ADD / ADDS / SUB / SUBS — dispatches between immediate
    /// and shifted-register forms based on the operand types.
    fn encode_add_sub(&mut self, opcode: Aarch64Opcode, operands: &[MachineOperand]) {
        let (op, s) = match opcode {
            Aarch64Opcode::ADD => (0, 0),
            Aarch64Opcode::ADDS => (0, 1),
            Aarch64Opcode::SUB => (1, 0),
            Aarch64Opcode::SUBS => (1, 1),
            _ => (0, 0),
        };
        let sf = self.sf_from_operands(operands);
        let rd = self.extract_reg(operands, 0);
        let rn = self.extract_reg(operands, 1);

        // Check if operand 2 is an immediate or register.
        match operands.get(2) {
            Some(MachineOperand::Immediate(imm)) => {
                let imm_val = *imm as u64;
                // If the immediate fits in 12 bits, use the immediate form.
                if imm_val <= 0xFFF {
                    let enc = Self::encode_add_sub_imm_fn(sf, op, s, 0, imm_val as u32, rn, rd);
                    self.emit_u32(enc);
                } else if (imm_val & 0xFFF) == 0 && (imm_val >> 12) <= 0xFFF {
                    // Shifted by 12: sh=1
                    let enc =
                        Self::encode_add_sub_imm_fn(sf, op, s, 1, (imm_val >> 12) as u32, rn, rd);
                    self.emit_u32(enc);
                } else {
                    // Fall back: use immediate form with truncated value.
                    let enc =
                        Self::encode_add_sub_imm_fn(sf, op, s, 0, (imm_val & 0xFFF) as u32, rn, rd);
                    self.emit_u32(enc);
                }
            }
            Some(MachineOperand::Register(_)) => {
                let rm = self.extract_reg(operands, 2);
                // Optional shift operand at index 3 (shift type) and 4 (amount).
                let (shift_enc, shift_amt) = self.extract_shift(operands, 3);
                let enc =
                    Self::encode_add_sub_shifted_fn(sf, op, s, shift_enc, rm, shift_amt, rn, rd);
                self.emit_u32(enc);
            }
            _ => {
                // Two-register form: Rd, Rn — treat as ADD Rd, Rn, #0
                let enc = Self::encode_add_sub_imm_fn(sf, op, s, 0, 0, rn, rd);
                self.emit_u32(enc);
            }
        }
    }

    /// Extract shift type encoding and amount from operands starting at `idx`.
    /// Returns (shift_encoding, amount).  Default is (0, 0) = LSL #0 = no shift.
    fn extract_shift(&self, operands: &[MachineOperand], idx: usize) -> (u32, u32) {
        match operands.get(idx) {
            Some(MachineOperand::Immediate(shift_val)) => {
                let shift_type = (*shift_val >> 8) as u32 & 0x3;
                let amount = (*shift_val & 0xFF) as u32;
                (shift_type, amount & 0x3F)
            }
            _ => (0, 0),
        }
    }

    /// Encode AND / ANDS / ORR / EOR — dispatches between bitmask immediate
    /// and shifted-register forms.
    fn encode_logical(&mut self, opcode: Aarch64Opcode, operands: &[MachineOperand]) {
        let opc = match opcode {
            Aarch64Opcode::AND => 0b00,
            Aarch64Opcode::ORR => 0b01,
            Aarch64Opcode::EOR => 0b10,
            Aarch64Opcode::ANDS => 0b11,
            _ => 0b00,
        };
        let sf = self.sf_from_operands(operands);
        let rd = self.extract_reg(operands, 0);
        let rn = self.extract_reg(operands, 1);
        let reg_size = if sf == 1 { 64 } else { 32 };

        match operands.get(2) {
            Some(MachineOperand::Immediate(imm)) => {
                let val = *imm as u64;
                if let Some((n, immr, imms)) = encode_bitmask_immediate(val, reg_size) {
                    let enc = Self::encode_logical_imm_fn(sf, opc, n, immr, imms, rn, rd);
                    self.emit_u32(enc);
                } else {
                    // Cannot encode as bitmask immediate — fall through to
                    // a shifted register encoding with immediate = 0.
                    // In practice the isel should not produce this case.
                    let enc = Self::encode_logical_shifted_fn(sf, opc, 0, 0, 31, 0, rn, rd);
                    self.emit_u32(enc);
                }
            }
            Some(MachineOperand::Register(_)) => {
                let rm = self.extract_reg(operands, 2);
                let (shift_enc, shift_amt) = self.extract_shift(operands, 3);
                let enc =
                    Self::encode_logical_shifted_fn(sf, opc, shift_enc, 0, rm, shift_amt, rn, rd);
                self.emit_u32(enc);
            }
            _ => {
                let enc = Self::encode_logical_shifted_fn(sf, opc, 0, 0, 31, 0, rn, rd);
                self.emit_u32(enc);
            }
        }
    }

    /// Encode LSL / LSR / ASR — dispatches between immediate (bitfield)
    /// and register (2-source) forms.
    fn encode_shift(&mut self, opcode: Aarch64Opcode, operands: &[MachineOperand]) {
        let sf = self.sf_from_operands(operands);
        let rd = self.extract_reg(operands, 0);
        let rn = self.extract_reg(operands, 1);
        let n = sf; // N must equal sf for 64-bit bitfield operations.
        let reg_bits: u32 = if sf == 1 { 64 } else { 32 };

        match operands.get(2) {
            Some(MachineOperand::Immediate(imm)) => {
                let amount = (*imm as u32) & (reg_bits - 1);
                let enc = match opcode {
                    Aarch64Opcode::LSL => {
                        // LSL #n = UBFM Rd, Rn, #(-n mod size), #(size-1-n)
                        let immr = (reg_bits.wrapping_sub(amount)) & (reg_bits - 1);
                        let imms = reg_bits - 1 - amount;
                        Self::encode_bitfield_fn(sf, 0b10, n, immr, imms, rn, rd)
                    }
                    Aarch64Opcode::LSR => {
                        // LSR #n = UBFM Rd, Rn, #n, #(size-1)
                        Self::encode_bitfield_fn(sf, 0b10, n, amount, reg_bits - 1, rn, rd)
                    }
                    Aarch64Opcode::ASR => {
                        // ASR #n = SBFM Rd, Rn, #n, #(size-1)
                        Self::encode_bitfield_fn(sf, 0b00, n, amount, reg_bits - 1, rn, rd)
                    }
                    _ => 0xD503201F, // NOP fallback
                };
                self.emit_u32(enc);
            }
            Some(MachineOperand::Register(_)) => {
                let rm = self.extract_reg(operands, 2);
                let dp2_opcode = match opcode {
                    Aarch64Opcode::LSL => 0b001000, // LSLV
                    Aarch64Opcode::LSR => 0b001001, // LSRV
                    Aarch64Opcode::ASR => 0b001010, // ASRV
                    _ => 0b001000,
                };
                let enc = self.encode_dp_2src_fn(sf, 0, rm, dp2_opcode, rn, rd);
                self.emit_u32(enc);
            }
            _ => {
                self.emit_u32(0xD503201F); // NOP
            }
        }
    }

    /// Encode MOV (register-to-register).
    /// MOV Xd, Xm = ORR Xd, XZR, Xm
    fn encode_mov(&mut self, operands: &[MachineOperand]) {
        let sf = self.sf_from_operands(operands);
        let rd = self.extract_reg(operands, 0);

        match operands.get(1) {
            Some(MachineOperand::Register(_)) => {
                let rm = self.extract_reg(operands, 1);
                // MOV Xd, Xm = ORR Xd, XZR, Xm
                let enc = Self::encode_logical_shifted_fn(sf, 0b01, 0, 0, rm, 0, 31, rd);
                self.emit_u32(enc);
            }
            Some(MachineOperand::Immediate(imm)) => {
                // Small immediate MOV — use MOVZ.
                let val = *imm as u64;
                let enc = Self::encode_move_wide_fn(sf, 0b10, 0, (val & 0xFFFF) as u32, rd);
                self.emit_u32(enc);
            }
            _ => {
                // MOV Xd, XZR
                let enc = Self::encode_logical_shifted_fn(sf, 0b01, 0, 0, 31, 0, 31, rd);
                self.emit_u32(enc);
            }
        }
    }

    /// Encode MOVZ / MOVK / MOVN — move wide immediate.
    fn encode_movzk(&mut self, opcode: Aarch64Opcode, operands: &[MachineOperand]) {
        let sf = self.sf_from_operands(operands);
        let rd = self.extract_reg(operands, 0);
        let imm16 = self.extract_imm(operands, 1) as u32;
        // hw (halfword position) is in operand 2 if present.
        let hw = if operands.len() > 2 {
            self.extract_imm(operands, 2) as u32 & 0x3
        } else {
            0
        };
        let opc = match opcode {
            Aarch64Opcode::MOVN => 0b00,
            Aarch64Opcode::MOVZ => 0b10,
            Aarch64Opcode::MOVK => 0b11,
            _ => 0b10,
        };
        let enc = Self::encode_move_wide_fn(sf, opc, hw, imm16 & 0xFFFF, rd);
        self.emit_u32(enc);
    }

    /// Encode CSEL / CSINC / CSINV / CSNEG.
    /// Operands: Rd, Rn, Rm, condition (as Immediate).
    fn encode_cond_sel(&mut self, opcode: Aarch64Opcode, operands: &[MachineOperand]) {
        let sf = self.sf_from_operands(operands);
        let rd = self.extract_reg(operands, 0);
        let rn = self.extract_reg(operands, 1);
        let rm = self.extract_reg(operands, 2);
        let cond = self.extract_imm(operands, 3) as u32 & 0xF;

        let (op, op2) = match opcode {
            Aarch64Opcode::CSEL => (0, 0),
            Aarch64Opcode::CSINC => (0, 1),
            Aarch64Opcode::CSINV => (1, 0),
            Aarch64Opcode::CSNEG => (1, 1),
            _ => (0, 0),
        };
        let enc = Self::encode_cond_select_fn(sf, op, 0, rm, cond, op2, rn, rd);
        self.emit_u32(enc);
    }

    // -----------------------------------------------------------------------
    // Load / Store helpers
    // -----------------------------------------------------------------------

    /// Encode 64-bit LDR/STR (size=11, opc=01/00).
    fn encode_ldst_64(&mut self, operands: &[MachineOperand], is_load: bool) {
        let rt = self.extract_reg(operands, 0);
        let opc: u32 = if is_load { 0b01 } else { 0b00 };

        match operands.get(1) {
            Some(MachineOperand::Memory { base, offset }) => {
                let rn = Self::reg_num(base);
                let off = *offset;
                // Unsigned offset encoding: offset must be non-negative and 8-byte aligned.
                if off >= 0 && (off as u32 % 8 == 0) && ((off as u32 / 8) <= 0xFFF) {
                    let imm12 = off as u32 / 8;
                    let enc = Self::encode_ldst_unsigned_imm_fn(0b11, 0, opc, imm12, rn, rt);
                    self.emit_u32(enc);
                } else {
                    // Unscaled immediate form: LDUR/STUR
                    self.encode_ldst_unscaled(0b11, opc, off, rn, rt);
                }
            }
            Some(MachineOperand::Symbol(sym)) => {
                // Symbol reference — emit relocation.
                self.relocations.push(Relocation {
                    offset: self.offset as u64,
                    symbol: sym.clone(),
                    reloc_type: if is_load {
                        RelocationType::Aarch64_ABS64
                    } else {
                        RelocationType::Aarch64_ABS64
                    },
                    addend: 0,
                    section_index: 0,
                });
                let enc = Self::encode_ldst_unsigned_imm_fn(0b11, 0, opc, 0, 31, rt);
                self.emit_u32(enc);
            }
            _ => {
                // Default: [SP, #0]
                let enc = Self::encode_ldst_unsigned_imm_fn(0b11, 0, opc, 0, 31, rt);
                self.emit_u32(enc);
            }
        }
    }

    /// Encode 32-bit LDR/STR (size=10, opc=01/00).
    fn encode_ldst_32(&mut self, operands: &[MachineOperand], is_load: bool) {
        let rt = self.extract_reg(operands, 0);
        let opc: u32 = if is_load { 0b01 } else { 0b00 };
        let (rn, off) = self.extract_memory(operands, 1);
        if off >= 0 && (off as u32 % 4 == 0) && ((off as u32 / 4) <= 0xFFF) {
            let imm12 = off as u32 / 4;
            let enc = Self::encode_ldst_unsigned_imm_fn(0b10, 0, opc, imm12, rn, rt);
            self.emit_u32(enc);
        } else {
            self.encode_ldst_unscaled(0b10, opc, off, rn, rt);
        }
    }

    /// Encode 16-bit LDR/STR (size=01, opc=01/00).
    fn encode_ldst_16(&mut self, operands: &[MachineOperand], is_load: bool) {
        let rt = self.extract_reg(operands, 0);
        let opc: u32 = if is_load { 0b01 } else { 0b00 };
        let (rn, off) = self.extract_memory(operands, 1);
        if off >= 0 && (off as u32 % 2 == 0) && ((off as u32 / 2) <= 0xFFF) {
            let imm12 = off as u32 / 2;
            let enc = Self::encode_ldst_unsigned_imm_fn(0b01, 0, opc, imm12, rn, rt);
            self.emit_u32(enc);
        } else {
            self.encode_ldst_unscaled(0b01, opc, off, rn, rt);
        }
    }

    /// Encode 8-bit LDR/STR (size=00, opc=01/00).
    fn encode_ldst_8(&mut self, operands: &[MachineOperand], is_load: bool) {
        let rt = self.extract_reg(operands, 0);
        let opc: u32 = if is_load { 0b01 } else { 0b00 };
        let (rn, off) = self.extract_memory(operands, 1);
        if off >= 0 && (off as u32) <= 0xFFF {
            let imm12 = off as u32;
            let enc = Self::encode_ldst_unsigned_imm_fn(0b00, 0, opc, imm12, rn, rt);
            self.emit_u32(enc);
        } else {
            self.encode_ldst_unscaled(0b00, opc, off, rn, rt);
        }
    }

    /// Encode sign-extending loads (LDRSB, LDRSH, LDRSW).
    fn encode_ldst_sign_ext(&mut self, operands: &[MachineOperand], size: u32, opc: u32) {
        let rt = self.extract_reg(operands, 0);
        let (rn, off) = self.extract_memory(operands, 1);
        let scale = 1u32 << size;
        if off >= 0 && (off as u32 % scale == 0) && ((off as u32 / scale) <= 0xFFF) {
            let imm12 = off as u32 / scale;
            let enc = Self::encode_ldst_unsigned_imm_fn(size, 0, opc, imm12, rn, rt);
            self.emit_u32(enc);
        } else {
            self.encode_ldst_unscaled(size, opc, off, rn, rt);
        }
    }

    /// Emit an unscaled load/store (LDUR/STUR) for offsets that do not fit
    /// the unsigned-immediate encoding.
    /// `[size(2)][111][V(1)][00][opc(2)][0][imm9(9)][00][Rn(5)][Rt(5)]`
    fn encode_ldst_unscaled(&mut self, size: u32, opc: u32, offset: i32, rn: u32, rt: u32) {
        let imm9 = (offset as u32) & 0x1FF;
        let enc = (size << 30)
            | (0b111 << 27)
            | (0 << 26)
            | (0b00 << 24)
            | ((opc & 0x3) << 22)
            | (0 << 21)
            | (imm9 << 12)
            | (0b00 << 10)
            | ((rn & 0x1F) << 5)
            | (rt & 0x1F);
        self.emit_u32(enc);
    }

    /// Encode LDP / STP (integer register pair).
    /// Operands: Rt, Rt2, Memory{base, offset}.
    fn encode_ldp_stp(&mut self, operands: &[MachineOperand], is_load: bool) {
        let rt = self.extract_reg(operands, 0);
        let rt2 = self.extract_reg(operands, 1);
        let (rn, off) = self.extract_memory(operands, 2);
        let l: u32 = if is_load { 1 } else { 0 };

        // Signed offset scaled by 8 (64-bit registers).
        let imm7 = ((off / 8) as u32) & 0x7F;

        // Check if this is a pre-indexed form (offset applied before access).
        // Convention: if there is an Immediate(1) at index 3, it's pre-indexed.
        let pre_index = matches!(operands.get(3), Some(MachineOperand::Immediate(1)));

        let enc = if pre_index {
            Self::encode_ldst_pair_pre_fn(0b10, 0, l, imm7, rt2, rn, rt)
        } else {
            Self::encode_ldst_pair_fn(0b10, 0, l, imm7, rt2, rn, rt)
        };
        self.emit_u32(enc);
    }

    // -----------------------------------------------------------------------
    // PC-relative addressing
    // -----------------------------------------------------------------------

    /// Encode ADRP Xd, symbol — emits relocation.
    fn encode_adrp(&mut self, operands: &[MachineOperand]) {
        let rd = self.extract_reg(operands, 0);

        if let Some(sym) = self.extract_symbol(operands, 1) {
            self.relocations.push(Relocation {
                offset: self.offset as u64,
                symbol: sym.to_string(),
                reloc_type: RelocationType::Aarch64_ADR_PREL_PG_HI21,
                addend: 0,
                section_index: 0,
            });
        }

        // Emit with zero offset — the linker will patch.
        let enc = Self::encode_pc_rel_fn(1, 0, 0, rd);
        self.emit_u32(enc);
    }

    /// Encode ADR Xd, label/symbol.
    fn encode_adr(&mut self, operands: &[MachineOperand]) {
        let rd = self.extract_reg(operands, 0);

        if let Some(sym) = self.extract_symbol(operands, 1) {
            self.relocations.push(Relocation {
                offset: self.offset as u64,
                symbol: sym.to_string(),
                reloc_type: RelocationType::Aarch64_ADD_ABS_LO12_NC,
                addend: 0,
                section_index: 0,
            });
        }

        let enc = Self::encode_pc_rel_fn(0, 0, 0, rd);
        self.emit_u32(enc);
    }

    // -----------------------------------------------------------------------
    // Branch encoding
    // -----------------------------------------------------------------------

    /// Encode B (unconditional branch).
    fn encode_branch_b(&mut self, operands: &[MachineOperand]) {
        // Operand can be a Label or a Symbol.
        match operands.first() {
            Some(MachineOperand::Label(lbl)) => {
                let lbl = *lbl;
                if let Some(&target_off) = self.label_offsets.get(&lbl) {
                    // Backward branch — offset already known.
                    let delta = (target_off as i64 - self.offset as i64) >> 2;
                    let enc = Self::encode_branch_imm_fn(0, (delta as u32) & 0x3FFFFFF);
                    self.emit_u32(enc);
                } else {
                    // Forward branch — record fixup, emit placeholder.
                    self.fixups.push(BranchFixup {
                        label: lbl,
                        code_offset: self.offset,
                        fixup_type: AArch64FixupType::Branch26,
                    });
                    let enc = Self::encode_branch_imm_fn(0, 0);
                    self.emit_u32(enc);
                }
            }
            Some(MachineOperand::Symbol(sym)) => {
                self.relocations.push(Relocation {
                    offset: self.offset as u64,
                    symbol: sym.clone(),
                    reloc_type: RelocationType::Aarch64_JUMP26,
                    addend: 0,
                    section_index: 0,
                });
                let enc = Self::encode_branch_imm_fn(0, 0);
                self.emit_u32(enc);
            }
            Some(MachineOperand::Immediate(off)) => {
                let delta = (*off >> 2) as u32;
                let enc = Self::encode_branch_imm_fn(0, delta & 0x3FFFFFF);
                self.emit_u32(enc);
            }
            _ => {
                let enc = Self::encode_branch_imm_fn(0, 0);
                self.emit_u32(enc);
            }
        }
    }

    /// Encode BL (branch with link — function call).
    fn encode_branch_bl(&mut self, operands: &[MachineOperand]) {
        match operands.first() {
            Some(MachineOperand::Symbol(sym)) => {
                self.relocations.push(Relocation {
                    offset: self.offset as u64,
                    symbol: sym.clone(),
                    reloc_type: RelocationType::Aarch64_CALL26,
                    addend: 0,
                    section_index: 0,
                });
                let enc = Self::encode_branch_imm_fn(1, 0);
                self.emit_u32(enc);
            }
            Some(MachineOperand::Label(lbl)) => {
                let lbl = *lbl;
                if let Some(&target_off) = self.label_offsets.get(&lbl) {
                    let delta = (target_off as i64 - self.offset as i64) >> 2;
                    let enc = Self::encode_branch_imm_fn(1, (delta as u32) & 0x3FFFFFF);
                    self.emit_u32(enc);
                } else {
                    self.fixups.push(BranchFixup {
                        label: lbl,
                        code_offset: self.offset,
                        fixup_type: AArch64FixupType::Branch26,
                    });
                    let enc = Self::encode_branch_imm_fn(1, 0);
                    self.emit_u32(enc);
                }
            }
            Some(MachineOperand::Immediate(off)) => {
                let delta = (*off >> 2) as u32;
                let enc = Self::encode_branch_imm_fn(1, delta & 0x3FFFFFF);
                self.emit_u32(enc);
            }
            _ => {
                let enc = Self::encode_branch_imm_fn(1, 0);
                self.emit_u32(enc);
            }
        }
    }

    /// Encode B.cond — conditional branch.
    /// Operands: condition (Immediate), label/symbol.
    fn encode_b_cond(&mut self, operands: &[MachineOperand]) {
        let cond = self.extract_imm(operands, 0) as u32 & 0xF;

        match operands.get(1) {
            Some(MachineOperand::Label(lbl)) => {
                let lbl = *lbl;
                if let Some(&target_off) = self.label_offsets.get(&lbl) {
                    let delta = (target_off as i64 - self.offset as i64) >> 2;
                    let enc = Self::encode_cond_branch_fn((delta as u32) & 0x7FFFF, cond);
                    self.emit_u32(enc);
                } else {
                    self.fixups.push(BranchFixup {
                        label: lbl,
                        code_offset: self.offset,
                        fixup_type: AArch64FixupType::CondBranch19,
                    });
                    let enc = Self::encode_cond_branch_fn(0, cond);
                    self.emit_u32(enc);
                }
            }
            Some(MachineOperand::Immediate(off)) => {
                let delta = (*off >> 2) as u32;
                let enc = Self::encode_cond_branch_fn(delta & 0x7FFFF, cond);
                self.emit_u32(enc);
            }
            _ => {
                let enc = Self::encode_cond_branch_fn(0, cond);
                self.emit_u32(enc);
            }
        }
    }

    /// Encode CBZ / CBNZ.
    /// Operands: Rt, label (Label or Immediate).
    fn encode_cbz_cbnz(&mut self, operands: &[MachineOperand], is_nz: bool) {
        let sf = self.sf_from_operands(operands);
        let rt = self.extract_reg(operands, 0);
        let op: u32 = if is_nz { 1 } else { 0 };

        match operands.get(1) {
            Some(MachineOperand::Label(lbl)) => {
                let lbl = *lbl;
                if let Some(&target_off) = self.label_offsets.get(&lbl) {
                    let delta = (target_off as i64 - self.offset as i64) >> 2;
                    let enc = Self::encode_compare_branch_fn(sf, op, (delta as u32) & 0x7FFFF, rt);
                    self.emit_u32(enc);
                } else {
                    self.fixups.push(BranchFixup {
                        label: lbl,
                        code_offset: self.offset,
                        fixup_type: AArch64FixupType::CondBranch19,
                    });
                    let enc = Self::encode_compare_branch_fn(sf, op, 0, rt);
                    self.emit_u32(enc);
                }
            }
            Some(MachineOperand::Immediate(off)) => {
                let delta = (*off >> 2) as u32;
                let enc = Self::encode_compare_branch_fn(sf, op, delta & 0x7FFFF, rt);
                self.emit_u32(enc);
            }
            _ => {
                let enc = Self::encode_compare_branch_fn(sf, op, 0, rt);
                self.emit_u32(enc);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Floating-point encoding helpers
    // -----------------------------------------------------------------------

    /// Encode a 2-operand FP arithmetic instruction (FADD, FSUB, FMUL, FDIV, etc.).
    fn encode_fp_arith2(&mut self, operands: &[MachineOperand], ftype: u32, fp_opcode: u32) {
        let rd = self.extract_reg(operands, 0);
        let rn = self.extract_reg(operands, 1);
        let rm = self.extract_reg(operands, 2);
        let enc = Self::encode_fp_dp2_fn(ftype, rm, fp_opcode, rn, rd);
        self.emit_u32(enc);
    }

    /// Encode a 1-source FP instruction (FSQRT, FABS, FNEG).
    fn encode_fp_unary(&mut self, operands: &[MachineOperand], ftype: u32, opcode: u32) {
        let rd = self.extract_reg(operands, 0);
        let rn = self.extract_reg(operands, 1);
        let enc = Self::encode_fp_dp1_fn(ftype, opcode, rn, rd);
        self.emit_u32(enc);
    }

    /// Encode FMOV (register move within FP or GPR↔FP).
    fn encode_fmov(&mut self, operands: &[MachineOperand]) {
        let rd = self.extract_reg(operands, 0);
        let rn = self.extract_reg(operands, 1);
        let rd_is_fp =
            matches!(operands.get(0), Some(MachineOperand::Register(r)) if Self::is_fp_phys(r));
        let rn_is_fp =
            matches!(operands.get(1), Some(MachineOperand::Register(r)) if Self::is_fp_phys(r));

        if rd_is_fp && rn_is_fp {
            // FMOV within FP regs — use FP 1-source encoding.
            // ftype = 01 for double, opcode = 000000
            let enc = Self::encode_fp_dp1_fn(0b01, 0b000000, rn, rd);
            self.emit_u32(enc);
        } else if rd_is_fp && !rn_is_fp {
            // GPR → FP: FMOV Dd, Xn
            // sf=1, ftype=01, rmode=00, opcode=111
            let enc = Self::encode_fp_int_conv_fn(1, 0b01, 0b00, 0b111, rn, rd);
            self.emit_u32(enc);
        } else if !rd_is_fp && rn_is_fp {
            // FP → GPR: FMOV Xd, Dn
            // sf=1, ftype=01, rmode=00, opcode=110
            let enc = Self::encode_fp_int_conv_fn(1, 0b01, 0b00, 0b110, rn, rd);
            self.emit_u32(enc);
        } else {
            // Both GPR — encode as a regular MOV.
            let enc = Self::encode_logical_shifted_fn(1, 0b01, 0, 0, rn, 0, 31, rd);
            self.emit_u32(enc);
        }
    }

    /// Encode FCMP.
    fn encode_fp_compare(&mut self, operands: &[MachineOperand], ftype: u32) {
        let rn = self.extract_reg(operands, 0);
        let rm = self.extract_reg(operands, 1);
        // opcode2 = 0b00000 for normal compare.
        let enc = Self::encode_fp_cmp_fn(ftype, rm, rn, 0b00000);
        self.emit_u32(enc);
    }

    /// Encode FCSEL.
    fn encode_fcsel(&mut self, operands: &[MachineOperand], ftype: u32) {
        let rd = self.extract_reg(operands, 0);
        let rn = self.extract_reg(operands, 1);
        let rm = self.extract_reg(operands, 2);
        let cond = self.extract_imm(operands, 3) as u32 & 0xF;
        // FCSEL: [M(1)][0][S(1)][11110][ftype(2)][1][Rm(5)][cond(4)][11][Rn(5)][Rd(5)]
        let enc = (0b0001_1110u32 << 24)
            | ((ftype & 0x3) << 22)
            | (1 << 21)
            | ((rm & 0x1F) << 16)
            | ((cond & 0xF) << 12)
            | (0b11 << 10)
            | ((rn & 0x1F) << 5)
            | (rd & 0x1F);
        self.emit_u32(enc);
    }

    /// Encode FP ↔ integer conversion (SCVTF, UCVTF, FCVTZS, FCVTZU).
    fn encode_fp_int_conv(
        &mut self,
        operands: &[MachineOperand],
        ftype: u32,
        rmode: u32,
        opcode: u32,
    ) {
        let rd = self.extract_reg(operands, 0);
        let rn = self.extract_reg(operands, 1);
        let enc = Self::encode_fp_int_conv_fn(1, ftype, rmode, opcode, rn, rd);
        self.emit_u32(enc);
    }

    /// Encode FP precision conversion (FCVT single↔double).
    fn encode_fp_cvt_precision(&mut self, operands: &[MachineOperand], ftype: u32, opc: u32) {
        let rd = self.extract_reg(operands, 0);
        let rn = self.extract_reg(operands, 1);
        let enc = Self::encode_fp_cvt_fn(ftype, opc, rn, rd);
        self.emit_u32(enc);
    }

    /// Encode FP load/store (single and double).
    fn encode_fp_ldst(&mut self, operands: &[MachineOperand], size: u32, opc: u32) {
        let rt = self.extract_reg(operands, 0);
        let (rn, off) = self.extract_memory(operands, 1);
        let scale = 1u32 << (size & 0x3); // 4 for single, 8 for double
        if off >= 0 && (off as u32 % scale == 0) && ((off as u32 / scale) <= 0xFFF) {
            let imm12 = off as u32 / scale;
            let enc = Self::encode_ldst_unsigned_imm_fn(size, 1, opc, imm12, rn, rt);
            self.emit_u32(enc);
        } else {
            // Use unscaled form.
            let imm9 = (off as u32) & 0x1FF;
            let enc = (size << 30)
                | (0b111 << 27)
                | (1 << 26)
                | (0b00 << 24)
                | ((opc & 0x3) << 22)
                | (imm9 << 12)
                | (0b00 << 10)
                | ((rn & 0x1F) << 5)
                | (rt & 0x1F);
            self.emit_u32(enc);
        }
    }

    /// Encode FP load/store pair (double precision).
    fn encode_fp_ldp_stp(&mut self, operands: &[MachineOperand], is_load: bool) {
        let rt = self.extract_reg(operands, 0);
        let rt2 = self.extract_reg(operands, 1);
        let (rn, off) = self.extract_memory(operands, 2);
        let l: u32 = if is_load { 1 } else { 0 };
        // FP pair, opc=01 for 64-bit (D), V=1
        let imm7 = ((off / 8) as u32) & 0x7F;
        let enc = Self::encode_ldst_pair_fn(0b01, 1, l, imm7, rt2, rn, rt);
        self.emit_u32(enc);
    }

    // -----------------------------------------------------------------------
    // Remaining opcode handler (extend, bit-manipulation, etc.)
    // -----------------------------------------------------------------------

    /// Handle opcodes not covered by the main match arms.
    fn encode_remaining(&mut self, opcode: Aarch64Opcode, operands: &[MachineOperand]) {
        match opcode {
            // Sign / zero extend
            Aarch64Opcode::SXTB => {
                let sf = self.sf_from_operands(operands);
                let n = sf;
                let rd = self.extract_reg(operands, 0);
                let rn = self.extract_reg(operands, 1);
                let enc = Self::encode_bitfield_fn(sf, 0b00, n, 0, 7, rn, rd);
                self.emit_u32(enc);
            }
            Aarch64Opcode::SXTH => {
                let sf = self.sf_from_operands(operands);
                let n = sf;
                let rd = self.extract_reg(operands, 0);
                let rn = self.extract_reg(operands, 1);
                let enc = Self::encode_bitfield_fn(sf, 0b00, n, 0, 15, rn, rd);
                self.emit_u32(enc);
            }
            Aarch64Opcode::SXTW => {
                let rd = self.extract_reg(operands, 0);
                let rn = self.extract_reg(operands, 1);
                let enc = Self::encode_bitfield_fn(1, 0b00, 1, 0, 31, rn, rd);
                self.emit_u32(enc);
            }
            Aarch64Opcode::UXTB => {
                let rd = self.extract_reg(operands, 0);
                let rn = self.extract_reg(operands, 1);
                // UXTB = UBFM Wd, Wn, #0, #7  (sf=0)
                let enc = Self::encode_bitfield_fn(0, 0b10, 0, 0, 7, rn, rd);
                self.emit_u32(enc);
            }
            Aarch64Opcode::UXTH => {
                let rd = self.extract_reg(operands, 0);
                let rn = self.extract_reg(operands, 1);
                // UXTH = UBFM Wd, Wn, #0, #15  (sf=0)
                let enc = Self::encode_bitfield_fn(0, 0b10, 0, 0, 15, rn, rd);
                self.emit_u32(enc);
            }

            // Bit manipulation
            Aarch64Opcode::CLZ => {
                let sf = self.sf_from_operands(operands);
                let rd = self.extract_reg(operands, 0);
                let rn = self.extract_reg(operands, 1);
                let enc = Self::encode_dp_1src_fn(sf, 0, 0b00000, 0b000100, rn, rd);
                self.emit_u32(enc);
            }
            Aarch64Opcode::RBIT => {
                let sf = self.sf_from_operands(operands);
                let rd = self.extract_reg(operands, 0);
                let rn = self.extract_reg(operands, 1);
                let enc = Self::encode_dp_1src_fn(sf, 0, 0b00000, 0b000000, rn, rd);
                self.emit_u32(enc);
            }
            Aarch64Opcode::REV => {
                let sf = self.sf_from_operands(operands);
                let rd = self.extract_reg(operands, 0);
                let rn = self.extract_reg(operands, 1);
                // REV Xd, Xn: opcode=000011 for 64-bit, 000010 for 32-bit
                let opc = if sf == 1 { 0b000011 } else { 0b000010 };
                let enc = Self::encode_dp_1src_fn(sf, 0, 0b00000, opc, rn, rd);
                self.emit_u32(enc);
            }
            Aarch64Opcode::REV16 => {
                let sf = self.sf_from_operands(operands);
                let rd = self.extract_reg(operands, 0);
                let rn = self.extract_reg(operands, 1);
                let enc = Self::encode_dp_1src_fn(sf, 0, 0b00000, 0b000001, rn, rd);
                self.emit_u32(enc);
            }
            Aarch64Opcode::REV32 => {
                let rd = self.extract_reg(operands, 0);
                let rn = self.extract_reg(operands, 1);
                // REV32 is only valid for 64-bit (sf=1)
                let enc = Self::encode_dp_1src_fn(1, 0, 0b00000, 0b000010, rn, rd);
                self.emit_u32(enc);
            }

            // Multiply variants
            Aarch64Opcode::SMULL => {
                let (rd, rn, rm) = self.extract_3reg(operands);
                let enc = self.encode_dp_3src(1, 0, 0b001, rm, 0, 31, rn, rd);
                self.emit_u32(enc);
            }
            Aarch64Opcode::UMULL => {
                let (rd, rn, rm) = self.extract_3reg(operands);
                let enc = self.encode_dp_3src(1, 0, 0b101, rm, 0, 31, rn, rd);
                self.emit_u32(enc);
            }
            Aarch64Opcode::SMULH => {
                let (rd, rn, rm) = self.extract_3reg(operands);
                let enc = self.encode_dp_3src(1, 0, 0b010, rm, 0, 31, rn, rd);
                self.emit_u32(enc);
            }
            Aarch64Opcode::UMULH => {
                let (rd, rn, rm) = self.extract_3reg(operands);
                let enc = self.encode_dp_3src(1, 0, 0b110, rm, 0, 31, rn, rd);
                self.emit_u32(enc);
            }

            // ROR (register form)
            Aarch64Opcode::ROR => {
                let sf = self.sf_from_operands(operands);
                let rd = self.extract_reg(operands, 0);
                let rn = self.extract_reg(operands, 1);
                match operands.get(2) {
                    Some(MachineOperand::Register(_)) => {
                        let rm = self.extract_reg(operands, 2);
                        let enc = self.encode_dp_2src_fn(sf, 0, rm, 0b001011, rn, rd);
                        self.emit_u32(enc);
                    }
                    Some(MachineOperand::Immediate(imm)) => {
                        // ROR by immediate = EXTR Xd, Xn, Xn, #imm
                        let n = sf;
                        let amt = (*imm as u32) & 0x3F;
                        // EXTR: [sf(1)][00][100111][N(1)][0][Rm(5)][imms(6)][Rn(5)][Rd(5)]
                        let enc = (sf << 31) | (0b00 << 29) | (0b100111 << 23)
                            | (n << 22) | (0 << 21)
                            | ((rn & 0x1F) << 16) // Rm = Rn for ROR
                            | ((amt & 0x3F) << 10)
                            | ((rn & 0x1F) << 5)
                            | (rd & 0x1F);
                        self.emit_u32(enc);
                    }
                    _ => {
                        self.emit_u32(0xD503201F); // NOP
                    }
                }
            }

            // Logical NOT variants
            Aarch64Opcode::MVN => {
                let sf = self.sf_from_operands(operands);
                let rd = self.extract_reg(operands, 0);
                let rm = self.extract_reg(operands, 1);
                // MVN Xd, Xm = ORN Xd, XZR, Xm
                let enc = Self::encode_logical_shifted_fn(sf, 0b01, 0, 1, rm, 0, 31, rd);
                self.emit_u32(enc);
            }
            Aarch64Opcode::ORN => {
                let sf = self.sf_from_operands(operands);
                let rd = self.extract_reg(operands, 0);
                let rn = self.extract_reg(operands, 1);
                let rm = self.extract_reg(operands, 2);
                let enc = Self::encode_logical_shifted_fn(sf, 0b01, 0, 1, rm, 0, rn, rd);
                self.emit_u32(enc);
            }
            Aarch64Opcode::EON => {
                let sf = self.sf_from_operands(operands);
                let rd = self.extract_reg(operands, 0);
                let rn = self.extract_reg(operands, 1);
                let rm = self.extract_reg(operands, 2);
                let enc = Self::encode_logical_shifted_fn(sf, 0b10, 0, 1, rm, 0, rn, rd);
                self.emit_u32(enc);
            }
            Aarch64Opcode::BIC => {
                let sf = self.sf_from_operands(operands);
                let rd = self.extract_reg(operands, 0);
                let rn = self.extract_reg(operands, 1);
                let rm = self.extract_reg(operands, 2);
                let enc = Self::encode_logical_shifted_fn(sf, 0b00, 0, 1, rm, 0, rn, rd);
                self.emit_u32(enc);
            }
            Aarch64Opcode::BICS => {
                let sf = self.sf_from_operands(operands);
                let rd = self.extract_reg(operands, 0);
                let rn = self.extract_reg(operands, 1);
                let rm = self.extract_reg(operands, 2);
                let enc = Self::encode_logical_shifted_fn(sf, 0b11, 0, 1, rm, 0, rn, rd);
                self.emit_u32(enc);
            }

            // TBZ / TBNZ
            Aarch64Opcode::TBZ | Aarch64Opcode::TBNZ => {
                let rt = self.extract_reg(operands, 0);
                let bit_pos = self.extract_imm(operands, 1) as u32;
                let b5 = (bit_pos >> 5) & 1;
                let b40 = bit_pos & 0x1F;
                let op: u32 = if opcode == Aarch64Opcode::TBNZ { 1 } else { 0 };

                match operands.get(2) {
                    Some(MachineOperand::Label(lbl)) => {
                        let lbl = *lbl;
                        if let Some(&target_off) = self.label_offsets.get(&lbl) {
                            let delta = (target_off as i64 - self.offset as i64) >> 2;
                            let enc = Self::encode_test_branch_fn(
                                b5,
                                op,
                                b40,
                                (delta as u32) & 0x3FFF,
                                rt,
                            );
                            self.emit_u32(enc);
                        } else {
                            self.fixups.push(BranchFixup {
                                label: lbl,
                                code_offset: self.offset,
                                fixup_type: AArch64FixupType::TestBranch14,
                            });
                            let enc = Self::encode_test_branch_fn(b5, op, b40, 0, rt);
                            self.emit_u32(enc);
                        }
                    }
                    Some(MachineOperand::Immediate(off)) => {
                        let delta = (*off >> 2) as u32;
                        let enc = Self::encode_test_branch_fn(b5, op, b40, delta & 0x3FFF, rt);
                        self.emit_u32(enc);
                    }
                    _ => {
                        let enc = Self::encode_test_branch_fn(b5, op, b40, 0, rt);
                        self.emit_u32(enc);
                    }
                }
            }

            // ADR (already handled in main match but listed here defensively)
            Aarch64Opcode::ADR => self.encode_adr(operands),

            _ => {
                // Catch-all: emit NOP for any truly unhandled opcode.
                self.emit_u32(0xD503201F);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Branch fixup resolution
    // -----------------------------------------------------------------------

    /// Resolve all forward branch fixups by patching the offset fields
    /// of previously emitted branch instructions.
    fn resolve_labels(&mut self) {
        let fixups = std::mem::take(&mut self.fixups);
        for fixup in &fixups {
            let target_offset = match self.label_offsets.get(&fixup.label) {
                Some(&off) => off,
                None => continue, // Unresolved — leave as zero (linker will handle).
            };

            let delta_bytes = target_offset as i64 - fixup.code_offset as i64;
            let delta_instr = delta_bytes >> 2; // Convert to instruction units.

            let existing = self.read_u32(fixup.code_offset);

            let patched = match fixup.fixup_type {
                AArch64FixupType::Branch26 => {
                    // Clear old imm26 field, insert new offset.
                    let mask = 0x3FFFFFF;
                    (existing & !mask) | ((delta_instr as u32) & mask)
                }
                AArch64FixupType::CondBranch19 => {
                    // imm19 at bits [23:5].
                    let mask = 0x7FFFF << 5;
                    (existing & !mask) | (((delta_instr as u32) & 0x7FFFF) << 5)
                }
                AArch64FixupType::TestBranch14 => {
                    // imm14 at bits [18:5].
                    let mask = 0x3FFF << 5;
                    (existing & !mask) | (((delta_instr as u32) & 0x3FFF) << 5)
                }
                AArch64FixupType::AdrpPage21 => {
                    // immhi at bits [23:5], immlo at bits [30:29].
                    let imm = delta_instr as u32;
                    let immlo = imm & 0x3;
                    let immhi = (imm >> 2) & 0x7FFFF;
                    let mask_lo = 0x3 << 29;
                    let mask_hi = 0x7FFFF << 5;
                    (existing & !(mask_lo | mask_hi)) | (immlo << 29) | (immhi << 5)
                }
            };

            self.patch_u32(fixup.code_offset, patched);
        }
    }
} // impl Aarch64Encoder

// ---------------------------------------------------------------------------
// Public free-standing function: bitmask immediate encoding
// ---------------------------------------------------------------------------

/// Determine whether a 64-bit value can be encoded as an AArch64
/// bitmask immediate and, if so, return `(N, immr, imms)`.
///
/// AArch64 bitmask immediates represent **repeating bit patterns** whose
/// element size is a power of two (2, 4, 8, 16, 32, or 64 bits).  Within
/// each element, the pattern is a contiguous run of ones, optionally
/// rotated.  The triple `(N, immr, imms)` encodes the pattern:
///
/// - `N`: 1 if element size is 64, 0 otherwise.
/// - `immr`: the right-rotation amount.
/// - `imms`: encodes the number of set bits minus one, combined with
///   element-size information.
///
/// The special values `0` and all-ones (for the register width) are
/// **not** encodable as bitmask immediates.
///
/// # Arguments
///
/// * `value` — The immediate value to encode.
/// * `reg_size` — 32 or 64, indicating the register width.
///
/// # Returns
///
/// `Some((N, immr, imms))` if encodable, `None` otherwise.
pub fn encode_bitmask_immediate(value: u64, reg_size: u32) -> Option<(u32, u32, u32)> {
    // All-zeros and all-ones (for the register width) are not encodable.
    let mask = if reg_size == 32 {
        0xFFFF_FFFF_u64
    } else {
        0xFFFF_FFFF_FFFF_FFFF_u64
    };
    let value = value & mask;
    if value == 0 || value == mask {
        return None;
    }

    // For 32-bit registers, the value must be a repeating 32-bit pattern.
    // We replicate the lower 32 bits into the upper 32 bits so the 64-bit
    // algorithm works uniformly.
    let value = if reg_size == 32 {
        let lo = value & 0xFFFF_FFFF;
        lo | (lo << 32)
    } else {
        value
    };

    // Try each possible element size: 2, 4, 8, 16, 32, 64.
    let sizes: &[u32] = &[2, 4, 8, 16, 32, 64];
    for &size in sizes {
        let elem_mask = if size == 64 {
            0xFFFF_FFFF_FFFF_FFFFu64
        } else {
            (1u64 << size) - 1
        };

        // Extract the base element pattern.
        let pattern = value & elem_mask;

        // Check that the value is a repeating pattern of this element size.
        let mut valid = true;
        let mut pos = size;
        while pos < 64 {
            if ((value >> pos) & elem_mask) != pattern {
                valid = false;
                break;
            }
            pos += size;
        }
        if !valid {
            continue;
        }

        // All-zeros and all-ones within the element are not encodable.
        if pattern == 0 || pattern == elem_mask {
            continue;
        }

        // Find the rotation and number of consecutive ones.
        // First, find a rotation of the pattern that places all the ones
        // in the least-significant positions (i.e., a contiguous run of
        // ones starting at bit 0).
        let mut rotated = pattern;
        let mut rotation = 0u32;
        // Rotate right until bit 0 is set AND bit (size-1) is clear,
        // meaning we found the start of the ones run.
        // Alternatively, find the right rotation to normalize.
        let mut found = false;
        for r in 0..size {
            let shifted = rotate_right(pattern, r, size);
            // Check that the shifted value is a contiguous run of ones
            // starting from bit 0.
            let ones = shifted.trailing_ones();
            if ones > 0 && ones < size {
                // Verify the rest is all zeros.
                let remaining = shifted >> ones;
                if remaining == 0 {
                    rotated = shifted;
                    rotation = r;
                    found = true;
                    break;
                }
            }
        }

        if !found {
            continue;
        }

        let ones_count = rotated.trailing_ones();

        // Compute (N, immr, imms).
        let n: u32;
        let imms: u32;

        if size == 64 {
            n = 1;
            imms = ones_count - 1;
        } else {
            n = 0;
            // imms encodes both the element size and the number of ones.
            // The top bits of imms (that correspond to the element size
            // mask) are set to identify the pattern width, and the lower
            // bits encode (ones_count - 1).
            //
            // For element size `e`, the imms field has the top bits set
            // as follows:
            //   e=2:  0b111100 | (ones-1)
            //   e=4:  0b111000 | (ones-1)
            //   e=8:  0b110000 | (ones-1)
            //   e=16: 0b100000 | (ones-1)
            //   e=32: 0b000000 | (ones-1)
            //
            // More precisely: imms = (NOT(size_mask)) | (ones - 1)
            // where size_mask = size * 2 - 1.
            let size_encoding = match size {
                2 => 0b111100,
                4 => 0b111000,
                8 => 0b110000,
                16 => 0b100000,
                32 => 0b000000,
                _ => unreachable!(),
            };
            imms = (size_encoding | (ones_count - 1)) & 0x3F;
        }

        let immr = rotation & 0x3F;

        // For 32-bit registers with N=1, the encoding is invalid.
        if reg_size == 32 && n == 1 {
            continue;
        }

        return Some((n, immr, imms));
    }

    None
}

/// Helper: rotate `value` right by `amount` bits within `width` bits.
fn rotate_right(value: u64, amount: u32, width: u32) -> u64 {
    if width == 0 || amount == 0 {
        return value;
    }
    let amount = amount % width;
    let mask = if width == 64 {
        0xFFFF_FFFF_FFFF_FFFFu64
    } else {
        (1u64 << width) - 1
    };
    let v = value & mask;
    ((v >> amount) | (v << (width - amount))) & mask
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::aarch64::isel::{Aarch64Condition, Aarch64Opcode};
    use crate::codegen::regalloc::PhysReg;
    use crate::codegen::{MachineInstr, MachineOperand};

    /// Helper: create a MachineInstr from an Aarch64Opcode and operands.
    fn make_instr(opcode: Aarch64Opcode, operands: Vec<MachineOperand>) -> MachineInstr {
        MachineInstr {
            opcode: opcode.as_u32(),
            operands,
            loc: None,
        }
    }

    /// Helper: encode a single instruction and return the 4-byte result.
    fn encode_single(opcode: Aarch64Opcode, operands: Vec<MachineOperand>) -> u32 {
        let mut enc = Aarch64Encoder::new();
        let instr = make_instr(opcode, operands);
        enc.encode_instruction(&instr);
        assert_eq!(
            enc.code.len(),
            4,
            "AArch64 instructions must be exactly 4 bytes"
        );
        enc.read_u32(0)
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

    // ===================================================================
    // Fixed-width invariant
    // ===================================================================

    #[test]
    fn test_all_instructions_are_4_bytes() {
        let test_cases = vec![
            make_instr(Aarch64Opcode::NOP, vec![]),
            make_instr(Aarch64Opcode::RET, vec![]),
            make_instr(Aarch64Opcode::ADD, vec![reg(1), reg(2), imm(100), imm(64)]),
            make_instr(Aarch64Opcode::SUB, vec![reg(3), reg(4), reg(5), imm(64)]),
        ];

        for instr in &test_cases {
            let mut enc = Aarch64Encoder::new();
            enc.encode_instruction(instr);
            assert_eq!(
                enc.code.len(),
                4,
                "Every AArch64 instruction must be 4 bytes"
            );
        }
    }

    // ===================================================================
    // Little-endian byte order
    // ===================================================================

    #[test]
    fn test_little_endian_byte_order() {
        let mut enc = Aarch64Encoder::new();
        enc.emit_u32(0xD503201F); // NOP
        assert_eq!(enc.code, vec![0x1F, 0x20, 0x03, 0xD5]);
    }

    // ===================================================================
    // NOP
    // ===================================================================

    #[test]
    fn test_nop() {
        let val = encode_single(Aarch64Opcode::NOP, vec![]);
        assert_eq!(val, 0xD503201F);
    }

    // ===================================================================
    // ADD/SUB immediate
    // ===================================================================

    #[test]
    fn test_add_x1_x2_100() {
        // ADD X1, X2, #100  (64-bit, sf=1, op=0, S=0)
        let val = encode_single(Aarch64Opcode::ADD, vec![reg(1), reg(2), imm(100), imm(64)]);
        // sf=1, op=0, S=0, 100010, sh=0, imm12=100(0x64), Rn=2, Rd=1
        let expected = (1u32 << 31) | (0b100010 << 23) | (100 << 10) | (2 << 5) | 1;
        assert_eq!(val, expected);
    }

    #[test]
    fn test_sub_w3_w4_255() {
        // SUB W3, W4, #255  (32-bit, sf=0)
        let val = encode_single(Aarch64Opcode::SUB, vec![reg(3), reg(4), imm(255), imm(32)]);
        let expected = (0u32 << 31) // sf=0
            | (1 << 30)  // op=1 (SUB)
            | (0 << 29)  // S=0
            | (0b100010 << 23)
            | (255 << 10)
            | (4 << 5)
            | 3;
        assert_eq!(val, expected);
    }

    #[test]
    fn test_cmp_x5_0() {
        // CMP X5, #0 = SUBS XZR, X5, #0
        let val = encode_single(Aarch64Opcode::SUBS, vec![reg(31), reg(5), imm(0), imm(64)]);
        let expected = (1u32 << 31) // sf=1
            | (1 << 30)  // op=1 (SUB)
            | (1 << 29)  // S=1
            | (0b100010 << 23)
            | (0 << 10)
            | (5 << 5)
            | 31;
        assert_eq!(val, expected);
    }

    // ===================================================================
    // ADD/SUB shifted register
    // ===================================================================

    #[test]
    fn test_add_x1_x2_x3_no_shift() {
        // ADD X1, X2, X3
        let val = encode_single(Aarch64Opcode::ADD, vec![reg(1), reg(2), reg(3), imm(64)]);
        let expected = (1u32 << 31) // sf=1
            | (0 << 30)  // op=0
            | (0 << 29)  // S=0
            | (0b01011 << 24)
            | (0 << 22)  // shift=LSL
            | (0 << 21)  // bit 21=0
            | (3 << 16)
            | (0 << 10)  // imm6=0
            | (2 << 5)
            | 1;
        assert_eq!(val, expected);
    }

    // ===================================================================
    // Move wide
    // ===================================================================

    #[test]
    fn test_movz_x1_0x1234() {
        // MOVZ X1, #0x1234
        let val = encode_single(
            Aarch64Opcode::MOVZ,
            vec![reg(1), imm(0x1234), imm(0), imm(64)],
        );
        let expected = (1u32 << 31) // sf=1
            | (0b10 << 29) // opc=10 (MOVZ)
            | (0b100101 << 23)
            | (0 << 21) // hw=0
            | (0x1234 << 5)
            | 1;
        assert_eq!(val, expected);
    }

    #[test]
    fn test_movk_x1_0x5678_lsl16() {
        // MOVK X1, #0x5678, LSL #16
        let val = encode_single(
            Aarch64Opcode::MOVK,
            vec![reg(1), imm(0x5678), imm(1), imm(64)], // hw=1 for LSL #16
        );
        let expected = (1u32 << 31) // sf=1
            | (0b11 << 29) // opc=11 (MOVK)
            | (0b100101 << 23)
            | (1 << 21) // hw=1
            | (0x5678 << 5)
            | 1;
        assert_eq!(val, expected);
    }

    // ===================================================================
    // Branch instructions
    // ===================================================================

    #[test]
    fn test_b_offset_8() {
        // B +8 → offset = 8 bytes = 2 instructions
        let val = encode_single(Aarch64Opcode::B, vec![imm(8)]);
        let expected = (0u32 << 31) // op=0 (B)
            | (0b00101 << 26)
            | 2; // imm26 = 8/4 = 2
        assert_eq!(val, expected);
    }

    #[test]
    fn test_bl_offset_100() {
        // BL +100 → offset = 100 bytes = 25 instructions
        let val = encode_single(Aarch64Opcode::BL, vec![imm(100)]);
        let expected = (1u32 << 31) // op=1 (BL)
            | (0b00101 << 26)
            | 25; // imm26 = 100/4 = 25
        assert_eq!(val, expected);
    }

    #[test]
    fn test_ret() {
        // RET (defaults to X30)
        let val = encode_single(Aarch64Opcode::RET, vec![]);
        // [1101011][0010][11111][000000][11110][00000]
        let expected = (0b1101011u32 << 25) | (0b0010 << 21) | (0b11111 << 16) | (30 << 5); // Rn = X30
        assert_eq!(val, expected);
    }

    #[test]
    fn test_b_cond_eq() {
        // B.EQ +12 → imm19 = 12/4 = 3, cond = EQ = 0000
        let val = encode_single(
            Aarch64Opcode::B_cond,
            vec![imm(Aarch64Condition::EQ.encoding() as i64), imm(12)],
        );
        let expected = (0b01010100u32 << 24)
            | (3 << 5)  // imm19 = 3
            | 0; // cond = EQ = 0000
        assert_eq!(val, expected);
    }

    #[test]
    fn test_cbz_x5_offset_16() {
        // CBZ X5, +16 → imm19 = 16/4 = 4
        let val = encode_single(Aarch64Opcode::CBZ, vec![reg(5), imm(16), imm(64)]);
        let expected = (1u32 << 31) // sf=1
            | (0b011010 << 25)
            | (0 << 24) // op=0 (CBZ)
            | (4 << 5)  // imm19=4
            | 5; // Rt=5
        assert_eq!(val, expected);
    }

    // ===================================================================
    // Load / Store
    // ===================================================================

    #[test]
    fn test_ldr_x1_x2_8() {
        // LDR X1, [X2, #8] → size=11, V=0, opc=01, imm12=8/8=1
        let val = encode_single(Aarch64Opcode::LDR, vec![reg(1), mem(2, 8)]);
        let expected = (0b11u32 << 30)
            | (0b111 << 27)
            | (0 << 26)   // V=0
            | (0b01 << 24)
            | (0b01 << 22) // opc=01 (load)
            | (1 << 10)    // imm12 = 8/8 = 1
            | (2 << 5)     // Rn = 2
            | 1; // Rt = 1
        assert_eq!(val, expected);
    }

    #[test]
    fn test_str_w3_sp_16() {
        // STR W3, [SP, #16] → size=10, V=0, opc=00, imm12=16/4=4, Rn=31(SP)
        let val = encode_single(Aarch64Opcode::STRW, vec![reg(3), mem(31, 16)]);
        let expected = (0b10u32 << 30)
            | (0b111 << 27)
            | (0 << 26)
            | (0b01 << 24)
            | (0b00 << 22) // opc=00 (store)
            | (4 << 10)    // imm12 = 16/4 = 4
            | (31 << 5)
            | 3;
        assert_eq!(val, expected);
    }

    #[test]
    fn test_ldrb_w4_x5_0() {
        // LDRB W4, [X5, #0] → size=00, V=0, opc=01, imm12=0
        let val = encode_single(Aarch64Opcode::LDRB, vec![reg(4), mem(5, 0)]);
        let expected = (0b00u32 << 30)
            | (0b111 << 27)
            | (0 << 26)
            | (0b01 << 24)
            | (0b01 << 22)
            | (0 << 10)
            | (5 << 5)
            | 4;
        assert_eq!(val, expected);
    }

    #[test]
    fn test_ldp_x1_x2_sp_16() {
        // LDP X1, X2, [SP, #16]
        let val = encode_single(Aarch64Opcode::LDP, vec![reg(1), reg(2), mem(31, 16)]);
        // opc=10, V=0, 010, L=1, imm7 = 16/8 = 2, Rt2=2, Rn=31, Rt=1
        let expected = (0b10u32 << 30)
            | (0b101 << 27)
            | (0 << 26)  // V=0
            | (0b010 << 23)
            | (1 << 22)  // L=1 (load)
            | (2 << 15)  // imm7 = 2
            | (2 << 10)  // Rt2 = 2
            | (31 << 5)  // Rn = SP
            | 1; // Rt = 1
        assert_eq!(val, expected);
    }

    // ===================================================================
    // Conditional select
    // ===================================================================

    #[test]
    fn test_csel_x1_x2_x3_eq() {
        // CSEL X1, X2, X3, EQ
        let val = encode_single(
            Aarch64Opcode::CSEL,
            vec![
                reg(1),
                reg(2),
                reg(3),
                imm(Aarch64Condition::EQ.encoding() as i64),
                imm(64),
            ],
        );
        let expected = (1u32 << 31) // sf=1
            | (0 << 30)  // op=0
            | (0 << 29)  // S=0
            | (0b11010100u32 << 21)
            | (3 << 16)  // Rm=3
            | (0 << 12)  // cond=EQ=0000
            | (0 << 10)  // op2=0
            | (2 << 5)   // Rn=2
            | 1; // Rd=1
        assert_eq!(val, expected);
    }

    #[test]
    fn test_csinc_x4_x5_x6_ne() {
        // CSINC X4, X5, X6, NE
        let val = encode_single(
            Aarch64Opcode::CSINC,
            vec![
                reg(4),
                reg(5),
                reg(6),
                imm(Aarch64Condition::NE.encoding() as i64),
                imm(64),
            ],
        );
        let expected = (1u32 << 31)
            | (0 << 30)
            | (0 << 29)
            | (0b11010100u32 << 21)
            | (6 << 16)
            | (1 << 12)  // cond=NE=0001
            | (1 << 10)  // op2=1 (CSINC)
            | (5 << 5)
            | 4;
        assert_eq!(val, expected);
    }

    // ===================================================================
    // Floating-point
    // ===================================================================

    #[test]
    fn test_fadd_d0_d1_d2() {
        // FADD D0, D1, D2  (double precision)
        // V0-V31 = PhysReg(32)-PhysReg(63)
        let val = encode_single(
            Aarch64Opcode::FADD_D,
            vec![reg(32), reg(33), reg(34)], // D0, D1, D2
        );
        // ftype=01 (double), opcode=0010 (FADD)
        let expected = (0b0001_1110u32 << 24)
            | (0b01 << 22)  // ftype=01
            | (1 << 21)
            | (2 << 16)     // Rm=2 (D2 → encoding 2)
            | (0b0010 << 12)
            | (0b10 << 10)
            | (1 << 5)      // Rn=1 (D1 → encoding 1)
            | 0; // Rd=0 (D0 → encoding 0)
        assert_eq!(val, expected);
    }

    #[test]
    fn test_fcmp_d3_d4() {
        // FCMP D3, D4
        let val = encode_single(
            Aarch64Opcode::FCMP_D,
            vec![reg(35), reg(36)], // D3, D4
        );
        let expected = (0b0001_1110u32 << 24)
            | (0b01 << 22)
            | (1 << 21)
            | (4 << 16)     // Rm=4
            | (0b001000 << 10)
            | (3 << 5)      // Rn=3
            | 0b00000; // opcode2=0
        assert_eq!(val, expected);
    }

    #[test]
    fn test_scvtf_d5_x6() {
        // SCVTF D5, X6
        let val = encode_single(
            Aarch64Opcode::SCVTF_D,
            vec![reg(37), reg(6)], // D5, X6
        );
        // sf=1, ftype=01 (double), rmode=00, opcode=010
        let expected = (1u32 << 31)
            | (0b0001_1110u32 << 24)
            | (0b01 << 22)
            | (1 << 21)
            | (0b00 << 19)
            | (0b010 << 16)
            | (6 << 5)
            | 5;
        assert_eq!(val, expected);
    }

    // ===================================================================
    // Bitmask immediate encoding
    // ===================================================================

    #[test]
    fn test_bitmask_immediate_0xff() {
        // 0xFF = 8 consecutive ones = element size 8, ones=8
        let result = encode_bitmask_immediate(0xFF, 64);
        assert!(result.is_some());
        let (n, immr, imms) = result.unwrap();
        // For element size=8, N=0, imms starts with 0b110000
        // ones=8 means all bits set in 8-bit element → not encodable
        // Actually 0xFF repeated in 64 bits = 0x00FF00FF00FF00FF
        // Let me check: 0xFF in 64-bit is just 0x00000000000000FF
        // which is a 64-bit value with 8 ones at the bottom.
        // Element size = 64, N=1, ones=8, rotation=0
        assert_eq!(n, 1);
        assert_eq!(immr, 0);
        assert_eq!(imms, 7); // ones - 1 = 8 - 1 = 7
    }

    #[test]
    fn test_bitmask_immediate_not_encodable_zero() {
        assert!(encode_bitmask_immediate(0, 64).is_none());
    }

    #[test]
    fn test_bitmask_immediate_not_encodable_all_ones() {
        assert!(encode_bitmask_immediate(0xFFFFFFFFFFFFFFFF, 64).is_none());
    }

    #[test]
    fn test_bitmask_immediate_32bit_all_ones() {
        assert!(encode_bitmask_immediate(0xFFFFFFFF, 32).is_none());
    }

    #[test]
    fn test_bitmask_immediate_repeating_pattern() {
        // 0x5555555555555555 = alternating 01 pattern (element size 2, ones=1)
        let result = encode_bitmask_immediate(0x5555555555555555, 64);
        assert!(result.is_some());
    }

    #[test]
    fn test_bitmask_immediate_0xffff() {
        // 0xFFFF = 16 consecutive ones at bottom (in 64-bit context)
        let result = encode_bitmask_immediate(0xFFFF, 64);
        assert!(result.is_some());
        let (n, immr, imms) = result.unwrap();
        assert_eq!(n, 1);
        assert_eq!(immr, 0);
        assert_eq!(imms, 15); // 16 - 1 = 15
    }

    // ===================================================================
    // encode_function — multi-instruction test with branch fixup
    // ===================================================================

    #[test]
    fn test_encode_function_basic() {
        let mut enc = Aarch64Encoder::new();
        let instrs = vec![
            make_instr(Aarch64Opcode::NOP, vec![]),
            make_instr(Aarch64Opcode::RET, vec![]),
        ];
        let code = enc.encode_function(&instrs);
        assert_eq!(code.len(), 8); // 2 instructions × 4 bytes
    }

    #[test]
    fn test_code_size() {
        let mut enc = Aarch64Encoder::new();
        assert_eq!(enc.code_size(), 0);
        let instrs = vec![
            make_instr(Aarch64Opcode::NOP, vec![]),
            make_instr(Aarch64Opcode::NOP, vec![]),
            make_instr(Aarch64Opcode::NOP, vec![]),
        ];
        enc.encode_function(&instrs);
        assert_eq!(enc.code_size(), 12);
    }

    #[test]
    fn test_get_relocations_bl_symbol() {
        let mut enc = Aarch64Encoder::new();
        let instrs = vec![make_instr(
            Aarch64Opcode::BL,
            vec![MachineOperand::Symbol("printf".to_string())],
        )];
        enc.encode_function(&instrs);
        let relocs = enc.get_relocations();
        assert_eq!(relocs.len(), 1);
        assert_eq!(relocs[0].symbol, "printf");
        assert_eq!(relocs[0].reloc_type, RelocationType::Aarch64_CALL26);
    }

    // ===================================================================
    // MOV register
    // ===================================================================

    #[test]
    fn test_mov_x1_x2() {
        // MOV X1, X2 = ORR X1, XZR, X2
        let val = encode_single(Aarch64Opcode::MOV, vec![reg(1), reg(2), imm(64)]);
        let expected = Aarch64Encoder::encode_logical_shifted_fn(1, 0b01, 0, 0, 2, 0, 31, 1);
        assert_eq!(val, expected);
    }

    // ===================================================================
    // Division
    // ===================================================================

    #[test]
    fn test_sdiv_x1_x2_x3() {
        let val = encode_single(Aarch64Opcode::SDIV, vec![reg(1), reg(2), reg(3), imm(64)]);
        // sf=1, 0, S=0, 11010110, Rm=3, opcode=000011, Rn=2, Rd=1
        let expected =
            (1u32 << 31) | (0b11010110u32 << 21) | (3 << 16) | (0b000011 << 10) | (2 << 5) | 1;
        assert_eq!(val, expected);
    }

    // ===================================================================
    // MUL (MADD with Ra=XZR)
    // ===================================================================

    #[test]
    fn test_mul_x1_x2_x3() {
        let val = encode_single(Aarch64Opcode::MUL, vec![reg(1), reg(2), reg(3), imm(64)]);
        // MADD X1, X2, X3, XZR
        // sf=1, op54=00, 11011, op31=000, Rm=3, o0=0, Ra=31, Rn=2, Rd=1
        let expected = (1u32 << 31)
            | (0b11011 << 24)
            | (0b000 << 21)
            | (3 << 16)
            | (0 << 15)
            | (31 << 10)
            | (2 << 5)
            | 1;
        assert_eq!(val, expected);
    }

    // ===================================================================
    // Shift immediate
    // ===================================================================

    #[test]
    fn test_lsr_x1_x2_imm4() {
        // LSR X1, X2, #4 = UBFM X1, X2, #4, #63
        let val = encode_single(Aarch64Opcode::LSR, vec![reg(1), reg(2), imm(4), imm(64)]);
        // sf=1, opc=10, 100110, N=1, immr=4, imms=63, Rn=2, Rd=1
        let expected = (1u32 << 31)
            | (0b10 << 29)
            | (0b100110 << 23)
            | (1 << 22) // N=1
            | (4 << 16) // immr=4
            | (63 << 10) // imms=63
            | (2 << 5)
            | 1;
        assert_eq!(val, expected);
    }

    // ===================================================================
    // BR (register branch)
    // ===================================================================

    #[test]
    fn test_br_x10() {
        // BR X10
        let val = encode_single(Aarch64Opcode::BR, vec![reg(10)]);
        let expected = (0b1101011u32 << 25)
            | (0b0000 << 21) // opc=0000 for BR
            | (0b11111 << 16)
            | (10 << 5);
        assert_eq!(val, expected);
    }

    #[test]
    fn test_blr_x8() {
        // BLR X8
        let val = encode_single(Aarch64Opcode::BLR, vec![reg(8)]);
        let expected = (0b1101011u32 << 25)
            | (0b0001 << 21) // opc=0001 for BLR
            | (0b11111 << 16)
            | (8 << 5);
        assert_eq!(val, expected);
    }

    // ===================================================================
    // ADRP relocation
    // ===================================================================

    #[test]
    fn test_adrp_relocation() {
        let mut enc = Aarch64Encoder::new();
        let instr = make_instr(
            Aarch64Opcode::ADRP,
            vec![reg(0), MachineOperand::Symbol("my_global".to_string())],
        );
        enc.encode_instruction(&instr);
        assert_eq!(enc.code.len(), 4);
        let relocs = enc.get_relocations();
        assert_eq!(relocs.len(), 1);
        assert_eq!(relocs[0].symbol, "my_global");
        assert_eq!(
            relocs[0].reloc_type,
            RelocationType::Aarch64_ADR_PREL_PG_HI21
        );
    }

    // ===================================================================
    // SXTW / SXTB / UXTB
    // ===================================================================

    #[test]
    fn test_sxtw() {
        // SXTW X1, W2 = SBFM X1, X2, #0, #31
        let val = encode_single(Aarch64Opcode::SXTW, vec![reg(1), reg(2)]);
        // sf=1, opc=00, 100110, N=1, immr=0, imms=31, Rn=2, Rd=1
        let expected = (1u32 << 31)
            | (0b00 << 29)
            | (0b100110 << 23)
            | (1 << 22)
            | (0 << 16)
            | (31 << 10)
            | (2 << 5)
            | 1;
        assert_eq!(val, expected);
    }

    // ===================================================================
    // Patch / read roundtrip
    // ===================================================================

    #[test]
    fn test_patch_u32_roundtrip() {
        let mut enc = Aarch64Encoder::new();
        enc.emit_u32(0xDEADBEEF);
        assert_eq!(enc.read_u32(0), 0xDEADBEEF);
        enc.patch_u32(0, 0xCAFEBABE);
        assert_eq!(enc.read_u32(0), 0xCAFEBABE);
    }

    // ===================================================================
    // Branch fixup (forward branch)
    // ===================================================================

    #[test]
    fn test_forward_branch_fixup() {
        let mut enc = Aarch64Encoder::new();
        // Emit a B to label 42 (not yet seen).
        let instr_b = make_instr(Aarch64Opcode::B, vec![MachineOperand::Label(42)]);
        enc.encode_instruction(&instr_b);

        // Emit a NOP as the target of label 42.
        enc.record_label(42);
        let instr_nop = make_instr(Aarch64Opcode::NOP, vec![]);
        enc.encode_instruction(&instr_nop);

        // Resolve fixups.
        enc.resolve_labels();

        // The B instruction at offset 0 should now have imm26 = 1
        // (target at offset 4, delta = 4 bytes = 1 instruction).
        let b_enc = enc.read_u32(0);
        let imm26 = b_enc & 0x3FFFFFF;
        assert_eq!(imm26, 1);
    }
}
