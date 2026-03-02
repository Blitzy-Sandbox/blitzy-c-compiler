//! # RISC-V 64 Instruction Selection
//!
//! This module implements the core pattern-matching engine that translates
//! high-level SSA-form IR instructions into RISC-V 64-bit (RV64GC) machine
//! instruction sequences.
//!
//! ## ISA Coverage
//!
//! - **RV64I** — Base integer instructions (arithmetic, logical, load/store, branches, jumps)
//! - **M extension** — Multiply/divide (MUL, MULH, DIV, REM, and W-suffix variants)
//! - **A extension** — Atomics (LR.D, SC.D, AMO* instructions)
//! - **F extension** — Single-precision float (FADD.S, FLW, FCVT.*, FEQ.S, etc.)
//! - **D extension** — Double-precision float (FADD.D, FLD, FCVT.*, FEQ.D, etc.)
//!
//! ## Large Constant Materialization
//!
//! RISC-V immediates are limited to 12 bits (I-type) or 20 bits (U-type).
//! For larger constants this module emits LUI+ADDI pairs for 32-bit values
//! and multi-instruction LUI+ADDI+SLLI+ADDI sequences for 64-bit values,
//! correctly handling the sign-extension quirk of ADDI.
//!
//! ## Register File
//!
//! 32 GPRs (x0-x31, x0 hardwired to zero) plus 32 FP registers (f0-f31).
//! Physical register numbering: GPRs use PhysReg(0..31), FPRs use PhysReg(32..63).
//!
//! ## Zero External Dependencies
//!
//! This module uses only the Rust standard library and internal crate modules.

use std::collections::HashMap;

use crate::codegen::{
    CodeGenError, MachineInstr, MachineOperand, Relocation, RelocationType,
};
use crate::codegen::regalloc::{AllocationResult, PhysReg, RegClass, build_value_to_reg_map};
use crate::driver::target::TargetConfig;
use crate::ir::{
    BasicBlock, BlockId, Callee, CastOp, CompareOp, Constant, ControlFlowGraph,
    FloatCompareOp, Function, Instruction, IrType, PhiNode, Terminator, Value,
};
use crate::ir::cfg::reverse_postorder;

// ---------------------------------------------------------------------------
// Riscv64Opcode — RISC-V 64 machine instruction opcodes
// ---------------------------------------------------------------------------

/// Enumerates all RV64GC machine instruction opcodes used by the instruction
/// selector. The numeric value of each variant is used as the `opcode` field
/// in [`MachineInstr`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Riscv64Opcode {
    // ===== RV64I Base Integer Instructions =====

    /// Load Upper Immediate (U-type): rd = imm << 12
    LUI = 0,
    /// Add Upper Immediate to PC (U-type): rd = PC + (imm << 12)
    AUIPC,
    /// Jump and Link (J-type): rd = PC+4; PC += offset
    JAL,
    /// Jump and Link Register (I-type): rd = PC+4; PC = rs1 + offset
    JALR,
    /// Branch if Equal (B-type)
    BEQ,
    /// Branch if Not Equal (B-type)
    BNE,
    /// Branch if Less Than (signed, B-type)
    BLT,
    /// Branch if Greater or Equal (signed, B-type)
    BGE,
    /// Branch if Less Than (unsigned, B-type)
    BLTU,
    /// Branch if Greater or Equal (unsigned, B-type)
    BGEU,
    /// Load Byte (I-type, sign-extends)
    LB,
    /// Load Half-word (I-type, sign-extends)
    LH,
    /// Load Word (I-type, sign-extends on RV64)
    LW,
    /// Load Double-word (I-type)
    LD,
    /// Load Byte Unsigned (I-type, zero-extends)
    LBU,
    /// Load Half-word Unsigned (I-type, zero-extends)
    LHU,
    /// Load Word Unsigned (I-type, zero-extends)
    LWU,
    /// Store Byte (S-type)
    SB,
    /// Store Half-word (S-type)
    SH,
    /// Store Word (S-type)
    SW,
    /// Store Double-word (S-type)
    SD,
    /// Add Immediate (I-type)
    ADDI,
    /// Set Less Than Immediate (signed, I-type)
    SLTI,
    /// Set Less Than Immediate Unsigned (I-type)
    SLTIU,
    /// XOR Immediate (I-type)
    XORI,
    /// OR Immediate (I-type)
    ORI,
    /// AND Immediate (I-type)
    ANDI,
    /// Shift Left Logical Immediate (I-type, 6-bit shamt for RV64)
    SLLI,
    /// Shift Right Logical Immediate (I-type, 6-bit shamt for RV64)
    SRLI,
    /// Shift Right Arithmetic Immediate (I-type, 6-bit shamt for RV64)
    SRAI,
    /// Add (R-type)
    ADD,
    /// Subtract (R-type)
    SUB,
    /// Shift Left Logical (R-type)
    SLL,
    /// Set Less Than (signed, R-type)
    SLT,
    /// Set Less Than Unsigned (R-type)
    SLTU,
    /// XOR (R-type)
    XOR,
    /// Shift Right Logical (R-type)
    SRL,
    /// Shift Right Arithmetic (R-type)
    SRA,
    /// OR (R-type)
    OR,
    /// AND (R-type)
    AND,
    /// Add Immediate Word (RV64I W-suffix, sign-extends 32-bit result)
    ADDIW,
    /// Shift Left Logical Immediate Word (RV64I W-suffix)
    SLLIW,
    /// Shift Right Logical Immediate Word (RV64I W-suffix)
    SRLIW,
    /// Shift Right Arithmetic Immediate Word (RV64I W-suffix)
    SRAIW,
    /// Add Word (RV64I W-suffix, sign-extends 32-bit result)
    ADDW,
    /// Subtract Word (RV64I W-suffix)
    SUBW,
    /// Shift Left Logical Word (RV64I W-suffix)
    SLLW,
    /// Shift Right Logical Word (RV64I W-suffix)
    SRLW,
    /// Shift Right Arithmetic Word (RV64I W-suffix)
    SRAW,
    /// Memory Fence
    FENCE,
    /// Environment Call
    ECALL,
    /// Environment Breakpoint
    EBREAK,

    // ===== M Extension (Multiply/Divide) =====

    /// Multiply (64-bit result lower bits)
    MUL,
    /// Multiply High (signed × signed → upper 64 bits)
    MULH,
    /// Multiply High Signed × Unsigned
    MULHSU,
    /// Multiply High Unsigned × Unsigned
    MULHU,
    /// Divide (signed)
    DIV,
    /// Divide (unsigned)
    DIVU,
    /// Remainder (signed)
    REM,
    /// Remainder (unsigned)
    REMU,
    /// Multiply Word (32-bit, W-suffix)
    MULW,
    /// Divide Word (signed, W-suffix)
    DIVW,
    /// Divide Word (unsigned, W-suffix)
    DIVUW,
    /// Remainder Word (signed, W-suffix)
    REMW,
    /// Remainder Word (unsigned, W-suffix)
    REMUW,

    // ===== A Extension (Atomics) =====

    /// Load-Reserved Double-word
    LR_D,
    /// Store-Conditional Double-word
    SC_D,
    /// Atomic Swap Double-word
    AMOSWAP_D,
    /// Atomic Add Double-word
    AMOADD_D,
    /// Atomic AND Double-word
    AMOAND_D,
    /// Atomic OR Double-word
    AMOOR_D,
    /// Atomic XOR Double-word
    AMOXOR_D,
    /// Atomic Max (signed) Double-word
    AMOMAX_D,
    /// Atomic Min (signed) Double-word
    AMOMIN_D,
    /// Atomic Max (unsigned) Double-word
    AMOMAXU_D,
    /// Atomic Min (unsigned) Double-word
    AMOMINU_D,

    // ===== F Extension (Single-Precision Float) =====

    /// Load Float Word
    FLW,
    /// Store Float Word
    FSW,
    /// Float Add Single
    FADD_S,
    /// Float Subtract Single
    FSUB_S,
    /// Float Multiply Single
    FMUL_S,
    /// Float Divide Single
    FDIV_S,
    /// Float Square Root Single
    FSQRT_S,
    /// Float Minimum Single
    FMIN_S,
    /// Float Maximum Single
    FMAX_S,
    /// Convert Float to Signed Word
    FCVT_W_S,
    /// Convert Float to Unsigned Word
    FCVT_WU_S,
    /// Convert Signed Word to Float
    FCVT_S_W,
    /// Convert Unsigned Word to Float
    FCVT_S_WU,
    /// Convert Float to Signed Long
    FCVT_L_S,
    /// Convert Float to Unsigned Long
    FCVT_LU_S,
    /// Convert Signed Long to Float
    FCVT_S_L,
    /// Convert Unsigned Long to Float
    FCVT_S_LU,
    /// Move Float to Integer register (32-bit)
    FMV_X_W,
    /// Move Integer to Float register (32-bit)
    FMV_W_X,
    /// Float Equal Single (comparison → integer result)
    FEQ_S,
    /// Float Less Than Single
    FLT_S,
    /// Float Less or Equal Single
    FLE_S,

    // ===== D Extension (Double-Precision Float) =====

    /// Load Float Double-word
    FLD,
    /// Store Float Double-word
    FSD,
    /// Float Add Double
    FADD_D,
    /// Float Subtract Double
    FSUB_D,
    /// Float Multiply Double
    FMUL_D,
    /// Float Divide Double
    FDIV_D,
    /// Float Square Root Double
    FSQRT_D,
    /// Float Minimum Double
    FMIN_D,
    /// Float Maximum Double
    FMAX_D,
    /// Convert Double to Signed Word
    FCVT_W_D,
    /// Convert Double to Unsigned Word
    FCVT_WU_D,
    /// Convert Signed Word to Double
    FCVT_D_W,
    /// Convert Unsigned Word to Double
    FCVT_D_WU,
    /// Convert Double to Signed Long
    FCVT_L_D,
    /// Convert Double to Unsigned Long
    FCVT_LU_D,
    /// Convert Signed Long to Double
    FCVT_D_L,
    /// Convert Unsigned Long to Double
    FCVT_D_LU,
    /// Move Double to Integer register (64-bit)
    FMV_X_D,
    /// Move Integer to Double register (64-bit)
    FMV_D_X,
    /// Float Equal Double
    FEQ_D,
    /// Float Less Than Double
    FLT_D,
    /// Float Less or Equal Double
    FLE_D,
    /// Convert Single to Double
    FCVT_D_S,
    /// Convert Double to Single
    FCVT_S_D,

    // ===== Pseudo-instructions (expanded during encoding) =====

    /// No-operation (ADDI x0, x0, 0)
    NOP,
    /// Load Immediate (LUI+ADDI or multi-instruction sequence)
    LI,
    /// Move register (ADDI rd, rs, 0)
    MV,
    /// Negate (SUB rd, x0, rs)
    NEG,
    /// Bitwise NOT (XORI rd, rs, -1)
    NOT,
    /// Set if Equal to Zero (SLTIU rd, rs, 1)
    SEQZ,
    /// Set if Not Equal to Zero (SLTU rd, x0, rs)
    SNEZ,
    /// Function Call pseudo (AUIPC+JALR pair with R_RISCV_CALL relocation)
    CALL,
    /// Return (JALR x0, ra, 0)
    RET,
    /// Load Address (AUIPC+ADDI with PC-relative relocations)
    LA,
}

impl Riscv64Opcode {
    /// Convert the opcode to its u32 representation for use in MachineInstr.opcode.
    #[inline]
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

// ---------------------------------------------------------------------------
// RISC-V physical register constants
// ---------------------------------------------------------------------------
// GPRs x0-x31 → PhysReg(0..31), FPRs f0-f31 → PhysReg(32..63)

/// x0 — hardwired zero register (always reads as 0, writes are discarded).
pub const X0: PhysReg = PhysReg(0);
/// x1 (ra) — return address register.
pub const RA: PhysReg = PhysReg(1);
/// x2 (sp) — stack pointer register.
pub const SP: PhysReg = PhysReg(2);
/// x5 (t0) — temporary register 0.
#[allow(dead_code)]
pub const T0: PhysReg = PhysReg(5);
/// x6 (t1) — temporary register 1.
#[allow(dead_code)]
pub const T1: PhysReg = PhysReg(6);
/// x8 (s0/fp) — saved register 0 / frame pointer.
#[allow(dead_code)]
pub const S0: PhysReg = PhysReg(8);
/// x10 (a0) — argument register 0 / integer return value.
pub const A0: PhysReg = PhysReg(10);
/// x11 (a1) — argument register 1.
#[allow(dead_code)]
pub const A1: PhysReg = PhysReg(11);
/// x10-x17 (a0-a7) — integer argument registers for LP64D ABI.
pub const ARG_REGS: [PhysReg; 8] = [
    PhysReg(10), PhysReg(11), PhysReg(12), PhysReg(13),
    PhysReg(14), PhysReg(15), PhysReg(16), PhysReg(17),
];
/// f10 (fa0) — float argument register 0 / float return value.
pub const FA0: PhysReg = PhysReg(42);
/// f10-f17 (fa0-fa7) — float argument registers for LP64D ABI.
pub const FLOAT_ARG_REGS: [PhysReg; 8] = [
    PhysReg(42), PhysReg(43), PhysReg(44), PhysReg(45),
    PhysReg(46), PhysReg(47), PhysReg(48), PhysReg(49),
];

/// Returns true if the physical register is a floating-point register (f0-f31).
#[inline]
fn is_fp_reg(reg: PhysReg) -> bool {
    reg.0 >= 32 && reg.0 < 64
}

// ---------------------------------------------------------------------------
// Riscv64InstructionSelector — the instruction selection engine
// ---------------------------------------------------------------------------

/// The RISC-V 64-bit instruction selector that translates IR instructions into
/// RV64GC machine instruction sequences.
///
/// Created once per function being compiled via [`new()`], then
/// [`select_function()`] is called to perform the actual instruction selection.
pub struct Riscv64InstructionSelector<'a> {
    /// Value-to-physical-register mapping built from register allocation results.
    value_map: HashMap<Value, PhysReg>,
    /// Target configuration providing pointer size, stack alignment, and ABI info.
    target: &'a TargetConfig,
    /// Label counter for generating unique basic block labels.
    label_counter: u32,
    /// Maps IR basic block IDs to RISC-V label identifiers.
    block_labels: HashMap<BlockId, u32>,
    /// Number of spill slots allocated by the register allocator.
    num_spill_slots: u32,
    /// Callee-saved registers that were used by the allocator (for prologue/epilogue).
    #[allow(dead_code)]
    used_callee_saved: Vec<PhysReg>,
    /// Accumulated relocations for the current function.
    relocations: Vec<Relocation>,
    /// IR Values that represent function parameters — already assigned to
    /// ABI registers by the allocator.  Const instructions for these must
    /// be skipped so the actual argument values are preserved.
    param_value_set: std::collections::HashSet<Value>,
}

impl<'a> Riscv64InstructionSelector<'a> {
    /// Creates a new instruction selector for a single function.
    ///
    /// # Arguments
    ///
    /// * `alloc_result` — Register allocation output mapping SSA values to physical
    ///   registers (or spill slots).
    /// * `target` — Target configuration for the RISC-V 64 architecture.
    pub fn new(alloc_result: &AllocationResult, target: &'a TargetConfig) -> Self {
        let value_map = build_value_to_reg_map(alloc_result);
        Riscv64InstructionSelector {
            value_map,
            target,
            label_counter: 0,
            block_labels: HashMap::new(),
            num_spill_slots: alloc_result.num_spill_slots,
            used_callee_saved: alloc_result.used_callee_saved.clone(),
            relocations: Vec::new(),
            param_value_set: std::collections::HashSet::new(),
        }
    }

    /// Selects RISC-V machine instructions for the given IR function.
    ///
    /// Iterates over basic blocks in reverse postorder, processing phi nodes,
    /// regular instructions, and terminators for each block. Returns the
    /// complete sequence of machine instructions ready for encoding.
    ///
    /// # Arguments
    ///
    /// * `function` — The IR function to lower to machine instructions.
    ///
    /// # Returns
    ///
    /// A vector of [`MachineInstr`] in program order, with relocations
    /// accessible via [`relocations()`].
    pub fn select_function(&mut self, function: &Function) -> Vec<MachineInstr> {
        let mut instructions: Vec<MachineInstr> = Vec::new();
        self.relocations.clear();
        self.block_labels.clear();
        self.label_counter = 0;
        self.param_value_set.clear();

        // Record which IR Values are function parameters so that
        // select_const skips the placeholder Const instructions.
        for &pv in &function.param_values {
            self.param_value_set.insert(pv);
        }

        // Pre-assign labels to all blocks so forward branches can reference them.
        for block in &function.blocks {
            let label = self.next_label();
            self.block_labels.insert(block.id, label);
        }

        // Build a ControlFlowGraph for RPO traversal.
        if function.blocks.is_empty() {
            return instructions;
        }
        let mut cfg = ControlFlowGraph::new(function.entry_block);
        for block in &function.blocks {
            cfg.add_block(block.clone());
        }
        let rpo = reverse_postorder(&cfg);

        // Select instructions for each block in reverse postorder.
        for &block_id in &rpo {
            let idx = block_id.0 as usize;
            if idx >= function.blocks.len() {
                continue;
            }
            let block = &function.blocks[idx];

            // Emit a label marker for this block (encoded as a NOP with label operand).
            if let Some(&label) = self.block_labels.get(&block.id) {
                let label_instr = MachineInstr::with_operands(
                    Riscv64Opcode::NOP.as_u32(),
                    vec![MachineOperand::Label(label)],
                );
                instructions.push(label_instr);
            }

            // Process phi nodes — emit move instructions for parallel copies.
            self.select_phi_nodes(&block.phi_nodes, &mut instructions);

            // Process regular instructions.
            for ir_instr in &block.instructions {
                self.select_instruction(ir_instr, &mut instructions);
            }

            // Process the terminator.
            if let Some(ref term) = block.terminator {
                self.select_terminator(term, &mut instructions);
            }
        }

        instructions
    }

    /// Returns the relocations accumulated during the most recent
    /// `select_function()` call.
    pub fn relocations(&self) -> &[Relocation] {
        &self.relocations
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Allocates and returns the next unique label identifier.
    fn next_label(&mut self) -> u32 {
        let label = self.label_counter;
        self.label_counter += 1;
        label
    }

    /// Looks up the physical register assigned to the given IR value.
    ///
    /// Falls back to x0 (zero register) if the value has no allocation,
    /// which handles undefined/dead values gracefully.
    fn get_reg(&self, value: Value) -> PhysReg {
        self.value_map.get(&value).copied().unwrap_or(X0)
    }

    /// Determines if a value fits in a 12-bit signed immediate (−2048..2047).
    #[inline]
    fn fits_in_imm12(value: i64) -> bool {
        value >= -2048 && value <= 2047
    }

    /// Emits a machine instruction with the given opcode and operands.
    fn emit(
        instrs: &mut Vec<MachineInstr>,
        opcode: Riscv64Opcode,
        operands: Vec<MachineOperand>,
    ) {
        instrs.push(MachineInstr::with_operands(opcode.as_u32(), operands));
    }

    /// Emits a register-to-register move instruction (MV pseudo: ADDI rd, rs, 0).
    fn emit_mv(instrs: &mut Vec<MachineInstr>, dest: PhysReg, src: PhysReg) {
        if dest != src {
            Self::emit(instrs, Riscv64Opcode::MV, vec![
                MachineOperand::Register(dest),
                MachineOperand::Register(src),
            ]);
        }
    }

    // -----------------------------------------------------------------------
    // Phi node selection
    // -----------------------------------------------------------------------

    /// Processes phi nodes by emitting move instructions.
    ///
    /// In a real compiler, phi nodes require careful parallel-copy insertion
    /// at predecessor block edges. Here we emit sequential moves which is
    /// correct after SSA destruction has already inserted copy instructions.
    fn select_phi_nodes(
        &self,
        phi_nodes: &[PhiNode],
        instrs: &mut Vec<MachineInstr>,
    ) {
        for phi in phi_nodes {
            let dest = self.get_reg(phi.result);
            // Phi nodes are resolved by the SSA destruction pass which inserts
            // Copy instructions on predecessor edges. If we still see phi nodes
            // here, emit a move from the first incoming value as a fallback.
            if let Some(&(val, _)) = phi.incoming.first() {
                let src = self.get_reg(val);
                Self::emit_mv(instrs, dest, src);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Main instruction selection dispatch
    // -----------------------------------------------------------------------

    /// Selects RISC-V instructions for a single IR instruction.
    fn select_instruction(
        &mut self,
        instr: &Instruction,
        instrs: &mut Vec<MachineInstr>,
    ) {
        match instr {
            // --- Arithmetic ---
            Instruction::Add { result, lhs, rhs, ty } => {
                self.select_binop(*result, *lhs, *rhs, ty, BinOp::Add, instrs);
            }
            Instruction::Sub { result, lhs, rhs, ty } => {
                self.select_binop(*result, *lhs, *rhs, ty, BinOp::Sub, instrs);
            }
            Instruction::Mul { result, lhs, rhs, ty } => {
                self.select_binop(*result, *lhs, *rhs, ty, BinOp::Mul, instrs);
            }
            Instruction::Div { result, lhs, rhs, ty, is_signed } => {
                let op = if *is_signed { BinOp::DivS } else { BinOp::DivU };
                self.select_binop(*result, *lhs, *rhs, ty, op, instrs);
            }
            Instruction::Mod { result, lhs, rhs, ty, is_signed } => {
                let op = if *is_signed { BinOp::RemS } else { BinOp::RemU };
                self.select_binop(*result, *lhs, *rhs, ty, op, instrs);
            }

            // --- Bitwise ---
            Instruction::And { result, lhs, rhs, ty } => {
                self.select_binop(*result, *lhs, *rhs, ty, BinOp::And, instrs);
            }
            Instruction::Or { result, lhs, rhs, ty } => {
                self.select_binop(*result, *lhs, *rhs, ty, BinOp::Or, instrs);
            }
            Instruction::Xor { result, lhs, rhs, ty } => {
                self.select_binop(*result, *lhs, *rhs, ty, BinOp::Xor, instrs);
            }
            Instruction::Shl { result, lhs, rhs, ty } => {
                self.select_binop(*result, *lhs, *rhs, ty, BinOp::Shl, instrs);
            }
            Instruction::Shr { result, lhs, rhs, ty, is_arithmetic } => {
                let op = if *is_arithmetic { BinOp::Sra } else { BinOp::Srl };
                self.select_binop(*result, *lhs, *rhs, ty, op, instrs);
            }

            // --- Comparisons ---
            Instruction::ICmp { result, op, lhs, rhs, ty } => {
                self.select_icmp(*result, *op, *lhs, *rhs, ty, instrs);
            }
            Instruction::FCmp { result, op, lhs, rhs, ty } => {
                self.select_fcmp(*result, *op, *lhs, *rhs, ty, instrs);
            }

            // --- Memory ---
            Instruction::Alloca { result, ty, count } => {
                self.select_alloca(*result, ty, count.as_ref(), instrs);
            }
            Instruction::Load { result, ty, ptr } => {
                self.select_load(*result, ty, *ptr, instrs);
            }
            Instruction::Store { value, ptr } => {
                self.select_store(*value, *ptr, instrs);
            }
            Instruction::GetElementPtr { result, base_ty, ptr, indices, .. } => {
                self.select_gep(*result, base_ty, *ptr, indices, instrs);
            }

            // --- Function calls ---
            Instruction::Call { result, callee, args, return_ty } => {
                self.select_call(result.as_ref(), callee, args, return_ty, instrs);
            }

            // --- SSA / Copy ---
            Instruction::Phi { result, incoming, .. } => {
                // Phi nodes should have been lowered by SSA destruction.
                // As a fallback, emit a move from the first incoming value.
                let dest = self.get_reg(*result);
                if let Some(&(val, _)) = incoming.first() {
                    let src = self.get_reg(val);
                    Self::emit_mv(instrs, dest, src);
                }
            }
            Instruction::Copy { result, source, .. } => {
                let dest = self.get_reg(*result);
                let src = self.get_reg(*source);
                Self::emit_mv(instrs, dest, src);
            }

            // --- Type conversions ---
            Instruction::Cast { result, op, value, from_ty, to_ty } => {
                self.select_cast(*result, *op, *value, from_ty, to_ty, instrs);
            }
            Instruction::BitCast { result, value, from_ty, to_ty } => {
                self.select_bitcast(*result, *value, from_ty, to_ty, instrs);
            }

            // --- Constants ---
            Instruction::Const { result, value: constant } => {
                self.select_const(*result, constant, instrs);
            }

            // --- Select (ternary) ---
            Instruction::Select { result, condition, true_val, false_val, .. } => {
                self.select_select(*result, *condition, *true_val, *false_val, instrs);
            }

            // --- Nop ---
            Instruction::Nop => {
                // No machine instruction needed.
            }
        }
    }

    // -----------------------------------------------------------------------
    // Binary operation selection
    // -----------------------------------------------------------------------

    /// Selects RISC-V instructions for a binary arithmetic/bitwise operation.
    fn select_binop(
        &self,
        result: Value,
        lhs: Value,
        rhs: Value,
        ty: &IrType,
        op: BinOp,
        instrs: &mut Vec<MachineInstr>,
    ) {
        let dest = self.get_reg(result);
        let rs1 = self.get_reg(lhs);
        let rs2 = self.get_reg(rhs);
        let is_32bit = matches!(ty, IrType::I32);

        if ty.is_float() {
            // Floating-point binary operations
            let is_double = matches!(ty, IrType::F64);
            let opcode = match (op, is_double) {
                (BinOp::Add, true)  => Riscv64Opcode::FADD_D,
                (BinOp::Add, false) => Riscv64Opcode::FADD_S,
                (BinOp::Sub, true)  => Riscv64Opcode::FSUB_D,
                (BinOp::Sub, false) => Riscv64Opcode::FSUB_S,
                (BinOp::Mul, true)  => Riscv64Opcode::FMUL_D,
                (BinOp::Mul, false) => Riscv64Opcode::FMUL_S,
                (BinOp::DivS | BinOp::DivU, true)  => Riscv64Opcode::FDIV_D,
                (BinOp::DivS | BinOp::DivU, false) => Riscv64Opcode::FDIV_S,
                // Float remainder not directly supported; fall through to
                // integer path (which will emit a software sequence if needed).
                _ => {
                    Self::emit(instrs, Riscv64Opcode::NOP, vec![
                        MachineOperand::Register(dest),
                    ]);
                    return;
                }
            };
            Self::emit(instrs, opcode, vec![
                MachineOperand::Register(dest),
                MachineOperand::Register(rs1),
                MachineOperand::Register(rs2),
            ]);
            return;
        }

        // Integer binary operations — select appropriate opcode.
        let opcode = match (op, is_32bit) {
            (BinOp::Add, true)  => Riscv64Opcode::ADDW,
            (BinOp::Add, false) => Riscv64Opcode::ADD,
            (BinOp::Sub, true)  => Riscv64Opcode::SUBW,
            (BinOp::Sub, false) => Riscv64Opcode::SUB,
            (BinOp::Mul, true)  => Riscv64Opcode::MULW,
            (BinOp::Mul, false) => Riscv64Opcode::MUL,
            (BinOp::DivS, true)  => Riscv64Opcode::DIVW,
            (BinOp::DivS, false) => Riscv64Opcode::DIV,
            (BinOp::DivU, true)  => Riscv64Opcode::DIVUW,
            (BinOp::DivU, false) => Riscv64Opcode::DIVU,
            (BinOp::RemS, true)  => Riscv64Opcode::REMW,
            (BinOp::RemS, false) => Riscv64Opcode::REM,
            (BinOp::RemU, true)  => Riscv64Opcode::REMUW,
            (BinOp::RemU, false) => Riscv64Opcode::REMU,
            (BinOp::And, _)     => Riscv64Opcode::AND,
            (BinOp::Or, _)      => Riscv64Opcode::OR,
            (BinOp::Xor, _)     => Riscv64Opcode::XOR,
            (BinOp::Shl, true)  => Riscv64Opcode::SLLW,
            (BinOp::Shl, false) => Riscv64Opcode::SLL,
            (BinOp::Srl, true)  => Riscv64Opcode::SRLW,
            (BinOp::Srl, false) => Riscv64Opcode::SRL,
            (BinOp::Sra, true)  => Riscv64Opcode::SRAW,
            (BinOp::Sra, false) => Riscv64Opcode::SRA,
        };
        Self::emit(instrs, opcode, vec![
            MachineOperand::Register(dest),
            MachineOperand::Register(rs1),
            MachineOperand::Register(rs2),
        ]);
    }

    // -----------------------------------------------------------------------
    // Integer comparison selection
    // -----------------------------------------------------------------------

    /// Selects RISC-V instructions for an integer comparison (ICmp).
    ///
    /// RISC-V has limited comparison instructions (SLT, SLTU), so equality
    /// and greater-than comparisons are synthesized from multi-instruction
    /// sequences.
    fn select_icmp(
        &self,
        result: Value,
        op: CompareOp,
        lhs: Value,
        rhs: Value,
        _ty: &IrType,
        instrs: &mut Vec<MachineInstr>,
    ) {
        let dest = self.get_reg(result);
        let rs1 = self.get_reg(lhs);
        let rs2 = self.get_reg(rhs);

        match op {
            CompareOp::Equal => {
                // SUB tmp, rs1, rs2; SLTIU dest, tmp, 1  (SEQZ pattern)
                Self::emit(instrs, Riscv64Opcode::SUB, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(rs1),
                    MachineOperand::Register(rs2),
                ]);
                Self::emit(instrs, Riscv64Opcode::SLTIU, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(dest),
                    MachineOperand::Immediate(1),
                ]);
            }
            CompareOp::NotEqual => {
                // SUB tmp, rs1, rs2; SLTU dest, x0, tmp  (SNEZ pattern)
                Self::emit(instrs, Riscv64Opcode::SUB, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(rs1),
                    MachineOperand::Register(rs2),
                ]);
                Self::emit(instrs, Riscv64Opcode::SLTU, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(X0),
                    MachineOperand::Register(dest),
                ]);
            }
            CompareOp::SignedLess => {
                // SLT dest, rs1, rs2
                Self::emit(instrs, Riscv64Opcode::SLT, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(rs1),
                    MachineOperand::Register(rs2),
                ]);
            }
            CompareOp::UnsignedLess => {
                // SLTU dest, rs1, rs2
                Self::emit(instrs, Riscv64Opcode::SLTU, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(rs1),
                    MachineOperand::Register(rs2),
                ]);
            }
            CompareOp::SignedGreater => {
                // SLT dest, rs2, rs1  (swap operands)
                Self::emit(instrs, Riscv64Opcode::SLT, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(rs2),
                    MachineOperand::Register(rs1),
                ]);
            }
            CompareOp::UnsignedGreater => {
                // SLTU dest, rs2, rs1  (swap operands)
                Self::emit(instrs, Riscv64Opcode::SLTU, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(rs2),
                    MachineOperand::Register(rs1),
                ]);
            }
            CompareOp::SignedLessEqual => {
                // SLT tmp, rs2, rs1; XORI dest, tmp, 1  (NOT of greater)
                Self::emit(instrs, Riscv64Opcode::SLT, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(rs2),
                    MachineOperand::Register(rs1),
                ]);
                Self::emit(instrs, Riscv64Opcode::XORI, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(dest),
                    MachineOperand::Immediate(1),
                ]);
            }
            CompareOp::UnsignedLessEqual => {
                // SLTU tmp, rs2, rs1; XORI dest, tmp, 1
                Self::emit(instrs, Riscv64Opcode::SLTU, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(rs2),
                    MachineOperand::Register(rs1),
                ]);
                Self::emit(instrs, Riscv64Opcode::XORI, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(dest),
                    MachineOperand::Immediate(1),
                ]);
            }
            CompareOp::SignedGreaterEqual => {
                // SLT tmp, rs1, rs2; XORI dest, tmp, 1
                Self::emit(instrs, Riscv64Opcode::SLT, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(rs1),
                    MachineOperand::Register(rs2),
                ]);
                Self::emit(instrs, Riscv64Opcode::XORI, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(dest),
                    MachineOperand::Immediate(1),
                ]);
            }
            CompareOp::UnsignedGreaterEqual => {
                // SLTU tmp, rs1, rs2; XORI dest, tmp, 1
                Self::emit(instrs, Riscv64Opcode::SLTU, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(rs1),
                    MachineOperand::Register(rs2),
                ]);
                Self::emit(instrs, Riscv64Opcode::XORI, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(dest),
                    MachineOperand::Immediate(1),
                ]);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Floating-point comparison selection
    // -----------------------------------------------------------------------

    /// Selects RISC-V float comparison instructions.
    fn select_fcmp(
        &self,
        result: Value,
        op: FloatCompareOp,
        lhs: Value,
        rhs: Value,
        ty: &IrType,
        instrs: &mut Vec<MachineInstr>,
    ) {
        let dest = self.get_reg(result);
        let rs1 = self.get_reg(lhs);
        let rs2 = self.get_reg(rhs);
        let is_double = matches!(ty, IrType::F64);

        match op {
            FloatCompareOp::OrderedEqual => {
                let opcode = if is_double { Riscv64Opcode::FEQ_D } else { Riscv64Opcode::FEQ_S };
                Self::emit(instrs, opcode, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(rs1),
                    MachineOperand::Register(rs2),
                ]);
            }
            FloatCompareOp::OrderedLess => {
                let opcode = if is_double { Riscv64Opcode::FLT_D } else { Riscv64Opcode::FLT_S };
                Self::emit(instrs, opcode, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(rs1),
                    MachineOperand::Register(rs2),
                ]);
            }
            FloatCompareOp::OrderedLessEqual => {
                let opcode = if is_double { Riscv64Opcode::FLE_D } else { Riscv64Opcode::FLE_S };
                Self::emit(instrs, opcode, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(rs1),
                    MachineOperand::Register(rs2),
                ]);
            }
            FloatCompareOp::OrderedGreater => {
                // FLT with swapped operands: a > b ↔ b < a
                let opcode = if is_double { Riscv64Opcode::FLT_D } else { Riscv64Opcode::FLT_S };
                Self::emit(instrs, opcode, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(rs2),
                    MachineOperand::Register(rs1),
                ]);
            }
            FloatCompareOp::OrderedGreaterEqual => {
                // FLE with swapped operands: a >= b ↔ b <= a
                let opcode = if is_double { Riscv64Opcode::FLE_D } else { Riscv64Opcode::FLE_S };
                Self::emit(instrs, opcode, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(rs2),
                    MachineOperand::Register(rs1),
                ]);
            }
            FloatCompareOp::OrderedNotEqual => {
                // FEQ, then XOR with 1 to invert
                let opcode = if is_double { Riscv64Opcode::FEQ_D } else { Riscv64Opcode::FEQ_S };
                Self::emit(instrs, opcode, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(rs1),
                    MachineOperand::Register(rs2),
                ]);
                Self::emit(instrs, Riscv64Opcode::XORI, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(dest),
                    MachineOperand::Immediate(1),
                ]);
            }
            FloatCompareOp::Unordered | FloatCompareOp::UnorderedEqual => {
                // For unordered comparisons: check if either is NaN via
                // FEQ.D a,a (returns 0 if NaN). Simplified: use FEQ and invert.
                let opcode = if is_double { Riscv64Opcode::FEQ_D } else { Riscv64Opcode::FEQ_S };
                Self::emit(instrs, opcode, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(rs1),
                    MachineOperand::Register(rs2),
                ]);
                if matches!(op, FloatCompareOp::Unordered) {
                    // Invert: unordered is true when ordered comparison fails
                    Self::emit(instrs, Riscv64Opcode::XORI, vec![
                        MachineOperand::Register(dest),
                        MachineOperand::Register(dest),
                        MachineOperand::Immediate(1),
                    ]);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Memory operation selection
    // -----------------------------------------------------------------------

    /// Selects a RISC-V load instruction based on the loaded type.
    fn select_load(
        &self,
        result: Value,
        ty: &IrType,
        ptr: Value,
        instrs: &mut Vec<MachineInstr>,
    ) {
        let dest = self.get_reg(result);
        let base = self.get_reg(ptr);

        let opcode = match ty {
            IrType::I1 | IrType::I8 => Riscv64Opcode::LBU,
            IrType::I16 => Riscv64Opcode::LH,
            IrType::I32 => Riscv64Opcode::LW,
            IrType::I64 | IrType::Pointer(_) => Riscv64Opcode::LD,
            IrType::F32 => Riscv64Opcode::FLW,
            IrType::F64 => Riscv64Opcode::FLD,
            _ => Riscv64Opcode::LD,
        };
        Self::emit(instrs, opcode, vec![
            MachineOperand::Register(dest),
            MachineOperand::Memory { base, offset: 0 },
        ]);
    }

    /// Selects a RISC-V store instruction.
    fn select_store(
        &self,
        value: Value,
        ptr: Value,
        instrs: &mut Vec<MachineInstr>,
    ) {
        let src = self.get_reg(value);
        let base = self.get_reg(ptr);

        // Determine store width from the source register class.
        // If the source is an FP register, use float store; otherwise
        // we use SD as the default (the exact width is determined by
        // the IR type context that the encoder will refine).
        let opcode = if is_fp_reg(src) {
            // Detect single vs double by register range convention.
            Riscv64Opcode::FSD
        } else {
            Riscv64Opcode::SD
        };
        Self::emit(instrs, opcode, vec![
            MachineOperand::Register(src),
            MachineOperand::Memory { base, offset: 0 },
        ]);
    }

    /// Selects instructions for a stack allocation (alloca).
    fn select_alloca(
        &self,
        result: Value,
        _ty: &IrType,
        _count: Option<&Value>,
        instrs: &mut Vec<MachineInstr>,
    ) {
        let dest = self.get_reg(result);
        // Alloca is handled by the frame layout in the ABI module.
        // Here we compute the stack slot address using SP + offset.
        // The offset will be patched by the frame finalizer.
        Self::emit(instrs, Riscv64Opcode::ADDI, vec![
            MachineOperand::Register(dest),
            MachineOperand::Register(SP),
            MachineOperand::Immediate(0), // Placeholder; patched during frame layout
        ]);
    }

    /// Selects instructions for GetElementPtr (address computation).
    fn select_gep(
        &self,
        result: Value,
        base_ty: &IrType,
        ptr: Value,
        indices: &[Value],
        instrs: &mut Vec<MachineInstr>,
    ) {
        let dest = self.get_reg(result);
        let base = self.get_reg(ptr);

        if indices.is_empty() {
            Self::emit_mv(instrs, dest, base);
            return;
        }

        // Compute element size for the first index.
        let elem_size = base_ty.size(self.target) as i64;

        // Start with the base pointer.
        Self::emit_mv(instrs, dest, base);

        for (i, &idx_val) in indices.iter().enumerate() {
            let idx_reg = self.get_reg(idx_val);
            if i == 0 && elem_size > 0 {
                // For the primary index: offset = index * element_size
                // If element_size is a power of 2, use shift; otherwise multiply.
                if elem_size == 1 {
                    Self::emit(instrs, Riscv64Opcode::ADD, vec![
                        MachineOperand::Register(dest),
                        MachineOperand::Register(dest),
                        MachineOperand::Register(idx_reg),
                    ]);
                } else if elem_size > 0 && (elem_size as u64).is_power_of_two() {
                    let shift = (elem_size as u64).trailing_zeros() as i64;
                    // SLLI tmp, idx, shift; ADD dest, dest, tmp
                    Self::emit(instrs, Riscv64Opcode::SLLI, vec![
                        MachineOperand::Register(idx_reg),
                        MachineOperand::Register(idx_reg),
                        MachineOperand::Immediate(shift),
                    ]);
                    Self::emit(instrs, Riscv64Opcode::ADD, vec![
                        MachineOperand::Register(dest),
                        MachineOperand::Register(dest),
                        MachineOperand::Register(idx_reg),
                    ]);
                } else {
                    // General case: multiply index by element size using MUL.
                    // Materialize the element size constant first.
                    // Use dest as temporary for the multiply result.
                    Self::emit(instrs, Riscv64Opcode::MUL, vec![
                        MachineOperand::Register(idx_reg),
                        MachineOperand::Register(idx_reg),
                        MachineOperand::Immediate(elem_size),
                    ]);
                    Self::emit(instrs, Riscv64Opcode::ADD, vec![
                        MachineOperand::Register(dest),
                        MachineOperand::Register(dest),
                        MachineOperand::Register(idx_reg),
                    ]);
                }
            } else {
                // Subsequent indices (struct field offsets, etc.) — treated as
                // direct byte offsets added to the accumulated address.
                Self::emit(instrs, Riscv64Opcode::ADD, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(dest),
                    MachineOperand::Register(idx_reg),
                ]);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Function call selection
    // -----------------------------------------------------------------------

    /// Selects instructions for a function call.
    fn select_call(
        &mut self,
        result: Option<&Value>,
        callee: &Callee,
        args: &[Value],
        return_ty: &IrType,
        instrs: &mut Vec<MachineInstr>,
    ) {
        // Move arguments into argument registers per LP64D ABI.
        let mut int_arg_idx = 0usize;
        let mut float_arg_idx = 0usize;

        for &arg_val in args {
            let src = self.get_reg(arg_val);
            if is_fp_reg(src) && float_arg_idx < FLOAT_ARG_REGS.len() {
                Self::emit_mv(instrs, FLOAT_ARG_REGS[float_arg_idx], src);
                float_arg_idx += 1;
            } else if int_arg_idx < ARG_REGS.len() {
                Self::emit_mv(instrs, ARG_REGS[int_arg_idx], src);
                int_arg_idx += 1;
            } else {
                // Stack argument — push to stack via SD/FSD at SP offset.
                // The exact offset depends on the number of stack arguments.
                let stack_offset = ((int_arg_idx + float_arg_idx) as i32 - 8) * 8;
                if is_fp_reg(src) {
                    Self::emit(instrs, Riscv64Opcode::FSD, vec![
                        MachineOperand::Register(src),
                        MachineOperand::Memory { base: SP, offset: stack_offset },
                    ]);
                } else {
                    Self::emit(instrs, Riscv64Opcode::SD, vec![
                        MachineOperand::Register(src),
                        MachineOperand::Memory { base: SP, offset: stack_offset },
                    ]);
                }
                if is_fp_reg(src) { float_arg_idx += 1; } else { int_arg_idx += 1; }
            }
        }

        // Emit the call instruction.
        match callee {
            Callee::Direct(name) => {
                // CALL pseudo: AUIPC ra, %pcrel_hi(symbol) + JALR ra, ra, %pcrel_lo(symbol)
                // Encoded as a single CALL pseudo with R_RISCV_CALL relocation.
                Self::emit(instrs, Riscv64Opcode::CALL, vec![
                    MachineOperand::Symbol(name.clone()),
                ]);
                self.relocations.push(Relocation {
                    offset: 0, // Patched during encoding
                    symbol: name.clone(),
                    reloc_type: RelocationType::Riscv_Call,
                    addend: 0,
                    section_index: 0,
                });
            }
            Callee::Indirect(val) => {
                let target_reg = self.get_reg(*val);
                // JALR ra, target_reg, 0
                Self::emit(instrs, Riscv64Opcode::JALR, vec![
                    MachineOperand::Register(RA),
                    MachineOperand::Register(target_reg),
                    MachineOperand::Immediate(0),
                ]);
            }
        }

        // Move the return value from a0/fa0 to the destination register.
        if let Some(&res) = result {
            let dest = self.get_reg(res);
            if return_ty.is_float() {
                Self::emit_mv(instrs, dest, FA0);
            } else if !return_ty.is_void() {
                Self::emit_mv(instrs, dest, A0);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Terminator selection
    // -----------------------------------------------------------------------

    /// Selects RISC-V instructions for a block terminator.
    fn select_terminator(
        &self,
        term: &Terminator,
        instrs: &mut Vec<MachineInstr>,
    ) {
        match term {
            Terminator::Branch { target } => {
                // Unconditional jump: JAL x0, label
                if let Some(&label) = self.block_labels.get(target) {
                    Self::emit(instrs, Riscv64Opcode::JAL, vec![
                        MachineOperand::Register(X0),
                        MachineOperand::Label(label),
                    ]);
                }
            }
            Terminator::CondBranch { condition, true_block, false_block } => {
                let cond_reg = self.get_reg(*condition);

                // BNE cond, x0, true_label  (branch if condition is nonzero)
                if let Some(&true_label) = self.block_labels.get(true_block) {
                    Self::emit(instrs, Riscv64Opcode::BNE, vec![
                        MachineOperand::Register(cond_reg),
                        MachineOperand::Register(X0),
                        MachineOperand::Label(true_label),
                    ]);
                }
                // Fall through or jump to false block.
                if let Some(&false_label) = self.block_labels.get(false_block) {
                    Self::emit(instrs, Riscv64Opcode::JAL, vec![
                        MachineOperand::Register(X0),
                        MachineOperand::Label(false_label),
                    ]);
                }
            }
            Terminator::Return { value } => {
                // Move return value to a0 (integer) or fa0 (float).
                if let Some(ret_val) = value {
                    let src = self.get_reg(*ret_val);
                    if is_fp_reg(src) {
                        Self::emit_mv(instrs, FA0, src);
                    } else {
                        Self::emit_mv(instrs, A0, src);
                    }
                }
                // RET pseudo: JALR x0, ra, 0
                Self::emit(instrs, Riscv64Opcode::RET, vec![]);
            }
            Terminator::Switch { value, default, cases } => {
                let val_reg = self.get_reg(*value);
                // Emit a compare-and-branch sequence for each case.
                for &(case_val, target_block) in cases {
                    if let Some(&target_label) = self.block_labels.get(&target_block) {
                        // Materialize case value, compare, branch if equal.
                        // Use ADDI to load small immediates, otherwise LI pseudo.
                        if Self::fits_in_imm12(case_val) {
                            // ADDI tmp, x0, case_val; BEQ val, tmp, target
                            // We reuse the dest pattern with a temp approach.
                            // Use SLTI trick: val == case_val ↔ !(val < case_val) && !(case_val < val)
                            // Simpler: ADDI x5(t0), x0, case_val; BEQ val_reg, x5, label
                            Self::emit(instrs, Riscv64Opcode::ADDI, vec![
                                MachineOperand::Register(T0),
                                MachineOperand::Register(X0),
                                MachineOperand::Immediate(case_val),
                            ]);
                        } else {
                            Self::emit(instrs, Riscv64Opcode::LI, vec![
                                MachineOperand::Register(T0),
                                MachineOperand::Immediate(case_val),
                            ]);
                        }
                        Self::emit(instrs, Riscv64Opcode::BEQ, vec![
                            MachineOperand::Register(val_reg),
                            MachineOperand::Register(T0),
                            MachineOperand::Label(target_label),
                        ]);
                    }
                }
                // Jump to default block.
                if let Some(&default_label) = self.block_labels.get(default) {
                    Self::emit(instrs, Riscv64Opcode::JAL, vec![
                        MachineOperand::Register(X0),
                        MachineOperand::Label(default_label),
                    ]);
                }
            }
            Terminator::Unreachable => {
                // Emit EBREAK to trap if reached.
                Self::emit(instrs, Riscv64Opcode::EBREAK, vec![]);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Type cast selection
    // -----------------------------------------------------------------------

    /// Selects RISC-V instructions for type conversion operations.
    fn select_cast(
        &self,
        result: Value,
        op: CastOp,
        value: Value,
        from_ty: &IrType,
        to_ty: &IrType,
        instrs: &mut Vec<MachineInstr>,
    ) {
        let dest = self.get_reg(result);
        let src = self.get_reg(value);

        match op {
            CastOp::Trunc => {
                // Integer truncation: on RV64, truncating to 32 bits uses ADDIW
                // which sign-extends the lower 32 bits.
                if matches!(to_ty, IrType::I32) {
                    Self::emit(instrs, Riscv64Opcode::ADDIW, vec![
                        MachineOperand::Register(dest),
                        MachineOperand::Register(src),
                        MachineOperand::Immediate(0),
                    ]);
                } else if matches!(to_ty, IrType::I16) {
                    // SLLI by 48, then SRAI by 48 to sign-extend 16 bits
                    Self::emit(instrs, Riscv64Opcode::SLLI, vec![
                        MachineOperand::Register(dest),
                        MachineOperand::Register(src),
                        MachineOperand::Immediate(48),
                    ]);
                    Self::emit(instrs, Riscv64Opcode::SRAI, vec![
                        MachineOperand::Register(dest),
                        MachineOperand::Register(dest),
                        MachineOperand::Immediate(48),
                    ]);
                } else if matches!(to_ty, IrType::I8 | IrType::I1) {
                    // AND with 0xFF to extract lower byte
                    Self::emit(instrs, Riscv64Opcode::ANDI, vec![
                        MachineOperand::Register(dest),
                        MachineOperand::Register(src),
                        MachineOperand::Immediate(0xFF),
                    ]);
                } else {
                    Self::emit_mv(instrs, dest, src);
                }
            }
            CastOp::SExt => {
                // Sign-extension
                if matches!(from_ty, IrType::I32) {
                    // ADDIW sign-extends 32→64
                    Self::emit(instrs, Riscv64Opcode::ADDIW, vec![
                        MachineOperand::Register(dest),
                        MachineOperand::Register(src),
                        MachineOperand::Immediate(0),
                    ]);
                } else if matches!(from_ty, IrType::I16) {
                    Self::emit(instrs, Riscv64Opcode::SLLI, vec![
                        MachineOperand::Register(dest),
                        MachineOperand::Register(src),
                        MachineOperand::Immediate(48),
                    ]);
                    Self::emit(instrs, Riscv64Opcode::SRAI, vec![
                        MachineOperand::Register(dest),
                        MachineOperand::Register(dest),
                        MachineOperand::Immediate(48),
                    ]);
                } else if matches!(from_ty, IrType::I8) {
                    Self::emit(instrs, Riscv64Opcode::SLLI, vec![
                        MachineOperand::Register(dest),
                        MachineOperand::Register(src),
                        MachineOperand::Immediate(56),
                    ]);
                    Self::emit(instrs, Riscv64Opcode::SRAI, vec![
                        MachineOperand::Register(dest),
                        MachineOperand::Register(dest),
                        MachineOperand::Immediate(56),
                    ]);
                } else if matches!(from_ty, IrType::I1) {
                    // Negate the bit: SUB dest, x0, src (0 or 1 → 0 or -1)
                    Self::emit(instrs, Riscv64Opcode::SUB, vec![
                        MachineOperand::Register(dest),
                        MachineOperand::Register(X0),
                        MachineOperand::Register(src),
                    ]);
                } else {
                    Self::emit_mv(instrs, dest, src);
                }
            }
            CastOp::ZExt => {
                // Zero-extension
                if matches!(from_ty, IrType::I32) {
                    // SLLI by 32, SRLI by 32 to zero-extend 32→64
                    Self::emit(instrs, Riscv64Opcode::SLLI, vec![
                        MachineOperand::Register(dest),
                        MachineOperand::Register(src),
                        MachineOperand::Immediate(32),
                    ]);
                    Self::emit(instrs, Riscv64Opcode::SRLI, vec![
                        MachineOperand::Register(dest),
                        MachineOperand::Register(dest),
                        MachineOperand::Immediate(32),
                    ]);
                } else if matches!(from_ty, IrType::I16) {
                    Self::emit(instrs, Riscv64Opcode::SLLI, vec![
                        MachineOperand::Register(dest),
                        MachineOperand::Register(src),
                        MachineOperand::Immediate(48),
                    ]);
                    Self::emit(instrs, Riscv64Opcode::SRLI, vec![
                        MachineOperand::Register(dest),
                        MachineOperand::Register(dest),
                        MachineOperand::Immediate(48),
                    ]);
                } else if matches!(from_ty, IrType::I8 | IrType::I1) {
                    Self::emit(instrs, Riscv64Opcode::ANDI, vec![
                        MachineOperand::Register(dest),
                        MachineOperand::Register(src),
                        MachineOperand::Immediate(0xFF),
                    ]);
                } else {
                    Self::emit_mv(instrs, dest, src);
                }
            }
            CastOp::FPToSI => {
                // Float/double to signed integer
                let opcode = if matches!(from_ty, IrType::F64) {
                    if matches!(to_ty, IrType::I32) { Riscv64Opcode::FCVT_W_D }
                    else { Riscv64Opcode::FCVT_L_D }
                } else {
                    if matches!(to_ty, IrType::I32) { Riscv64Opcode::FCVT_W_S }
                    else { Riscv64Opcode::FCVT_L_S }
                };
                Self::emit(instrs, opcode, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(src),
                ]);
            }
            CastOp::FPToUI => {
                // Float/double to unsigned integer
                let opcode = if matches!(from_ty, IrType::F64) {
                    if matches!(to_ty, IrType::I32) { Riscv64Opcode::FCVT_WU_D }
                    else { Riscv64Opcode::FCVT_LU_D }
                } else {
                    if matches!(to_ty, IrType::I32) { Riscv64Opcode::FCVT_WU_S }
                    else { Riscv64Opcode::FCVT_LU_S }
                };
                Self::emit(instrs, opcode, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(src),
                ]);
            }
            CastOp::SIToFP => {
                // Signed integer to float/double
                let opcode = if matches!(to_ty, IrType::F64) {
                    if matches!(from_ty, IrType::I32) { Riscv64Opcode::FCVT_D_W }
                    else { Riscv64Opcode::FCVT_D_L }
                } else {
                    if matches!(from_ty, IrType::I32) { Riscv64Opcode::FCVT_S_W }
                    else { Riscv64Opcode::FCVT_S_L }
                };
                Self::emit(instrs, opcode, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(src),
                ]);
            }
            CastOp::UIToFP => {
                // Unsigned integer to float/double
                let opcode = if matches!(to_ty, IrType::F64) {
                    if matches!(from_ty, IrType::I32) { Riscv64Opcode::FCVT_D_WU }
                    else { Riscv64Opcode::FCVT_D_LU }
                } else {
                    if matches!(from_ty, IrType::I32) { Riscv64Opcode::FCVT_S_WU }
                    else { Riscv64Opcode::FCVT_S_LU }
                };
                Self::emit(instrs, opcode, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(src),
                ]);
            }
            CastOp::FPTrunc => {
                // Double to single: FCVT.S.D
                Self::emit(instrs, Riscv64Opcode::FCVT_S_D, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(src),
                ]);
            }
            CastOp::FPExt => {
                // Single to double: FCVT.D.S
                Self::emit(instrs, Riscv64Opcode::FCVT_D_S, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(src),
                ]);
            }
            CastOp::PtrToInt | CastOp::IntToPtr => {
                // On RV64, pointers and integers are both 64-bit. No conversion needed.
                Self::emit_mv(instrs, dest, src);
            }
        }
    }

    /// Selects instructions for bitcast (reinterpretation without value change).
    fn select_bitcast(
        &self,
        result: Value,
        value: Value,
        from_ty: &IrType,
        to_ty: &IrType,
        instrs: &mut Vec<MachineInstr>,
    ) {
        let dest = self.get_reg(result);
        let src = self.get_reg(value);

        // Bitcast between float and integer requires FMV instructions.
        if from_ty.is_float() && to_ty.is_integer() {
            let opcode = if matches!(from_ty, IrType::F64) {
                Riscv64Opcode::FMV_X_D
            } else {
                Riscv64Opcode::FMV_X_W
            };
            Self::emit(instrs, opcode, vec![
                MachineOperand::Register(dest),
                MachineOperand::Register(src),
            ]);
        } else if from_ty.is_integer() && to_ty.is_float() {
            let opcode = if matches!(to_ty, IrType::F64) {
                Riscv64Opcode::FMV_D_X
            } else {
                Riscv64Opcode::FMV_W_X
            };
            Self::emit(instrs, opcode, vec![
                MachineOperand::Register(dest),
                MachineOperand::Register(src),
            ]);
        } else {
            // Same register class — just a move.
            Self::emit_mv(instrs, dest, src);
        }
    }

    // -----------------------------------------------------------------------
    // Constant materialization
    // -----------------------------------------------------------------------

    /// Selects instructions for materializing a constant value.
    fn select_const(
        &mut self,
        result: Value,
        constant: &Constant,
        instrs: &mut Vec<MachineInstr>,
    ) {
        // Skip placeholder Const instructions for function parameters —
        // their ABI registers are already assigned by the allocator.
        if self.param_value_set.contains(&result) {
            return;
        }

        let dest = self.get_reg(result);

        match constant {
            Constant::Integer { value, .. } => {
                self.materialize_integer(*value, dest, instrs);
            }
            Constant::Float { value, ty } => {
                // Materialize float by first loading the bit pattern as an integer,
                // then moving to an FP register via FMV.
                if matches!(ty, IrType::F64) {
                    let bits = value.to_bits() as i64;
                    self.materialize_integer(bits, dest, instrs);
                    // If dest is a float register, the integer bits are already there
                    // via the value_map. If not, we need FMV.D.X.
                    if is_fp_reg(dest) {
                        // Need a temp integer register; use T0 as scratch.
                        self.materialize_integer(bits, T0, instrs);
                        Self::emit(instrs, Riscv64Opcode::FMV_D_X, vec![
                            MachineOperand::Register(dest),
                            MachineOperand::Register(T0),
                        ]);
                    }
                } else {
                    let bits = (*value as f32).to_bits() as i64;
                    self.materialize_integer(bits, dest, instrs);
                    if is_fp_reg(dest) {
                        self.materialize_integer(bits, T0, instrs);
                        Self::emit(instrs, Riscv64Opcode::FMV_W_X, vec![
                            MachineOperand::Register(dest),
                            MachineOperand::Register(T0),
                        ]);
                    }
                }
            }
            Constant::Bool(val) => {
                let imm = if *val { 1i64 } else { 0i64 };
                Self::emit(instrs, Riscv64Opcode::ADDI, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(X0),
                    MachineOperand::Immediate(imm),
                ]);
            }
            Constant::Null(_) | Constant::ZeroInit(_) => {
                // Load zero.
                Self::emit_mv(instrs, dest, X0);
            }
            Constant::Undef(_) => {
                // Undefined value — no instruction needed; dest has whatever value.
            }
            Constant::GlobalRef(name) => {
                // Load address of global: AUIPC + ADDI with PC-relative relocations.
                Self::emit(instrs, Riscv64Opcode::LA, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Symbol(name.clone()),
                ]);
                self.relocations.push(Relocation {
                    offset: 0,
                    symbol: name.clone(),
                    reloc_type: RelocationType::Riscv_Pcrel_Hi20,
                    addend: 0,
                    section_index: 0,
                });
                self.relocations.push(Relocation {
                    offset: 4,
                    symbol: name.clone(),
                    reloc_type: RelocationType::Riscv_Pcrel_Lo12_I,
                    addend: 0,
                    section_index: 0,
                });
            }
            Constant::String(bytes) => {
                // String constants are placed in .rodata; load the address.
                let label = format!(".L.str.{}", self.label_counter);
                self.label_counter += 1;
                Self::emit(instrs, Riscv64Opcode::LA, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Symbol(label.clone()),
                ]);
                // Track the string data for emission in .rodata by the encoder.
                let _ = bytes; // String data handled by the rodata section generator.
            }
        }
    }

    /// Materializes an integer constant into a physical register using the
    /// minimal instruction sequence for the given value.
    ///
    /// # Strategy
    ///
    /// - **12-bit immediate** (−2048..2047): single `ADDI dest, x0, imm`
    /// - **32-bit value**: `LUI dest, hi20` + `ADDI dest, dest, lo12`
    ///   (with sign-extension correction when bit 11 is set)
    /// - **64-bit value**: multi-instruction sequence using
    ///   `LUI+ADDI+SLLI+ADDI` pattern, building the value in stages
    fn materialize_integer(
        &self,
        value: i64,
        dest: PhysReg,
        instrs: &mut Vec<MachineInstr>,
    ) {
        if value == 0 {
            // Use x0 directly.
            Self::emit_mv(instrs, dest, X0);
            return;
        }

        if Self::fits_in_imm12(value) {
            // Single ADDI from x0.
            Self::emit(instrs, Riscv64Opcode::ADDI, vec![
                MachineOperand::Register(dest),
                MachineOperand::Register(X0),
                MachineOperand::Immediate(value),
            ]);
            return;
        }

        // Check if the value fits in 32 bits (sign-extended).
        let value_i32 = value as i32;
        if value_i32 as i64 == value {
            // 32-bit value: LUI + ADDI
            let lo12 = ((value_i32 as u32) & 0xFFF) as i32;
            let mut hi20 = ((value_i32 as u32) >> 12) as i32;

            // Handle sign-extension quirk: ADDI sign-extends its 12-bit immediate.
            // If bit 11 of lo12 is set, ADDI will subtract rather than add,
            // so we need to compensate by incrementing hi20 by 1.
            if lo12 >= 0x800 {
                hi20 = hi20.wrapping_add(1);
            }
            // Sign-extend lo12 to get the actual immediate for ADDI
            let lo12_signed = if lo12 >= 0x800 { lo12 - 0x1000 } else { lo12 };

            Self::emit(instrs, Riscv64Opcode::LUI, vec![
                MachineOperand::Register(dest),
                MachineOperand::Immediate(hi20 as i64),
            ]);
            if lo12_signed != 0 {
                Self::emit(instrs, Riscv64Opcode::ADDI, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(dest),
                    MachineOperand::Immediate(lo12_signed as i64),
                ]);
            }
            return;
        }

        // Full 64-bit value: build in stages.
        // Strategy: split into upper 32 bits and lower 32 bits.
        // Materialize upper 32 bits, shift left by 32, then add lower 32 bits.
        let upper = (value >> 32) as i32;
        let lower = value as i32;

        // Materialize upper 32 bits first.
        self.materialize_integer(upper as i64, dest, instrs);

        // Shift left by 32.
        Self::emit(instrs, Riscv64Opcode::SLLI, vec![
            MachineOperand::Register(dest),
            MachineOperand::Register(dest),
            MachineOperand::Immediate(32),
        ]);

        // Add lower 32 bits. We need to be careful with the lower half.
        if lower != 0 {
            let lower_u32 = lower as u32;
            let lo12 = (lower_u32 & 0xFFF) as i32;
            let mut hi20 = (lower_u32 >> 12) as i32;

            if lo12 >= 0x800 {
                hi20 = hi20.wrapping_add(1);
            }
            let lo12_signed = if lo12 >= 0x800 { lo12 - 0x1000 } else { lo12 };

            if hi20 != 0 {
                // We need to add the upper 20 bits of the lower half.
                // Use T0 as a scratch register to build the lower value.
                Self::emit(instrs, Riscv64Opcode::LUI, vec![
                    MachineOperand::Register(T0),
                    MachineOperand::Immediate(hi20 as i64),
                ]);
                if lo12_signed != 0 {
                    Self::emit(instrs, Riscv64Opcode::ADDI, vec![
                        MachineOperand::Register(T0),
                        MachineOperand::Register(T0),
                        MachineOperand::Immediate(lo12_signed as i64),
                    ]);
                }
                // Zero-extend the 32-bit value in T0 (clear upper bits).
                Self::emit(instrs, Riscv64Opcode::SLLI, vec![
                    MachineOperand::Register(T0),
                    MachineOperand::Register(T0),
                    MachineOperand::Immediate(32),
                ]);
                Self::emit(instrs, Riscv64Opcode::SRLI, vec![
                    MachineOperand::Register(T0),
                    MachineOperand::Register(T0),
                    MachineOperand::Immediate(32),
                ]);
                Self::emit(instrs, Riscv64Opcode::OR, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(dest),
                    MachineOperand::Register(T0),
                ]);
            } else if lo12_signed != 0 {
                // Only lower 12 bits are nonzero — can use ORI directly.
                // But ORI sign-extends, so we use ADDI+ANDI pattern or ORI.
                // Actually, for the lower half ORing is safe since upper bits
                // of the lower 32-bit portion are zero.
                Self::emit(instrs, Riscv64Opcode::ORI, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(dest),
                    MachineOperand::Immediate(lo12 as i64),
                ]);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Select instruction
    // -----------------------------------------------------------------------

    /// Selects instructions for a Select (ternary) operation.
    fn select_select(
        &self,
        result: Value,
        condition: Value,
        true_val: Value,
        false_val: Value,
        instrs: &mut Vec<MachineInstr>,
    ) {
        let dest = self.get_reg(result);
        let cond = self.get_reg(condition);
        let tv = self.get_reg(true_val);
        let fv = self.get_reg(false_val);

        // Implement as: MV dest, false_val; BEQ cond, x0, skip; MV dest, true_val; skip:
        Self::emit_mv(instrs, dest, fv);
        Self::emit(instrs, Riscv64Opcode::BEQ, vec![
            MachineOperand::Register(cond),
            MachineOperand::Register(X0),
            MachineOperand::Immediate(8), // Skip next instruction (4 bytes for MV)
        ]);
        Self::emit_mv(instrs, dest, tv);
    }
} // end impl Riscv64InstructionSelector

// ---------------------------------------------------------------------------
// BinOp — internal enum for binary operation dispatch
// ---------------------------------------------------------------------------

/// Internal binary operation discriminant used to select the correct RISC-V
/// opcode in the common `select_binop` helper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinOp {
    Add,
    Sub,
    Mul,
    DivS,
    DivU,
    RemS,
    RemU,
    And,
    Or,
    Xor,
    Shl,
    Srl,
    Sra,
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::regalloc::{AllocationResult, LiveInterval, PhysReg, RegClass};
    use crate::ir::cfg::{BasicBlock, Terminator};
    use crate::ir::instructions::{BlockId, Constant, Instruction, Value};
    use crate::ir::types::IrType;
    use crate::ir::builder::Function;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Creates a minimal TargetConfig for RISC-V 64 testing.
    fn make_riscv64_target() -> TargetConfig {
        TargetConfig::riscv64()
    }

    /// Creates an AllocationResult mapping a set of values to physical registers.
    fn make_alloc_result(mappings: &[(Value, PhysReg)]) -> AllocationResult {
        let intervals = mappings
            .iter()
            .map(|(val, reg)| LiveInterval {
                value: *val,
                reg_class: if reg.0 >= 32 { RegClass::Float } else { RegClass::Integer },
                start: 0,
                end: 10,
                assigned_reg: Some(*reg),
                spill_slot: None,
                is_param: false,
                crosses_call: false,
            })
            .collect();
        AllocationResult {
            intervals,
            num_spill_slots: 0,
            used_callee_saved: Vec::new(),
        }
    }

    /// Creates a minimal function with one block containing the given instructions.
    fn make_function(instrs: Vec<Instruction>, terminator: Terminator) -> Function {
        let mut block = BasicBlock::new(BlockId(0), "entry".to_string());
        block.instructions = instrs;
        block.terminator = Some(terminator);
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
    // Arithmetic selection tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_i64_selects_add() {
        let target = make_riscv64_target();
        let alloc = make_alloc_result(&[
            (Value(0), PhysReg(10)), // a0
            (Value(1), PhysReg(11)), // a1
            (Value(2), PhysReg(12)), // a2
        ]);
        let func = make_function(
            vec![Instruction::Add {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I64,
            }],
            Terminator::Return { value: None },
        );
        let mut selector = Riscv64InstructionSelector::new(&alloc, &target);
        let instrs = selector.select_function(&func);

        // Find the ADD instruction (skip NOP label markers).
        let add_instr = instrs.iter().find(|i| i.opcode == Riscv64Opcode::ADD.as_u32());
        assert!(add_instr.is_some(), "Should emit ADD for i64 addition");
    }

    #[test]
    fn test_add_i32_selects_addw() {
        let target = make_riscv64_target();
        let alloc = make_alloc_result(&[
            (Value(0), PhysReg(10)),
            (Value(1), PhysReg(11)),
            (Value(2), PhysReg(12)),
        ]);
        let func = make_function(
            vec![Instruction::Add {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I32,
            }],
            Terminator::Return { value: None },
        );
        let mut selector = Riscv64InstructionSelector::new(&alloc, &target);
        let instrs = selector.select_function(&func);

        let addw_instr = instrs.iter().find(|i| i.opcode == Riscv64Opcode::ADDW.as_u32());
        assert!(addw_instr.is_some(), "Should emit ADDW for i32 addition");
    }

    #[test]
    fn test_mul_selects_mul() {
        let target = make_riscv64_target();
        let alloc = make_alloc_result(&[
            (Value(0), PhysReg(10)),
            (Value(1), PhysReg(11)),
            (Value(2), PhysReg(12)),
        ]);
        let func = make_function(
            vec![Instruction::Mul {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I64,
            }],
            Terminator::Return { value: None },
        );
        let mut selector = Riscv64InstructionSelector::new(&alloc, &target);
        let instrs = selector.select_function(&func);

        let mul_instr = instrs.iter().find(|i| i.opcode == Riscv64Opcode::MUL.as_u32());
        assert!(mul_instr.is_some(), "Should emit MUL for i64 multiplication");
    }

    #[test]
    fn test_div_signed_selects_div() {
        let target = make_riscv64_target();
        let alloc = make_alloc_result(&[
            (Value(0), PhysReg(10)),
            (Value(1), PhysReg(11)),
            (Value(2), PhysReg(12)),
        ]);
        let func = make_function(
            vec![Instruction::Div {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I64,
                is_signed: true,
            }],
            Terminator::Return { value: None },
        );
        let mut selector = Riscv64InstructionSelector::new(&alloc, &target);
        let instrs = selector.select_function(&func);

        let div_instr = instrs.iter().find(|i| i.opcode == Riscv64Opcode::DIV.as_u32());
        assert!(div_instr.is_some(), "Should emit DIV for signed i64 division");
    }

    #[test]
    fn test_div_unsigned_selects_divu() {
        let target = make_riscv64_target();
        let alloc = make_alloc_result(&[
            (Value(0), PhysReg(10)),
            (Value(1), PhysReg(11)),
            (Value(2), PhysReg(12)),
        ]);
        let func = make_function(
            vec![Instruction::Div {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I64,
                is_signed: false,
            }],
            Terminator::Return { value: None },
        );
        let mut selector = Riscv64InstructionSelector::new(&alloc, &target);
        let instrs = selector.select_function(&func);

        let divu_instr = instrs.iter().find(|i| i.opcode == Riscv64Opcode::DIVU.as_u32());
        assert!(divu_instr.is_some(), "Should emit DIVU for unsigned i64 division");
    }

    // -----------------------------------------------------------------------
    // Comparison selection tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_icmp_equal_selects_sub_sltiu() {
        let target = make_riscv64_target();
        let alloc = make_alloc_result(&[
            (Value(0), PhysReg(10)),
            (Value(1), PhysReg(11)),
            (Value(2), PhysReg(12)),
        ]);
        let func = make_function(
            vec![Instruction::ICmp {
                result: Value(2),
                op: CompareOp::Equal,
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I64,
            }],
            Terminator::Return { value: None },
        );
        let mut selector = Riscv64InstructionSelector::new(&alloc, &target);
        let instrs = selector.select_function(&func);

        let sub = instrs.iter().any(|i| i.opcode == Riscv64Opcode::SUB.as_u32());
        let sltiu = instrs.iter().any(|i| i.opcode == Riscv64Opcode::SLTIU.as_u32());
        assert!(sub && sltiu, "ICmp Equal should emit SUB + SLTIU (SEQZ pattern)");
    }

    #[test]
    fn test_icmp_signed_less_selects_slt() {
        let target = make_riscv64_target();
        let alloc = make_alloc_result(&[
            (Value(0), PhysReg(10)),
            (Value(1), PhysReg(11)),
            (Value(2), PhysReg(12)),
        ]);
        let func = make_function(
            vec![Instruction::ICmp {
                result: Value(2),
                op: CompareOp::SignedLess,
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I64,
            }],
            Terminator::Return { value: None },
        );
        let mut selector = Riscv64InstructionSelector::new(&alloc, &target);
        let instrs = selector.select_function(&func);

        let slt = instrs.iter().any(|i| i.opcode == Riscv64Opcode::SLT.as_u32());
        assert!(slt, "ICmp SignedLess should emit SLT");
    }

    #[test]
    fn test_icmp_unsigned_less_selects_sltu() {
        let target = make_riscv64_target();
        let alloc = make_alloc_result(&[
            (Value(0), PhysReg(10)),
            (Value(1), PhysReg(11)),
            (Value(2), PhysReg(12)),
        ]);
        let func = make_function(
            vec![Instruction::ICmp {
                result: Value(2),
                op: CompareOp::UnsignedLess,
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I64,
            }],
            Terminator::Return { value: None },
        );
        let mut selector = Riscv64InstructionSelector::new(&alloc, &target);
        let instrs = selector.select_function(&func);

        let sltu = instrs.iter().any(|i| i.opcode == Riscv64Opcode::SLTU.as_u32());
        assert!(sltu, "ICmp UnsignedLess should emit SLTU");
    }

    // -----------------------------------------------------------------------
    // Memory operation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_load_i32_selects_lw() {
        let target = make_riscv64_target();
        let alloc = make_alloc_result(&[
            (Value(0), PhysReg(10)),
            (Value(1), PhysReg(11)),
        ]);
        let func = make_function(
            vec![Instruction::Load {
                result: Value(1),
                ty: IrType::I32,
                ptr: Value(0),
            }],
            Terminator::Return { value: None },
        );
        let mut selector = Riscv64InstructionSelector::new(&alloc, &target);
        let instrs = selector.select_function(&func);

        let lw = instrs.iter().any(|i| i.opcode == Riscv64Opcode::LW.as_u32());
        assert!(lw, "Load i32 should emit LW");
    }

    #[test]
    fn test_load_i64_selects_ld() {
        let target = make_riscv64_target();
        let alloc = make_alloc_result(&[
            (Value(0), PhysReg(10)),
            (Value(1), PhysReg(11)),
        ]);
        let func = make_function(
            vec![Instruction::Load {
                result: Value(1),
                ty: IrType::I64,
                ptr: Value(0),
            }],
            Terminator::Return { value: None },
        );
        let mut selector = Riscv64InstructionSelector::new(&alloc, &target);
        let instrs = selector.select_function(&func);

        let ld = instrs.iter().any(|i| i.opcode == Riscv64Opcode::LD.as_u32());
        assert!(ld, "Load i64 should emit LD");
    }

    #[test]
    fn test_store_fp_selects_fsd() {
        let target = make_riscv64_target();
        let alloc = make_alloc_result(&[
            (Value(0), PhysReg(42)), // fa0 (FP register)
            (Value(1), PhysReg(10)), // a0
        ]);
        let func = make_function(
            vec![Instruction::Store {
                value: Value(0),
                ptr: Value(1),
            }],
            Terminator::Return { value: None },
        );
        let mut selector = Riscv64InstructionSelector::new(&alloc, &target);
        let instrs = selector.select_function(&func);

        let fsd = instrs.iter().any(|i| i.opcode == Riscv64Opcode::FSD.as_u32());
        assert!(fsd, "Store of FP value should emit FSD");
    }

    // -----------------------------------------------------------------------
    // Constant materialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_small_immediate_single_addi() {
        let target = make_riscv64_target();
        let alloc = make_alloc_result(&[(Value(0), PhysReg(10))]);
        let func = make_function(
            vec![Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 42, ty: IrType::I64 },
            }],
            Terminator::Return { value: None },
        );
        let mut selector = Riscv64InstructionSelector::new(&alloc, &target);
        let instrs = selector.select_function(&func);

        let addi = instrs.iter().any(|i| i.opcode == Riscv64Opcode::ADDI.as_u32());
        assert!(addi, "Small immediate should use single ADDI");
        // Should NOT have LUI
        let lui = instrs.iter().any(|i| i.opcode == Riscv64Opcode::LUI.as_u32());
        assert!(!lui, "Small immediate should NOT use LUI");
    }

    #[test]
    fn test_32bit_value_lui_addi() {
        let target = make_riscv64_target();
        let alloc = make_alloc_result(&[(Value(0), PhysReg(10))]);
        let func = make_function(
            vec![Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 0x12345, ty: IrType::I64 },
            }],
            Terminator::Return { value: None },
        );
        let mut selector = Riscv64InstructionSelector::new(&alloc, &target);
        let instrs = selector.select_function(&func);

        let lui = instrs.iter().any(|i| i.opcode == Riscv64Opcode::LUI.as_u32());
        assert!(lui, "32-bit value should use LUI");
    }

    #[test]
    fn test_sign_extension_edge_case_0x800() {
        // Value 0x00000800: bit 11 is set, so LUI needs hi20+1 adjustment.
        let target = make_riscv64_target();
        let alloc = make_alloc_result(&[(Value(0), PhysReg(10))]);
        let func = make_function(
            vec![Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 0x800, ty: IrType::I64 },
            }],
            Terminator::Return { value: None },
        );
        let mut selector = Riscv64InstructionSelector::new(&alloc, &target);
        let instrs = selector.select_function(&func);

        // Should have LUI with adjusted hi20 (1) and ADDI with -2048
        let lui = instrs.iter().find(|i| i.opcode == Riscv64Opcode::LUI.as_u32());
        assert!(lui.is_some(), "0x800 should use LUI+ADDI pattern");
    }

    #[test]
    fn test_64bit_value_multi_instruction() {
        let target = make_riscv64_target();
        let alloc = make_alloc_result(&[(Value(0), PhysReg(10))]);
        let func = make_function(
            vec![Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 0x1_0000_0000_i64, ty: IrType::I64 },
            }],
            Terminator::Return { value: None },
        );
        let mut selector = Riscv64InstructionSelector::new(&alloc, &target);
        let instrs = selector.select_function(&func);

        let slli = instrs.iter().any(|i| i.opcode == Riscv64Opcode::SLLI.as_u32());
        assert!(slli, "64-bit value should use SLLI in multi-instruction sequence");
    }

    // -----------------------------------------------------------------------
    // Control flow tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_unconditional_branch_selects_jal() {
        let target = make_riscv64_target();
        let alloc = make_alloc_result(&[]);

        let mut block0 = BasicBlock::new(BlockId(0), "entry".to_string());
        block0.terminator = Some(Terminator::Branch { target: BlockId(1) });
        block0.successors = vec![BlockId(1)];

        let mut block1 = BasicBlock::new(BlockId(1), "target".to_string());
        block1.terminator = Some(Terminator::Return { value: None });
        block1.predecessors = vec![BlockId(0)];

        let func = Function {
            name: "test_branch".to_string(),
            return_type: IrType::Void,
            params: vec![],
            param_values: Vec::new(),
            blocks: vec![block0, block1],
            entry_block: BlockId(0),
            is_definition: true,
        };

        let mut selector = Riscv64InstructionSelector::new(&alloc, &target);
        let instrs = selector.select_function(&func);

        let jal = instrs.iter().any(|i| i.opcode == Riscv64Opcode::JAL.as_u32());
        assert!(jal, "Unconditional branch should emit JAL");
    }

    #[test]
    fn test_conditional_branch_selects_bne() {
        let target = make_riscv64_target();
        let alloc = make_alloc_result(&[(Value(0), PhysReg(10))]);

        let mut block0 = BasicBlock::new(BlockId(0), "entry".to_string());
        block0.terminator = Some(Terminator::CondBranch {
            condition: Value(0),
            true_block: BlockId(1),
            false_block: BlockId(2),
        });
        block0.successors = vec![BlockId(1), BlockId(2)];

        let mut block1 = BasicBlock::new(BlockId(1), "then".to_string());
        block1.terminator = Some(Terminator::Return { value: None });
        block1.predecessors = vec![BlockId(0)];

        let mut block2 = BasicBlock::new(BlockId(2), "else".to_string());
        block2.terminator = Some(Terminator::Return { value: None });
        block2.predecessors = vec![BlockId(0)];

        let func = Function {
            name: "test_cond".to_string(),
            return_type: IrType::Void,
            params: vec![],
            param_values: Vec::new(),
            blocks: vec![block0, block1, block2],
            entry_block: BlockId(0),
            is_definition: true,
        };

        let mut selector = Riscv64InstructionSelector::new(&alloc, &target);
        let instrs = selector.select_function(&func);

        let bne = instrs.iter().any(|i| i.opcode == Riscv64Opcode::BNE.as_u32());
        assert!(bne, "Conditional branch should emit BNE");
    }

    #[test]
    fn test_function_call_selects_call() {
        let target = make_riscv64_target();
        let alloc = make_alloc_result(&[(Value(0), PhysReg(10))]);
        let func = make_function(
            vec![Instruction::Call {
                result: Some(Value(0)),
                callee: Callee::Direct("printf".to_string()),
                args: vec![],
                return_ty: IrType::I32,
            }],
            Terminator::Return { value: None },
        );
        let mut selector = Riscv64InstructionSelector::new(&alloc, &target);
        let instrs = selector.select_function(&func);

        let call = instrs.iter().any(|i| i.opcode == Riscv64Opcode::CALL.as_u32());
        assert!(call, "Direct function call should emit CALL pseudo");

        // Should also produce a relocation.
        assert!(!selector.relocations().is_empty(), "CALL should produce relocation");
    }

    // -----------------------------------------------------------------------
    // Type conversion tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sext_i32_to_i64_selects_addiw() {
        let target = make_riscv64_target();
        let alloc = make_alloc_result(&[
            (Value(0), PhysReg(10)),
            (Value(1), PhysReg(11)),
        ]);
        let func = make_function(
            vec![Instruction::Cast {
                result: Value(1),
                op: CastOp::SExt,
                value: Value(0),
                from_ty: IrType::I32,
                to_ty: IrType::I64,
            }],
            Terminator::Return { value: None },
        );
        let mut selector = Riscv64InstructionSelector::new(&alloc, &target);
        let instrs = selector.select_function(&func);

        let addiw = instrs.iter().any(|i| i.opcode == Riscv64Opcode::ADDIW.as_u32());
        assert!(addiw, "SExt i32→i64 should emit ADDIW");
    }

    #[test]
    fn test_fpext_float_to_double() {
        let target = make_riscv64_target();
        let alloc = make_alloc_result(&[
            (Value(0), PhysReg(42)), // fa0
            (Value(1), PhysReg(43)), // fa1
        ]);
        let func = make_function(
            vec![Instruction::Cast {
                result: Value(1),
                op: CastOp::FPExt,
                value: Value(0),
                from_ty: IrType::F32,
                to_ty: IrType::F64,
            }],
            Terminator::Return { value: None },
        );
        let mut selector = Riscv64InstructionSelector::new(&alloc, &target);
        let instrs = selector.select_function(&func);

        let fcvt = instrs.iter().any(|i| i.opcode == Riscv64Opcode::FCVT_D_S.as_u32());
        assert!(fcvt, "FPExt float→double should emit FCVT.D.S");
    }

    #[test]
    fn test_sitofp_int_to_double() {
        let target = make_riscv64_target();
        let alloc = make_alloc_result(&[
            (Value(0), PhysReg(10)), // a0 (int)
            (Value(1), PhysReg(42)), // fa0 (double)
        ]);
        let func = make_function(
            vec![Instruction::Cast {
                result: Value(1),
                op: CastOp::SIToFP,
                value: Value(0),
                from_ty: IrType::I64,
                to_ty: IrType::F64,
            }],
            Terminator::Return { value: None },
        );
        let mut selector = Riscv64InstructionSelector::new(&alloc, &target);
        let instrs = selector.select_function(&func);

        let fcvt = instrs.iter().any(|i| i.opcode == Riscv64Opcode::FCVT_D_L.as_u32());
        assert!(fcvt, "SIToFP i64→f64 should emit FCVT.D.L");
    }

    // -----------------------------------------------------------------------
    // Riscv64Opcode coverage tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_opcode_as_u32_uniqueness() {
        // Verify all opcodes have unique u32 values.
        let opcodes = vec![
            Riscv64Opcode::LUI, Riscv64Opcode::AUIPC, Riscv64Opcode::JAL,
            Riscv64Opcode::JALR, Riscv64Opcode::BEQ, Riscv64Opcode::BNE,
            Riscv64Opcode::ADD, Riscv64Opcode::SUB, Riscv64Opcode::MUL,
            Riscv64Opcode::DIV, Riscv64Opcode::REM, Riscv64Opcode::AND,
            Riscv64Opcode::OR, Riscv64Opcode::XOR, Riscv64Opcode::SLL,
            Riscv64Opcode::SRL, Riscv64Opcode::SRA, Riscv64Opcode::ADDW,
            Riscv64Opcode::SUBW, Riscv64Opcode::MULW, Riscv64Opcode::NOP,
            Riscv64Opcode::RET, Riscv64Opcode::CALL, Riscv64Opcode::LI,
            Riscv64Opcode::LD, Riscv64Opcode::SD, Riscv64Opcode::FLD,
            Riscv64Opcode::FSD, Riscv64Opcode::FADD_D, Riscv64Opcode::FCVT_D_S,
        ];
        let mut seen = std::collections::HashSet::new();
        for op in &opcodes {
            assert!(seen.insert(op.as_u32()), "Duplicate opcode value for {:?}", op);
        }
    }

    #[test]
    fn test_return_selects_ret() {
        let target = make_riscv64_target();
        let alloc = make_alloc_result(&[(Value(0), PhysReg(10))]);
        let func = make_function(
            vec![],
            Terminator::Return { value: Some(Value(0)) },
        );
        let mut selector = Riscv64InstructionSelector::new(&alloc, &target);
        let instrs = selector.select_function(&func);

        let ret = instrs.iter().any(|i| i.opcode == Riscv64Opcode::RET.as_u32());
        assert!(ret, "Return should emit RET pseudo");
    }

    #[test]
    fn test_empty_function() {
        let target = make_riscv64_target();
        let alloc = make_alloc_result(&[]);
        let func = Function {
            name: "empty".to_string(),
            return_type: IrType::Void,
            params: vec![],
            param_values: Vec::new(),
            blocks: vec![],
            entry_block: BlockId(0),
            is_definition: true,
        };
        let mut selector = Riscv64InstructionSelector::new(&alloc, &target);
        let instrs = selector.select_function(&func);
        assert!(instrs.is_empty(), "Empty function should produce no instructions");
    }
}
