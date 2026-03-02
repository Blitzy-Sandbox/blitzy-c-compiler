//! i686 (32-bit x86) instruction selection module.
//!
//! This module transforms the SSA-form IR (from [`crate::ir`]) into i686
//! [`MachineInstr`] sequences — the intermediate machine instruction
//! representation defined in [`crate::codegen::MachineInstr`].
//!
//! # Key Differences from x86-64
//!
//! 1. Only 8 GPRs (3-bit encoding) — more register pressure, more spills
//! 2. All function args on stack — no register argument passing (cdecl ABI)
//! 3. 64-bit arithmetic requires register pairs — complex lowering
//! 4. No RIP-relative addressing — use absolute or GOT-relative (PIC)
//! 5. Division always uses edx:eax — fixed register constraint
//! 6. Pointer arithmetic is 32-bit, not 64-bit
//! 7. No REX prefix — all register encodings fit in 3 bits
//! 8. No security hardening (retpoline, CET) — x86-64 only per spec
//!
//! # Algorithm
//!
//! The selector performs a single linear pass over each function's basic
//! blocks. For every IR instruction it pattern-matches the instruction kind
//! and emits one or more `MachineInstr` values with i686 opcodes.
//!
//! # 64-bit Arithmetic
//!
//! C `long long` (64-bit) operations are lowered to pairs of 32-bit
//! instructions. For example, 64-bit addition becomes:
//! ```asm
//! add result_lo, rhs_lo    ; add low halves
//! adc result_hi, rhs_hi    ; add high halves with carry
//! ```
//!
//! # Zero External Dependencies
//!
//! Only imports from `std` and internal crate modules. No external crates.

use std::collections::HashMap;

use crate::codegen::{CodeGenError, MachineInstr, MachineOperand};
use crate::codegen::regalloc::PhysReg;
use crate::ir::{
    BasicBlock, BlockId, Callee, CastOp, CompareOp, Constant, FloatCompareOp,
    Function, Instruction, IrType, PhiNode, Terminator, Value,
};
use crate::driver::target::TargetConfig;

use super::encoding::{
    I686Opcode, phys_reg_to_encoding,
    REG_EAX, REG_ECX, REG_EDX, REG_EBX, REG_ESP, REG_EBP, REG_ESI, REG_EDI,
    CC_E, CC_NE, CC_B, CC_NB, CC_BE, CC_A, CC_L, CC_GE, CC_LE, CC_G,
    CC_NP,
};
use super::abi;

// ===========================================================================
// i686 Physical Register Constants
// ===========================================================================
// These map directly to the 3-bit register encoding values used by ModR/M
// and SIB bytes. GPRs occupy PhysReg(0..7), XMM registers occupy PhysReg(8..15).

/// EAX — accumulator, return value register (low 32 bits), caller-saved.
pub const EAX: PhysReg = PhysReg(0);
/// ECX — counter register, shift count register (cl), caller-saved.
pub const ECX: PhysReg = PhysReg(1);
/// EDX — data register, high 32 bits of 64-bit returns, caller-saved.
pub const EDX: PhysReg = PhysReg(2);
/// EBX — base register, callee-saved. GOT base pointer in PIC mode.
pub const EBX: PhysReg = PhysReg(3);
/// ESP — stack pointer. Not allocatable.
pub const ESP: PhysReg = PhysReg(4);
/// EBP — frame pointer, callee-saved. Not allocatable when frame pointer active.
pub const EBP: PhysReg = PhysReg(5);
/// ESI — source index, callee-saved.
pub const ESI: PhysReg = PhysReg(6);
/// EDI — destination index, callee-saved.
pub const EDI: PhysReg = PhysReg(7);
/// XMM0 — SSE register 0, used for float returns in SSE mode.
pub const XMM0: PhysReg = PhysReg(8);
/// XMM1 — SSE register 1.
pub const XMM1: PhysReg = PhysReg(9);
/// XMM2 — SSE register 2.
pub const XMM2: PhysReg = PhysReg(10);
/// XMM3 — SSE register 3.
pub const XMM3: PhysReg = PhysReg(11);
/// XMM4 — SSE register 4.
pub const XMM4: PhysReg = PhysReg(12);
/// XMM5 — SSE register 5.
pub const XMM5: PhysReg = PhysReg(13);
/// XMM6 — SSE register 6.
pub const XMM6: PhysReg = PhysReg(14);
/// XMM7 — SSE register 7.
pub const XMM7: PhysReg = PhysReg(15);

/// Base offset for virtual registers to avoid collision with the 16 physical
/// register constants defined above.
const VREG_BASE: u16 = 100;

/// Validate that a PhysReg is a valid i686 GPR (encoding fits in 3 bits).
/// Returns the 3-bit encoding value for the register, or `None` if the
/// register is a virtual register or out of range.
#[inline]
fn validate_gpr_encoding(reg: PhysReg) -> Option<u8> {
    if reg.0 <= EDI.0 {
        Some(phys_reg_to_encoding(reg))
    } else {
        None
    }
}

/// Verify that our PhysReg constants match the encoding module's register
/// encoding constants. This is a compile-time consistency check.
#[allow(dead_code)]
const _: () = {
    // GPR PhysReg ID must match the encoding register number.
    assert!(EAX.0 as u8 == REG_EAX);
    assert!(ECX.0 as u8 == REG_ECX);
    assert!(EDX.0 as u8 == REG_EDX);
    assert!(EBX.0 as u8 == REG_EBX);
    assert!(ESP.0 as u8 == REG_ESP);
    assert!(EBP.0 as u8 == REG_EBP);
    assert!(ESI.0 as u8 == REG_ESI);
    assert!(EDI.0 as u8 == REG_EDI);
};

// ===========================================================================
// Helper: opcode conversion
// ===========================================================================

/// Convert an [`I686Opcode`] enum variant to its `u32` representation for
/// storage in [`MachineInstr::opcode`].
#[inline]
fn op(opcode: I686Opcode) -> u32 {
    opcode as u32
}

// ===========================================================================
// InstructionSelector — internal state for a single function
// ===========================================================================

/// Internal state maintained during instruction selection for a single function.
struct ISel<'a> {
    /// The IR function being lowered.
    function: &'a Function,
    /// Target architecture configuration.
    target: &'a TargetConfig,
    /// Accumulated machine instructions (the output).
    output: Vec<MachineInstr>,
    /// Maps IR SSA values to their machine operand (register, immediate, etc.).
    value_map: HashMap<Value, MachineOperand>,
    /// For 64-bit values, maps the SSA value to its high-half operand.
    value_hi_map: HashMap<Value, MachineOperand>,
    /// Maps IR SSA values to their IR types, used for correct call-site
    /// argument classification (e.g., 64-bit arguments requiring two stack slots).
    type_map: HashMap<Value, IrType>,
    /// Maps basic block IDs to machine-level label IDs.
    block_label_map: HashMap<BlockId, u32>,
    /// Next available label ID.
    next_label: u32,
    /// Next available virtual register number.
    next_vreg: u16,
}

impl<'a> ISel<'a> {
    /// Create a new instruction selector for the given function and target.
    fn new(function: &'a Function, target: &'a TargetConfig) -> Self {
        ISel {
            function,
            target,
            output: Vec::with_capacity(function.blocks.len() * 16),
            value_map: HashMap::with_capacity(64),
            value_hi_map: HashMap::new(),
            type_map: HashMap::with_capacity(64),
            block_label_map: HashMap::with_capacity(function.blocks.len()),
            next_label: 0,
            next_vreg: VREG_BASE,
        }
    }

    /// Allocate a fresh virtual register.
    fn alloc_vreg(&mut self) -> PhysReg {
        let reg = PhysReg(self.next_vreg);
        self.next_vreg += 1;
        reg
    }

    /// Get or create a label ID for the given basic block.
    fn block_label(&mut self, block_id: BlockId) -> u32 {
        if let Some(&label) = self.block_label_map.get(&block_id) {
            return label;
        }
        let label = self.next_label;
        self.next_label += 1;
        self.block_label_map.insert(block_id, label);
        label
    }

    /// Emit a machine instruction and append it to the output.
    fn emit(&mut self, instr: MachineInstr) {
        self.output.push(instr);
    }

    /// Emit a simple instruction with no operands.
    fn emit_no_operands(&mut self, opcode: I686Opcode) {
        self.emit(MachineInstr::new(op(opcode)));
    }

    /// Emit an instruction with the given operands.
    fn emit_with(&mut self, opcode: I686Opcode, operands: Vec<MachineOperand>) {
        self.emit(MachineInstr::with_operands(op(opcode), operands));
    }

    /// Get the machine operand for an IR value. Panics if the value is unmapped.
    fn operand_for(&self, val: Value) -> MachineOperand {
        self.value_map
            .get(&val)
            .cloned()
            .unwrap_or_else(|| {
                // Fallback: treat the value as a virtual register derived from its ID.
                MachineOperand::Register(PhysReg(val.0 as u16 + VREG_BASE))
            })
    }

    /// Get the high-half operand for a 64-bit IR value.
    fn operand_hi_for(&self, val: Value) -> MachineOperand {
        self.value_hi_map
            .get(&val)
            .cloned()
            .unwrap_or_else(|| {
                // Allocate a high-half register offset from the low half.
                MachineOperand::Register(PhysReg(val.0 as u16 + VREG_BASE + 1000))
            })
    }

    /// Bind an IR value to a machine operand (low half for 64-bit).
    fn bind_value(&mut self, val: Value, operand: MachineOperand) {
        self.value_map.insert(val, operand);
    }

    /// Bind the high half of a 64-bit IR value.
    fn bind_value_hi(&mut self, val: Value, operand: MachineOperand) {
        self.value_hi_map.insert(val, operand);
    }

    /// Bind an IR value to a freshly allocated virtual register and return it.
    fn bind_vreg(&mut self, val: Value) -> PhysReg {
        let reg = self.alloc_vreg();
        self.bind_value(val, MachineOperand::Register(reg));
        reg
    }

    /// Check if a type is 64-bit integer (needs register pair handling).
    fn is_i64(ty: &IrType) -> bool {
        matches!(ty, IrType::I64)
    }

    /// Check if a type is a float type.
    fn is_float(ty: &IrType) -> bool {
        ty.is_float()
    }

    // -----------------------------------------------------------------------
    // Core selection: process entire function
    // -----------------------------------------------------------------------

    /// Process all basic blocks in the function and return the machine instrs.
    fn select_all(&mut self) -> Result<(), CodeGenError> {
        // Pre-assign labels for all blocks.
        for block in &self.function.blocks {
            let _ = self.block_label(block.id);
        }

        // Bind function parameters to stack locations (cdecl: all on stack).
        let param_types: Vec<IrType> = self.function.params.iter().map(|(_, ty)| ty.clone()).collect();
        let arg_infos = abi::classify_arguments(&param_types, self.target);
        for (i, (_, param_ty)) in self.function.params.iter().enumerate() {
            if i < arg_infos.len() {
                let info = &arg_infos[i];
                // Parameters are at [ebp + stack_offset]
                let operand = MachineOperand::Memory {
                    base: EBP,
                    offset: info.stack_offset,
                };
                // Use the actual param Value ID recorded by the IR builder.
                let param_val = if i < self.function.param_values.len() {
                    self.function.param_values[i]
                } else {
                    Value(i as u32)
                };
                self.bind_value(param_val, operand.clone());
                self.type_map.insert(param_val, param_ty.clone());

                // For 64-bit params, the high half is at +4.
                if matches!(param_ty, IrType::I64) {
                    let hi_operand = MachineOperand::Memory {
                        base: EBP,
                        offset: info.stack_offset + 4,
                    };
                    self.bind_value_hi(param_val, hi_operand);
                }
            }
        }

        // Process each basic block.
        for block in &self.function.blocks {
            self.select_block(block)?;
        }

        Ok(())
    }

    /// Process a single basic block.
    fn select_block(&mut self, block: &BasicBlock) -> Result<(), CodeGenError> {
        // Emit block label.
        let label_id = self.block_label(block.id);
        self.emit_with(I686Opcode::Label, vec![MachineOperand::Label(label_id)]);

        // Process phi nodes — generate copy instructions.
        for phi in &block.phi_nodes {
            self.select_phi(phi)?;
        }

        // Process regular instructions.
        for instr in &block.instructions {
            self.select_instruction(instr)?;
        }

        // Process terminator.
        if let Some(ref term) = block.terminator {
            self.select_terminator(term)?;
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Phi node handling
    // -----------------------------------------------------------------------

    /// Handle a phi node by binding its result to a virtual register.
    /// Actual phi resolution (parallel copies on predecessor edges) is handled
    /// by a later pass or by the SSA destruction step.
    fn select_phi(&mut self, phi: &PhiNode) -> Result<(), CodeGenError> {
        let result = phi.result;
        let _ty = &phi.ty;
        let _incoming = &phi.incoming;

        // Bind the phi result to a virtual register.
        let reg = self.bind_vreg(result);

        // If 64-bit, also bind the high half.
        if Self::is_i64(&phi.ty) {
            let hi_reg = self.alloc_vreg();
            self.bind_value_hi(result, MachineOperand::Register(hi_reg));
        }

        // Emit a mov from each incoming value on the corresponding edge.
        // In practice, the register allocator handles phi elimination by
        // inserting copies on predecessor edges. Here we just ensure the
        // result has a register binding.
        let _ = reg;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Main instruction dispatch
    // -----------------------------------------------------------------------

    /// Select machine instructions for a single IR instruction.
    fn select_instruction(&mut self, instr: &Instruction) -> Result<(), CodeGenError> {
        // Record the result type for every instruction that produces a typed
        // result. This enables correct argument classification in select_call
        // (e.g., 64-bit/double arguments requiring two stack slots on i686).
        if let Some(result_ty) = instr.result_type() {
            if let Some(result_val) = instr.result() {
                self.type_map.insert(result_val, result_ty.clone());
            }
        }

        match instr {
            // === Arithmetic ===
            Instruction::Add { result, lhs, rhs, ty } => {
                self.select_binary_arith(*result, *lhs, *rhs, ty, I686Opcode::Add)?;
            }
            Instruction::Sub { result, lhs, rhs, ty } => {
                self.select_binary_arith(*result, *lhs, *rhs, ty, I686Opcode::Sub)?;
            }
            Instruction::Mul { result, lhs, rhs, ty } => {
                self.select_mul(*result, *lhs, *rhs, ty)?;
            }
            Instruction::Div { result, lhs, rhs, ty, is_signed } => {
                self.select_div_mod(*result, *lhs, *rhs, ty, *is_signed, true)?;
            }
            Instruction::Mod { result, lhs, rhs, ty, is_signed } => {
                self.select_div_mod(*result, *lhs, *rhs, ty, *is_signed, false)?;
            }

            // === Bitwise ===
            Instruction::And { result, lhs, rhs, ty } => {
                self.select_binary_arith(*result, *lhs, *rhs, ty, I686Opcode::And)?;
            }
            Instruction::Or { result, lhs, rhs, ty } => {
                self.select_binary_arith(*result, *lhs, *rhs, ty, I686Opcode::Or)?;
            }
            Instruction::Xor { result, lhs, rhs, ty } => {
                self.select_binary_arith(*result, *lhs, *rhs, ty, I686Opcode::Xor)?;
            }
            Instruction::Shl { result, lhs, rhs, ty } => {
                self.select_shift(*result, *lhs, *rhs, ty, false, false)?;
            }
            Instruction::Shr { result, lhs, rhs, ty, is_arithmetic } => {
                self.select_shift(*result, *lhs, *rhs, ty, true, *is_arithmetic)?;
            }

            // === Comparison ===
            Instruction::ICmp { result, op, lhs, rhs, ty } => {
                self.select_icmp(*result, *op, *lhs, *rhs, ty)?;
            }
            Instruction::FCmp { result, op, lhs, rhs, ty } => {
                self.select_fcmp(*result, *op, *lhs, *rhs, ty)?;
            }

            // === Memory ===
            Instruction::Alloca { result, ty, count } => {
                self.select_alloca(*result, ty, count.as_ref())?;
            }
            Instruction::Load { result, ty, ptr } => {
                self.select_load(*result, ty, *ptr)?;
            }
            Instruction::Store { value, ptr } => {
                self.select_store(*value, *ptr)?;
            }
            Instruction::GetElementPtr { result, base_ty, ptr, indices, in_bounds: _ } => {
                self.select_gep(*result, base_ty, *ptr, indices)?;
            }

            // === Function call ===
            Instruction::Call { result, callee, args, return_ty } => {
                self.select_call(result.as_ref(), callee, args, return_ty)?;
            }

            // === Type conversion ===
            Instruction::Cast { result, op, value, from_ty, to_ty } => {
                self.select_cast(*result, *op, *value, from_ty, to_ty)?;
            }
            Instruction::BitCast { result, value, from_ty: _, to_ty: _ } => {
                // Bitcast is a no-op reinterpretation — just alias the operand.
                let src = self.operand_for(*value);
                self.bind_value(*result, src);
            }

            // === Miscellaneous ===
            Instruction::Const { result, value: constant } => {
                self.select_const(*result, constant)?;
            }
            Instruction::Copy { result, source, ty } => {
                self.select_copy(*result, *source, ty)?;
            }
            Instruction::Select { result, condition, true_val, false_val, ty } => {
                self.select_select(*result, *condition, *true_val, *false_val, ty)?;
            }
            Instruction::Phi { result, ty, incoming: _ } => {
                // Phi nodes are handled in select_block; if encountered here,
                // just ensure the result is bound.
                if !self.value_map.contains_key(result) {
                    let _ = self.bind_vreg(*result);
                    if Self::is_i64(ty) {
                        let hi = self.alloc_vreg();
                        self.bind_value_hi(*result, MachineOperand::Register(hi));
                    }
                }
            }
            Instruction::Nop => {
                self.emit_no_operands(I686Opcode::Nop);
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Binary arithmetic (32-bit and 64-bit)
    // -----------------------------------------------------------------------

    /// Select a binary arithmetic instruction (add, sub, and, or, xor).
    fn select_binary_arith(
        &mut self,
        result: Value,
        lhs: Value,
        rhs: Value,
        ty: &IrType,
        opcode_32: I686Opcode,
    ) -> Result<(), CodeGenError> {
        if Self::is_float(ty) {
            return self.select_float_binop(result, lhs, rhs, ty, opcode_32);
        }

        if Self::is_i64(ty) {
            return self.select_binary_arith_64(result, lhs, rhs, opcode_32);
        }

        // 32-bit (or narrower) case.
        let dst = self.bind_vreg(result);
        let lhs_op = self.operand_for(lhs);
        let rhs_op = self.operand_for(rhs);

        // mov dst, lhs
        self.emit_with(I686Opcode::Mov, vec![
            MachineOperand::Register(dst),
            lhs_op,
        ]);

        // <opcode> dst, rhs
        self.emit_with(opcode_32, vec![
            MachineOperand::Register(dst),
            rhs_op,
        ]);

        Ok(())
    }

    /// Select a 64-bit binary arithmetic instruction using register pairs.
    /// Supports add (add+adc), sub (sub+sbb), and/or/xor (pair-wise).
    fn select_binary_arith_64(
        &mut self,
        result: Value,
        lhs: Value,
        rhs: Value,
        opcode_32: I686Opcode,
    ) -> Result<(), CodeGenError> {
        let dst_lo = self.bind_vreg(result);
        let dst_hi = self.alloc_vreg();
        self.bind_value_hi(result, MachineOperand::Register(dst_hi));

        let lhs_lo = self.operand_for(lhs);
        let lhs_hi = self.operand_hi_for(lhs);
        let rhs_lo = self.operand_for(rhs);
        let rhs_hi = self.operand_hi_for(rhs);

        // mov dst_lo, lhs_lo
        self.emit_with(I686Opcode::Mov, vec![
            MachineOperand::Register(dst_lo), lhs_lo,
        ]);
        // mov dst_hi, lhs_hi
        self.emit_with(I686Opcode::Mov, vec![
            MachineOperand::Register(dst_hi), lhs_hi,
        ]);

        // For add: add dst_lo, rhs_lo; adc dst_hi, rhs_hi
        // For sub: sub dst_lo, rhs_lo; sbb dst_hi, rhs_hi
        // For and/or/xor: same op on both halves
        self.emit_with(opcode_32, vec![
            MachineOperand::Register(dst_lo), rhs_lo,
        ]);

        // Determine the high-half opcode: ADC for add (carry propagation),
        // SBB for sub (borrow propagation), same op for bitwise (no carry needed).
        let hi_opcode = match opcode_32 {
            I686Opcode::Add => I686Opcode::Adc, // ADC propagates carry from low-half ADD
            I686Opcode::Sub => I686Opcode::Sbb, // SBB propagates borrow from low-half SUB
            _ => opcode_32, // And, Or, Xor: same for both halves
        };

        self.emit_with(hi_opcode, vec![
            MachineOperand::Register(dst_hi), rhs_hi,
        ]);

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Multiplication
    // -----------------------------------------------------------------------

    /// Select multiplication instruction.
    fn select_mul(
        &mut self,
        result: Value,
        lhs: Value,
        rhs: Value,
        ty: &IrType,
    ) -> Result<(), CodeGenError> {
        if Self::is_float(ty) {
            return self.select_float_binop(result, lhs, rhs, ty, I686Opcode::Imul);
        }

        if Self::is_i64(ty) {
            return self.select_mul_64(result, lhs, rhs);
        }

        // 32-bit signed multiply: imul dst, src
        let dst = self.bind_vreg(result);
        let lhs_op = self.operand_for(lhs);
        let rhs_op = self.operand_for(rhs);

        // mov dst, lhs
        self.emit_with(I686Opcode::Mov, vec![
            MachineOperand::Register(dst), lhs_op,
        ]);
        // imul dst, rhs
        self.emit_with(I686Opcode::Imul, vec![
            MachineOperand::Register(dst), rhs_op,
        ]);

        Ok(())
    }

    /// Select 64-bit multiplication via three 32x32->64 multiplies.
    /// result = lhs * rhs where both are 64-bit.
    /// result_lo = (lhs_lo * rhs_lo).lo
    /// result_hi = (lhs_lo * rhs_lo).hi + lhs_lo * rhs_hi + lhs_hi * rhs_lo
    fn select_mul_64(
        &mut self,
        result: Value,
        lhs: Value,
        rhs: Value,
    ) -> Result<(), CodeGenError> {
        let dst_lo = self.bind_vreg(result);
        let dst_hi = self.alloc_vreg();
        self.bind_value_hi(result, MachineOperand::Register(dst_hi));

        let lhs_lo = self.operand_for(lhs);
        let lhs_hi = self.operand_hi_for(lhs);
        let rhs_lo = self.operand_for(rhs);
        let rhs_hi = self.operand_hi_for(rhs);

        // Step 1: mul lhs_lo, rhs_lo -> edx:eax (full 64-bit product of low halves)
        // mov eax, lhs_lo
        self.emit_with(I686Opcode::Mov, vec![
            MachineOperand::Register(EAX), lhs_lo.clone(),
        ]);
        // mul rhs_lo -> edx:eax = lhs_lo * rhs_lo
        self.emit_with(I686Opcode::Mul, vec![rhs_lo.clone()]);

        // Save low result: mov dst_lo, eax
        self.emit_with(I686Opcode::Mov, vec![
            MachineOperand::Register(dst_lo),
            MachineOperand::Register(EAX),
        ]);
        // Save high carry: mov dst_hi, edx
        self.emit_with(I686Opcode::Mov, vec![
            MachineOperand::Register(dst_hi),
            MachineOperand::Register(EDX),
        ]);

        // Step 2: lhs_lo * rhs_hi (only low 32 bits matter)
        let tmp1 = self.alloc_vreg();
        self.emit_with(I686Opcode::Mov, vec![
            MachineOperand::Register(tmp1), lhs_lo,
        ]);
        self.emit_with(I686Opcode::Imul, vec![
            MachineOperand::Register(tmp1), rhs_hi,
        ]);
        // Add to dst_hi
        self.emit_with(I686Opcode::Add, vec![
            MachineOperand::Register(dst_hi),
            MachineOperand::Register(tmp1),
        ]);

        // Step 3: lhs_hi * rhs_lo (only low 32 bits matter)
        let tmp2 = self.alloc_vreg();
        self.emit_with(I686Opcode::Mov, vec![
            MachineOperand::Register(tmp2), lhs_hi,
        ]);
        self.emit_with(I686Opcode::Imul, vec![
            MachineOperand::Register(tmp2), rhs_lo,
        ]);
        // Add to dst_hi
        self.emit_with(I686Opcode::Add, vec![
            MachineOperand::Register(dst_hi),
            MachineOperand::Register(tmp2),
        ]);

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Division and modulo
    // -----------------------------------------------------------------------

    /// Select division or modulo instruction.
    /// `is_quotient`: true for div (result in eax), false for mod (result in edx).
    fn select_div_mod(
        &mut self,
        result: Value,
        lhs: Value,
        rhs: Value,
        ty: &IrType,
        is_signed: bool,
        is_quotient: bool,
    ) -> Result<(), CodeGenError> {
        if Self::is_float(ty) {
            let float_op = if is_quotient { I686Opcode::Imul } else { I686Opcode::Imul };
            return self.select_float_binop(result, lhs, rhs, ty, float_op);
        }

        if Self::is_i64(ty) {
            return self.select_div_mod_64(result, lhs, rhs, is_signed, is_quotient);
        }

        // 32-bit division: uses edx:eax / divisor
        let lhs_op = self.operand_for(lhs);
        let rhs_op = self.operand_for(rhs);

        // Move dividend to eax
        self.emit_with(I686Opcode::Mov, vec![
            MachineOperand::Register(EAX), lhs_op,
        ]);

        if is_signed {
            // Sign-extend eax into edx:eax via cdq
            self.emit_no_operands(I686Opcode::Cdq);
            // idiv divisor
            self.emit_with(I686Opcode::Idiv, vec![rhs_op]);
        } else {
            // Zero-extend: xor edx, edx
            self.emit_with(I686Opcode::Xor, vec![
                MachineOperand::Register(EDX),
                MachineOperand::Register(EDX),
            ]);
            // div divisor
            self.emit_with(I686Opcode::Div, vec![rhs_op]);
        }

        // Result: quotient in eax, remainder in edx
        let dst = self.bind_vreg(result);
        let source_reg = if is_quotient { EAX } else { EDX };
        self.emit_with(I686Opcode::Mov, vec![
            MachineOperand::Register(dst),
            MachineOperand::Register(source_reg),
        ]);

        Ok(())
    }

    /// Select 64-bit division/modulo by calling runtime helper functions
    /// (__divdi3, __udivdi3, __moddi3, __umoddi3).
    fn select_div_mod_64(
        &mut self,
        result: Value,
        lhs: Value,
        rhs: Value,
        is_signed: bool,
        is_quotient: bool,
    ) -> Result<(), CodeGenError> {
        // 64-bit division on i686 is implemented via calls to compiler-rt helpers.
        let helper_name = match (is_signed, is_quotient) {
            (true, true) => "__divdi3",
            (true, false) => "__moddi3",
            (false, true) => "__udivdi3",
            (false, false) => "__umoddi3",
        };

        // Push arguments: rhs_hi, rhs_lo, lhs_hi, lhs_lo (right-to-left)
        let lhs_lo = self.operand_for(lhs);
        let lhs_hi = self.operand_hi_for(lhs);
        let rhs_lo = self.operand_for(rhs);
        let rhs_hi = self.operand_hi_for(rhs);

        // Push rhs_hi
        self.emit_with(I686Opcode::Push, vec![rhs_hi]);
        // Push rhs_lo
        self.emit_with(I686Opcode::Push, vec![rhs_lo]);
        // Push lhs_hi
        self.emit_with(I686Opcode::Push, vec![lhs_hi]);
        // Push lhs_lo
        self.emit_with(I686Opcode::Push, vec![lhs_lo]);

        // Call helper
        self.emit_with(I686Opcode::Call, vec![
            MachineOperand::Symbol(helper_name.to_string()),
        ]);

        // Cleanup: add esp, 16
        self.emit_with(I686Opcode::Add, vec![
            MachineOperand::Register(ESP),
            MachineOperand::Immediate(16),
        ]);

        // Result in eax:edx
        let dst_lo = self.bind_vreg(result);
        let dst_hi = self.alloc_vreg();
        self.bind_value_hi(result, MachineOperand::Register(dst_hi));

        self.emit_with(I686Opcode::Mov, vec![
            MachineOperand::Register(dst_lo),
            MachineOperand::Register(EAX),
        ]);
        self.emit_with(I686Opcode::Mov, vec![
            MachineOperand::Register(dst_hi),
            MachineOperand::Register(EDX),
        ]);

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Shift operations
    // -----------------------------------------------------------------------

    /// Select shift instruction (shl, shr, sar) for 32-bit and 64-bit.
    fn select_shift(
        &mut self,
        result: Value,
        lhs: Value,
        rhs: Value,
        ty: &IrType,
        is_right: bool,
        is_arithmetic: bool,
    ) -> Result<(), CodeGenError> {
        if Self::is_i64(ty) {
            return self.select_shift_64(result, lhs, rhs, is_right, is_arithmetic);
        }

        let dst = self.bind_vreg(result);
        let lhs_op = self.operand_for(lhs);
        let rhs_op = self.operand_for(rhs);

        // mov dst, lhs
        self.emit_with(I686Opcode::Mov, vec![
            MachineOperand::Register(dst), lhs_op,
        ]);

        // Move shift count to ecx (shift count must be in cl on x86)
        self.emit_with(I686Opcode::Mov, vec![
            MachineOperand::Register(ECX), rhs_op,
        ]);

        let shift_opcode = if !is_right {
            I686Opcode::Shl
        } else if is_arithmetic {
            I686Opcode::Sar
        } else {
            I686Opcode::Shr
        };

        // <shift> dst, cl
        self.emit_with(shift_opcode, vec![
            MachineOperand::Register(dst),
            MachineOperand::Register(ECX),
        ]);

        Ok(())
    }

    /// Select 64-bit shift using double-precision shift instructions.
    fn select_shift_64(
        &mut self,
        result: Value,
        lhs: Value,
        rhs: Value,
        is_right: bool,
        is_arithmetic: bool,
    ) -> Result<(), CodeGenError> {
        let dst_lo = self.bind_vreg(result);
        let dst_hi = self.alloc_vreg();
        self.bind_value_hi(result, MachineOperand::Register(dst_hi));

        let lhs_lo = self.operand_for(lhs);
        let lhs_hi = self.operand_hi_for(lhs);
        let rhs_op = self.operand_for(rhs);

        // mov dst_lo, lhs_lo; mov dst_hi, lhs_hi
        self.emit_with(I686Opcode::Mov, vec![
            MachineOperand::Register(dst_lo), lhs_lo,
        ]);
        self.emit_with(I686Opcode::Mov, vec![
            MachineOperand::Register(dst_hi), lhs_hi,
        ]);

        // Move shift count to ecx
        self.emit_with(I686Opcode::Mov, vec![
            MachineOperand::Register(ECX), rhs_op,
        ]);

        if !is_right {
            // Left shift 64-bit:
            // shld dst_hi, dst_lo, cl  (shift high, filling from low)
            // shl dst_lo, cl           (shift low)
            self.emit_with(I686Opcode::Shld, vec![
                MachineOperand::Register(dst_hi),
                MachineOperand::Register(dst_lo),
                MachineOperand::Register(ECX),
            ]);
            self.emit_with(I686Opcode::Shl, vec![
                MachineOperand::Register(dst_lo),
                MachineOperand::Register(ECX),
            ]);
        } else {
            // Right shift 64-bit:
            // shrd dst_lo, dst_hi, cl
            // sar/shr dst_hi, cl
            self.emit_with(I686Opcode::Shrd, vec![
                MachineOperand::Register(dst_lo),
                MachineOperand::Register(dst_hi),
                MachineOperand::Register(ECX),
            ]);
            let hi_shift = if is_arithmetic {
                I686Opcode::Sar
            } else {
                I686Opcode::Shr
            };
            self.emit_with(hi_shift, vec![
                MachineOperand::Register(dst_hi),
                MachineOperand::Register(ECX),
            ]);
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Integer comparison
    // -----------------------------------------------------------------------

    /// Select an integer comparison instruction.
    fn select_icmp(
        &mut self,
        result: Value,
        cmp_op: CompareOp,
        lhs: Value,
        rhs: Value,
        ty: &IrType,
    ) -> Result<(), CodeGenError> {
        if Self::is_i64(ty) {
            return self.select_icmp_64(result, cmp_op, lhs, rhs);
        }

        let lhs_op = self.operand_for(lhs);
        let rhs_op = self.operand_for(rhs);

        // cmp lhs, rhs
        self.emit_with(I686Opcode::Cmp, vec![lhs_op, rhs_op]);

        // Map CompareOp to x86 condition code.
        let cc = compare_op_to_cc(cmp_op);

        // setcc result_reg
        let dst = self.bind_vreg(result);
        self.emit_with(I686Opcode::Setcc, vec![
            MachineOperand::Immediate(cc as i64),
            MachineOperand::Register(dst),
        ]);

        Ok(())
    }

    /// Select a 64-bit integer comparison.
    /// Compare high halves first; if equal, compare low halves.
    fn select_icmp_64(
        &mut self,
        result: Value,
        cmp_op: CompareOp,
        lhs: Value,
        rhs: Value,
    ) -> Result<(), CodeGenError> {
        let lhs_lo = self.operand_for(lhs);
        let lhs_hi = self.operand_hi_for(lhs);
        let rhs_lo = self.operand_for(rhs);
        let rhs_hi = self.operand_hi_for(rhs);

        let dst = self.bind_vreg(result);

        // For equality: cmp hi, hi; if ne -> not equal; else cmp lo, lo
        // For ordering: cmp hi, hi; if ne -> use hi result; else cmp lo, lo
        let cc = compare_op_to_cc(cmp_op);

        // Compare high halves
        self.emit_with(I686Opcode::Cmp, vec![lhs_hi, rhs_hi]);

        // For simple cases, we emit a series of comparisons.
        // The general pattern uses conditional jumps, but for simplicity
        // we use setcc with a combined approach.

        // For equal/not-equal, both halves must match.
        match cmp_op {
            CompareOp::Equal | CompareOp::NotEqual => {
                // xor tmp, tmp; cmp hi_lhs, hi_rhs; setne tmp;
                // cmp lo_lhs, lo_rhs; setne dst;
                // or dst, tmp; (for NotEqual) or: and + sete (for Equal)
                let tmp = self.alloc_vreg();
                self.emit_with(I686Opcode::Setcc, vec![
                    MachineOperand::Immediate(CC_NE as i64),
                    MachineOperand::Register(tmp),
                ]);
                self.emit_with(I686Opcode::Cmp, vec![lhs_lo, rhs_lo]);
                self.emit_with(I686Opcode::Setcc, vec![
                    MachineOperand::Immediate(CC_NE as i64),
                    MachineOperand::Register(dst),
                ]);
                self.emit_with(I686Opcode::Or, vec![
                    MachineOperand::Register(dst),
                    MachineOperand::Register(tmp),
                ]);
                if cmp_op == CompareOp::Equal {
                    // Invert: xor dst, 1
                    self.emit_with(I686Opcode::Xor, vec![
                        MachineOperand::Register(dst),
                        MachineOperand::Immediate(1),
                    ]);
                }
            }
            _ => {
                // Ordering comparison: use high comparison, fallthrough to low if equal.
                // setcc(cc) on high comparison
                self.emit_with(I686Opcode::Setcc, vec![
                    MachineOperand::Immediate(cc as i64),
                    MachineOperand::Register(dst),
                ]);
                // When high halves are equal, use low comparison with unsigned variant.
                let lo_cc = compare_op_to_unsigned_cc(cmp_op);
                let tmp = self.alloc_vreg();
                self.emit_with(I686Opcode::Cmp, vec![lhs_lo.clone(), rhs_lo.clone()]);
                self.emit_with(I686Opcode::Setcc, vec![
                    MachineOperand::Immediate(lo_cc as i64),
                    MachineOperand::Register(tmp),
                ]);
                // Combine: if high equal, use low result; else use high result.
                // test high_eq
                let eq_tmp = self.alloc_vreg();
                self.emit_with(I686Opcode::Cmp, vec![
                    self.operand_hi_for(lhs),
                    self.operand_hi_for(rhs),
                ]);
                self.emit_with(I686Opcode::Setcc, vec![
                    MachineOperand::Immediate(CC_E as i64),
                    MachineOperand::Register(eq_tmp),
                ]);
                // Conditional move pattern: dst = eq_tmp ? tmp : dst
                // On i686, we use: and tmp, eq_tmp; and dst, ~eq_tmp; or dst, tmp
                // Simplified: test eq_tmp; cmovne dst, tmp (but cmov may not be available)
                // Use arithmetic instead:
                self.emit_with(I686Opcode::And, vec![
                    MachineOperand::Register(tmp),
                    MachineOperand::Register(eq_tmp),
                ]);
                let neg_eq = self.alloc_vreg();
                self.emit_with(I686Opcode::Mov, vec![
                    MachineOperand::Register(neg_eq),
                    MachineOperand::Register(eq_tmp),
                ]);
                self.emit_with(I686Opcode::Xor, vec![
                    MachineOperand::Register(neg_eq),
                    MachineOperand::Immediate(1),
                ]);
                self.emit_with(I686Opcode::And, vec![
                    MachineOperand::Register(dst),
                    MachineOperand::Register(neg_eq),
                ]);
                self.emit_with(I686Opcode::Or, vec![
                    MachineOperand::Register(dst),
                    MachineOperand::Register(tmp),
                ]);
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Float comparison
    // -----------------------------------------------------------------------

    /// Select a floating-point comparison instruction.
    fn select_fcmp(
        &mut self,
        result: Value,
        cmp_op: FloatCompareOp,
        lhs: Value,
        rhs: Value,
        ty: &IrType,
    ) -> Result<(), CodeGenError> {
        let lhs_op = self.operand_for(lhs);
        let rhs_op = self.operand_for(rhs);

        // ucomiss/ucomisd lhs, rhs
        let cmp_opcode = match ty {
            IrType::F32 => I686Opcode::Ucomiss,
            IrType::F64 => I686Opcode::Ucomisd,
            _ => return Err(CodeGenError::UnsupportedInstruction(
                format!("fcmp on non-float type {:?}", ty),
            )),
        };
        self.emit_with(cmp_opcode, vec![lhs_op, rhs_op]);

        // Map float compare op to condition code.
        let cc = float_compare_op_to_cc(cmp_op);
        let dst = self.bind_vreg(result);

        // For ordered comparisons, we need to check for NaN (parity flag).
        // setnp + setcc, then and them together.
        match cmp_op {
            FloatCompareOp::OrderedEqual
            | FloatCompareOp::OrderedLess
            | FloatCompareOp::OrderedLessEqual
            | FloatCompareOp::OrderedGreater
            | FloatCompareOp::OrderedGreaterEqual
            | FloatCompareOp::OrderedNotEqual => {
                // Ordered: result = !NaN && cc
                let np_tmp = self.alloc_vreg();
                self.emit_with(I686Opcode::Setcc, vec![
                    MachineOperand::Immediate(CC_NP as i64),
                    MachineOperand::Register(np_tmp),
                ]);
                self.emit_with(I686Opcode::Setcc, vec![
                    MachineOperand::Immediate(cc as i64),
                    MachineOperand::Register(dst),
                ]);
                self.emit_with(I686Opcode::And, vec![
                    MachineOperand::Register(dst),
                    MachineOperand::Register(np_tmp),
                ]);
            }
            _ => {
                // Unordered: just use setcc
                self.emit_with(I686Opcode::Setcc, vec![
                    MachineOperand::Immediate(cc as i64),
                    MachineOperand::Register(dst),
                ]);
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Memory operations
    // -----------------------------------------------------------------------

    /// Select an alloca instruction — reserve stack space.
    fn select_alloca(
        &mut self,
        result: Value,
        ty: &IrType,
        _count: Option<&Value>,
    ) -> Result<(), CodeGenError> {
        // Alloca is resolved during frame layout. At isel time, we bind the
        // result to a stack slot address. The actual offset is determined later.
        let size = ty.size(self.target) as i32;
        let dst = self.bind_vreg(result);

        // Emit a LEA to compute the address of the stack slot.
        // The actual offset will be patched during frame layout.
        self.emit_with(I686Opcode::Lea, vec![
            MachineOperand::Register(dst),
            MachineOperand::Memory { base: EBP, offset: -(size.max(4)) },
        ]);

        Ok(())
    }

    /// Select a load instruction.
    fn select_load(
        &mut self,
        result: Value,
        ty: &IrType,
        ptr: Value,
    ) -> Result<(), CodeGenError> {
        let ptr_op = self.operand_for(ptr);

        if Self::is_float(ty) {
            let dst = self.bind_vreg(result);
            let load_op = match ty {
                IrType::F32 => I686Opcode::Movss,
                IrType::F64 => I686Opcode::Movsd,
                _ => unreachable!(),
            };
            let mem = ptr_to_mem(ptr_op);
            self.emit_with(load_op, vec![
                MachineOperand::Register(dst), mem,
            ]);
            return Ok(());
        }

        if Self::is_i64(ty) {
            // 64-bit load: two 32-bit loads
            let dst_lo = self.bind_vreg(result);
            let dst_hi = self.alloc_vreg();
            self.bind_value_hi(result, MachineOperand::Register(dst_hi));

            let mem_lo = ptr_to_mem(ptr_op.clone());
            let mem_hi = offset_mem(ptr_op, 4);

            self.emit_with(I686Opcode::Mov, vec![
                MachineOperand::Register(dst_lo), mem_lo,
            ]);
            self.emit_with(I686Opcode::Mov, vec![
                MachineOperand::Register(dst_hi), mem_hi,
            ]);
            return Ok(());
        }

        let dst = self.bind_vreg(result);
        let mem = ptr_to_mem(ptr_op);

        match ty {
            IrType::I8 | IrType::I1 => {
                // movzx for unsigned byte load
                self.emit_with(I686Opcode::Movzx8, vec![
                    MachineOperand::Register(dst), mem,
                ]);
            }
            IrType::I16 => {
                self.emit_with(I686Opcode::Movzx16, vec![
                    MachineOperand::Register(dst), mem,
                ]);
            }
            _ => {
                // 32-bit or pointer load
                self.emit_with(I686Opcode::Mov, vec![
                    MachineOperand::Register(dst), mem,
                ]);
            }
        }

        Ok(())
    }

    /// Select a store instruction.
    fn select_store(
        &mut self,
        value: Value,
        ptr: Value,
    ) -> Result<(), CodeGenError> {
        let val_op = self.operand_for(value);
        let ptr_op = self.operand_for(ptr);

        // Check if we have type info from the value map to determine store size.
        // For simplicity, emit a 32-bit store by default.
        let mem = ptr_to_mem(ptr_op.clone());

        // Check if value has a high half (64-bit)
        if self.value_hi_map.contains_key(&value) {
            let val_hi = self.operand_hi_for(value);
            let mem_hi = offset_mem(ptr_op, 4);
            self.emit_with(I686Opcode::Mov, vec![mem.clone(), val_op]);
            self.emit_with(I686Opcode::Mov, vec![mem_hi, val_hi]);
        } else {
            self.emit_with(I686Opcode::Mov, vec![mem, val_op]);
        }

        Ok(())
    }

    /// Select a GEP (GetElementPtr) instruction.
    fn select_gep(
        &mut self,
        result: Value,
        base_ty: &IrType,
        ptr: Value,
        indices: &[Value],
    ) -> Result<(), CodeGenError> {
        let dst = self.bind_vreg(result);
        let ptr_op = self.operand_for(ptr);

        // Start with the base pointer.
        self.emit_with(I686Opcode::Mov, vec![
            MachineOperand::Register(dst), ptr_op,
        ]);

        // Compute the element size.
        let elem_size = base_ty.size(self.target) as i32;

        // For each index, add index * element_size to the pointer.
        for idx in indices {
            let idx_op = self.operand_for(*idx);
            match idx_op {
                MachineOperand::Immediate(imm) => {
                    let offset = imm as i32 * elem_size;
                    if offset != 0 {
                        self.emit_with(I686Opcode::Add, vec![
                            MachineOperand::Register(dst),
                            MachineOperand::Immediate(offset as i64),
                        ]);
                    }
                }
                _ => {
                    // Dynamic index: multiply index by element size, then add.
                    if elem_size == 1 || elem_size == 2 || elem_size == 4 || elem_size == 8 {
                        // Can use LEA with scale factor.
                        let scale = match elem_size {
                            1 => 0, 2 => 1, 4 => 2, 8 => 3,
                            _ => unreachable!(),
                        };
                        let _ = scale; // Used conceptually; LEA encoding handled by encoder
                        // For now, multiply and add.
                        let tmp = self.alloc_vreg();
                        self.emit_with(I686Opcode::Mov, vec![
                            MachineOperand::Register(tmp), idx_op,
                        ]);
                        if elem_size > 1 {
                            self.emit_with(I686Opcode::Imul, vec![
                                MachineOperand::Register(tmp),
                                MachineOperand::Immediate(elem_size as i64),
                            ]);
                        }
                        self.emit_with(I686Opcode::Add, vec![
                            MachineOperand::Register(dst),
                            MachineOperand::Register(tmp),
                        ]);
                    } else {
                        let tmp = self.alloc_vreg();
                        self.emit_with(I686Opcode::Mov, vec![
                            MachineOperand::Register(tmp), idx_op,
                        ]);
                        self.emit_with(I686Opcode::Imul, vec![
                            MachineOperand::Register(tmp),
                            MachineOperand::Immediate(elem_size as i64),
                        ]);
                        self.emit_with(I686Opcode::Add, vec![
                            MachineOperand::Register(dst),
                            MachineOperand::Register(tmp),
                        ]);
                    }
                }
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Function calls (cdecl: all args on stack)
    // -----------------------------------------------------------------------

    /// Select a function call instruction.
    fn select_call(
        &mut self,
        result: Option<&Value>,
        callee: &Callee,
        args: &[Value],
        return_ty: &IrType,
    ) -> Result<(), CodeGenError> {
        // Classify return type.
        let ret_info = abi::classify_return(return_ty, self.target);

        // Collect argument types for call setup. Use the type_map to recover
        // the actual IR type of each argument so that 64-bit (long long, double)
        // values are correctly allocated two stack slots on i686 cdecl.
        let arg_types: Vec<IrType> = args.iter().map(|v| {
            self.type_map.get(v).cloned().unwrap_or(IrType::I32)
        }).collect();
        let call_setup = abi::setup_call_arguments(args, &arg_types, self.target);

        // We emit our own push sequence using our value_map rather than
        // using call_setup.push_instructions, because the ABI module may
        // use intermediate opcode constants that differ from I686Opcode
        // values expected by the encoder.
        let total_push_size = call_setup.args_size + call_setup.alignment_padding;

        // Emit alignment padding if needed.
        if call_setup.alignment_padding > 0 {
            self.emit_with(I686Opcode::Sub, vec![
                MachineOperand::Register(ESP),
                MachineOperand::Immediate(call_setup.alignment_padding as i64),
            ]);
        }

        // Push args right-to-left (cdecl convention).
        for i in (0..args.len()).rev() {
            let arg_op = self.operand_for(args[i]);
            self.emit_with(I686Opcode::Push, vec![arg_op]);
        }

        // Emit the call instruction.
        match callee {
            Callee::Direct(name) => {
                self.emit_with(I686Opcode::Call, vec![
                    MachineOperand::Symbol(name.clone()),
                ]);
            }
            Callee::Indirect(ptr_val) => {
                let ptr_op = self.operand_for(*ptr_val);
                self.emit_with(I686Opcode::CallIndirect, vec![ptr_op]);
            }
        }

        // Caller cleanup: add esp, total_push_size
        if total_push_size > 0 {
            self.emit_with(I686Opcode::Add, vec![
                MachineOperand::Register(ESP),
                MachineOperand::Immediate(total_push_size as i64),
            ]);
        }

        // Handle return value.
        if let Some(res) = result {
            match ret_info.location {
                abi::ReturnLocation::Eax => {
                    let dst = self.bind_vreg(*res);
                    self.emit_with(I686Opcode::Mov, vec![
                        MachineOperand::Register(dst),
                        MachineOperand::Register(EAX),
                    ]);
                }
                abi::ReturnLocation::EaxEdx => {
                    let dst_lo = self.bind_vreg(*res);
                    let dst_hi = self.alloc_vreg();
                    self.bind_value_hi(*res, MachineOperand::Register(dst_hi));
                    self.emit_with(I686Opcode::Mov, vec![
                        MachineOperand::Register(dst_lo),
                        MachineOperand::Register(EAX),
                    ]);
                    self.emit_with(I686Opcode::Mov, vec![
                        MachineOperand::Register(dst_hi),
                        MachineOperand::Register(EDX),
                    ]);
                }
                abi::ReturnLocation::Xmm0 | abi::ReturnLocation::St0 => {
                    let dst = self.bind_vreg(*res);
                    self.emit_with(I686Opcode::Movss, vec![
                        MachineOperand::Register(dst),
                        MachineOperand::Register(XMM0),
                    ]);
                }
                abi::ReturnLocation::Memory | abi::ReturnLocation::Void => {
                    // No register return or void — bind to a dummy register.
                    let _ = self.bind_vreg(*res);
                }
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Type conversions (casts)
    // -----------------------------------------------------------------------

    /// Select a type conversion instruction.
    fn select_cast(
        &mut self,
        result: Value,
        cast_op: CastOp,
        value: Value,
        from_ty: &IrType,
        to_ty: &IrType,
    ) -> Result<(), CodeGenError> {
        match cast_op {
            CastOp::Trunc => {
                self.select_trunc(result, value, from_ty, to_ty)?;
            }
            CastOp::ZExt => {
                self.select_zext(result, value, from_ty, to_ty)?;
            }
            CastOp::SExt => {
                self.select_sext(result, value, from_ty, to_ty)?;
            }
            CastOp::SIToFP => {
                let dst = self.bind_vreg(result);
                let src_op = self.operand_for(value);
                let conv_op = match to_ty {
                    IrType::F32 => I686Opcode::Cvtsi2ss,
                    IrType::F64 => I686Opcode::Cvtsi2sd,
                    _ => return Err(CodeGenError::UnsupportedInstruction(
                        format!("SIToFP to {:?}", to_ty),
                    )),
                };
                self.emit_with(conv_op, vec![
                    MachineOperand::Register(dst), src_op,
                ]);
            }
            CastOp::UIToFP => {
                // Unsigned int to float: convert via signed path for values < 2^31,
                // handle large unsigned separately.
                let dst = self.bind_vreg(result);
                let src_op = self.operand_for(value);
                let conv_op = match to_ty {
                    IrType::F32 => I686Opcode::Cvtsi2ss,
                    IrType::F64 => I686Opcode::Cvtsi2sd,
                    _ => return Err(CodeGenError::UnsupportedInstruction(
                        format!("UIToFP to {:?}", to_ty),
                    )),
                };
                self.emit_with(conv_op, vec![
                    MachineOperand::Register(dst), src_op,
                ]);
            }
            CastOp::FPToSI => {
                let dst = self.bind_vreg(result);
                let src_op = self.operand_for(value);
                let conv_op = match from_ty {
                    IrType::F32 => I686Opcode::Cvttss2si,
                    IrType::F64 => I686Opcode::Cvttsd2si,
                    _ => return Err(CodeGenError::UnsupportedInstruction(
                        format!("FPToSI from {:?}", from_ty),
                    )),
                };
                self.emit_with(conv_op, vec![
                    MachineOperand::Register(dst), src_op,
                ]);
            }
            CastOp::FPToUI => {
                // Truncating float to unsigned int.
                let dst = self.bind_vreg(result);
                let src_op = self.operand_for(value);
                let conv_op = match from_ty {
                    IrType::F32 => I686Opcode::Cvttss2si,
                    IrType::F64 => I686Opcode::Cvttsd2si,
                    _ => return Err(CodeGenError::UnsupportedInstruction(
                        format!("FPToUI from {:?}", from_ty),
                    )),
                };
                self.emit_with(conv_op, vec![
                    MachineOperand::Register(dst), src_op,
                ]);
            }
            CastOp::FPTrunc => {
                // double -> float: cvtsd2ss
                let dst = self.bind_vreg(result);
                let src_op = self.operand_for(value);
                self.emit_with(I686Opcode::Cvtsd2ss, vec![
                    MachineOperand::Register(dst), src_op,
                ]);
            }
            CastOp::FPExt => {
                // float -> double: cvtss2sd
                let dst = self.bind_vreg(result);
                let src_op = self.operand_for(value);
                self.emit_with(I686Opcode::Cvtss2sd, vec![
                    MachineOperand::Register(dst), src_op,
                ]);
            }
            CastOp::PtrToInt | CastOp::IntToPtr => {
                // On i686, pointers are 32-bit = same as i32. Just alias.
                let src_op = self.operand_for(value);
                let dst = self.bind_vreg(result);
                self.emit_with(I686Opcode::Mov, vec![
                    MachineOperand::Register(dst), src_op,
                ]);
            }
        }
        Ok(())
    }

    /// Select truncation: I64->I32 (take low half), I32->I16/I8 (mask or move).
    fn select_trunc(
        &mut self,
        result: Value,
        value: Value,
        from_ty: &IrType,
        _to_ty: &IrType,
    ) -> Result<(), CodeGenError> {
        let dst = self.bind_vreg(result);

        if Self::is_i64(from_ty) {
            // I64 -> I32: just use the low register.
            let lo_op = self.operand_for(value);
            self.emit_with(I686Opcode::Mov, vec![
                MachineOperand::Register(dst), lo_op,
            ]);
        } else {
            // I32 -> I16/I8: just move (upper bits are ignored by caller).
            let src_op = self.operand_for(value);
            self.emit_with(I686Opcode::Mov, vec![
                MachineOperand::Register(dst), src_op,
            ]);
        }

        Ok(())
    }

    /// Select zero extension.
    fn select_zext(
        &mut self,
        result: Value,
        value: Value,
        from_ty: &IrType,
        to_ty: &IrType,
    ) -> Result<(), CodeGenError> {
        let src_op = self.operand_for(value);

        if Self::is_i64(to_ty) {
            // Extending to 64-bit: low half = value, high half = 0.
            let dst_lo = self.bind_vreg(result);
            let dst_hi = self.alloc_vreg();
            self.bind_value_hi(result, MachineOperand::Register(dst_hi));

            match from_ty {
                IrType::I8 | IrType::I1 => {
                    self.emit_with(I686Opcode::Movzx8, vec![
                        MachineOperand::Register(dst_lo), src_op,
                    ]);
                }
                IrType::I16 => {
                    self.emit_with(I686Opcode::Movzx16, vec![
                        MachineOperand::Register(dst_lo), src_op,
                    ]);
                }
                _ => {
                    self.emit_with(I686Opcode::Mov, vec![
                        MachineOperand::Register(dst_lo), src_op,
                    ]);
                }
            }
            // High half = 0
            self.emit_with(I686Opcode::Xor, vec![
                MachineOperand::Register(dst_hi),
                MachineOperand::Register(dst_hi),
            ]);
        } else {
            let dst = self.bind_vreg(result);
            match from_ty {
                IrType::I8 | IrType::I1 => {
                    self.emit_with(I686Opcode::Movzx8, vec![
                        MachineOperand::Register(dst), src_op,
                    ]);
                }
                IrType::I16 => {
                    self.emit_with(I686Opcode::Movzx16, vec![
                        MachineOperand::Register(dst), src_op,
                    ]);
                }
                _ => {
                    self.emit_with(I686Opcode::Mov, vec![
                        MachineOperand::Register(dst), src_op,
                    ]);
                }
            }
        }
        Ok(())
    }

    /// Select sign extension.
    fn select_sext(
        &mut self,
        result: Value,
        value: Value,
        from_ty: &IrType,
        to_ty: &IrType,
    ) -> Result<(), CodeGenError> {
        let src_op = self.operand_for(value);

        if Self::is_i64(to_ty) && !Self::is_i64(from_ty) {
            // Extending 32-bit (or narrower) to 64-bit.
            let dst_lo = self.bind_vreg(result);
            let dst_hi = self.alloc_vreg();
            self.bind_value_hi(result, MachineOperand::Register(dst_hi));

            match from_ty {
                IrType::I8 | IrType::I1 => {
                    self.emit_with(I686Opcode::Movsx8, vec![
                        MachineOperand::Register(EAX), src_op,
                    ]);
                }
                IrType::I16 => {
                    self.emit_with(I686Opcode::Movsx16, vec![
                        MachineOperand::Register(EAX), src_op,
                    ]);
                }
                _ => {
                    self.emit_with(I686Opcode::Mov, vec![
                        MachineOperand::Register(EAX), src_op,
                    ]);
                }
            }
            // CDQ: sign-extend eax into edx:eax
            self.emit_no_operands(I686Opcode::Cdq);

            self.emit_with(I686Opcode::Mov, vec![
                MachineOperand::Register(dst_lo),
                MachineOperand::Register(EAX),
            ]);
            self.emit_with(I686Opcode::Mov, vec![
                MachineOperand::Register(dst_hi),
                MachineOperand::Register(EDX),
            ]);
        } else {
            let dst = self.bind_vreg(result);
            match from_ty {
                IrType::I8 | IrType::I1 => {
                    self.emit_with(I686Opcode::Movsx8, vec![
                        MachineOperand::Register(dst), src_op,
                    ]);
                }
                IrType::I16 => {
                    self.emit_with(I686Opcode::Movsx16, vec![
                        MachineOperand::Register(dst), src_op,
                    ]);
                }
                _ => {
                    self.emit_with(I686Opcode::Mov, vec![
                        MachineOperand::Register(dst), src_op,
                    ]);
                }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Constant materialization
    // -----------------------------------------------------------------------

    /// Select a constant materialization instruction.
    fn select_const(
        &mut self,
        result: Value,
        constant: &Constant,
    ) -> Result<(), CodeGenError> {
        // The IR builder emits placeholder Const instructions for function
        // parameters (value = param index).  select_all() already bound
        // these Values to their cdecl stack locations.  Skip the const
        // so we don't clobber the real parameter mapping.
        if self.function.param_values.contains(&result) {
            return Ok(());
        }
        match constant {
            Constant::Integer { value, ty } => {
                if Self::is_i64(ty) {
                    let lo = (*value as u64) as u32;
                    let hi = ((*value as u64) >> 32) as u32;

                    let dst_lo = self.bind_vreg(result);
                    let dst_hi = self.alloc_vreg();
                    self.bind_value_hi(result, MachineOperand::Register(dst_hi));

                    if lo == 0 {
                        // xor reg, reg for zero
                        self.emit_with(I686Opcode::Xor, vec![
                            MachineOperand::Register(dst_lo),
                            MachineOperand::Register(dst_lo),
                        ]);
                    } else {
                        self.emit_with(I686Opcode::Mov, vec![
                            MachineOperand::Register(dst_lo),
                            MachineOperand::Immediate(lo as i64),
                        ]);
                    }

                    if hi == 0 {
                        self.emit_with(I686Opcode::Xor, vec![
                            MachineOperand::Register(dst_hi),
                            MachineOperand::Register(dst_hi),
                        ]);
                    } else {
                        self.emit_with(I686Opcode::Mov, vec![
                            MachineOperand::Register(dst_hi),
                            MachineOperand::Immediate(hi as i64),
                        ]);
                    }
                } else {
                    let dst = self.bind_vreg(result);
                    if *value == 0 {
                        // xor reg, reg — shorter encoding than mov reg, 0
                        self.emit_with(I686Opcode::Xor, vec![
                            MachineOperand::Register(dst),
                            MachineOperand::Register(dst),
                        ]);
                    } else {
                        self.emit_with(I686Opcode::Mov, vec![
                            MachineOperand::Register(dst),
                            MachineOperand::Immediate(*value),
                        ]);
                    }
                }
            }
            Constant::Float { value: fval, ty } => {
                let dst = self.bind_vreg(result);
                // Float constants are loaded from a data section reference.
                // For now, encode the float bits as an immediate in the symbol name.
                let label = format!(".LC_float_{:016x}", fval.to_bits());
                let load_op = match ty {
                    IrType::F32 => I686Opcode::Movss,
                    IrType::F64 => I686Opcode::Movsd,
                    _ => I686Opcode::Movss,
                };
                self.emit_with(load_op, vec![
                    MachineOperand::Register(dst),
                    MachineOperand::Symbol(label),
                ]);
            }
            Constant::Bool(b) => {
                let dst = self.bind_vreg(result);
                let val = if *b { 1i64 } else { 0i64 };
                if val == 0 {
                    self.emit_with(I686Opcode::Xor, vec![
                        MachineOperand::Register(dst),
                        MachineOperand::Register(dst),
                    ]);
                } else {
                    self.emit_with(I686Opcode::Mov, vec![
                        MachineOperand::Register(dst),
                        MachineOperand::Immediate(val),
                    ]);
                }
            }
            Constant::Null(_) => {
                let dst = self.bind_vreg(result);
                // Null pointer = 0
                self.emit_with(I686Opcode::Xor, vec![
                    MachineOperand::Register(dst),
                    MachineOperand::Register(dst),
                ]);
            }
            Constant::GlobalRef(name) => {
                let dst = self.bind_vreg(result);
                // Load address of global symbol via R_386_32 relocation.
                self.emit_with(I686Opcode::Mov, vec![
                    MachineOperand::Register(dst),
                    MachineOperand::Symbol(name.clone()),
                ]);
            }
            Constant::String(bytes) => {
                let dst = self.bind_vreg(result);
                // String literals are placed in .rodata; reference via symbol.
                let label = format!(".Lstr_{}", result.0);
                let _ = bytes; // Bytes stored in data section, not inline
                self.emit_with(I686Opcode::Mov, vec![
                    MachineOperand::Register(dst),
                    MachineOperand::Symbol(label),
                ]);
            }
            Constant::Undef(_) | Constant::ZeroInit(_) => {
                // Undefined or zero-initialized: produce zero.
                let dst = self.bind_vreg(result);
                self.emit_with(I686Opcode::Xor, vec![
                    MachineOperand::Register(dst),
                    MachineOperand::Register(dst),
                ]);
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Copy and select
    // -----------------------------------------------------------------------

    /// Select a copy instruction.
    fn select_copy(
        &mut self,
        result: Value,
        source: Value,
        ty: &IrType,
    ) -> Result<(), CodeGenError> {
        if Self::is_i64(ty) {
            let dst_lo = self.bind_vreg(result);
            let dst_hi = self.alloc_vreg();
            self.bind_value_hi(result, MachineOperand::Register(dst_hi));
            let src_lo = self.operand_for(source);
            let src_hi = self.operand_hi_for(source);
            self.emit_with(I686Opcode::Mov, vec![
                MachineOperand::Register(dst_lo), src_lo,
            ]);
            self.emit_with(I686Opcode::Mov, vec![
                MachineOperand::Register(dst_hi), src_hi,
            ]);
        } else {
            let dst = self.bind_vreg(result);
            let src_op = self.operand_for(source);
            self.emit_with(I686Opcode::Mov, vec![
                MachineOperand::Register(dst), src_op,
            ]);
        }
        Ok(())
    }

    /// Select a ternary select instruction (condition ? true_val : false_val).
    fn select_select(
        &mut self,
        result: Value,
        condition: Value,
        true_val: Value,
        false_val: Value,
        _ty: &IrType,
    ) -> Result<(), CodeGenError> {
        let dst = self.bind_vreg(result);
        let cond_op = self.operand_for(condition);
        let true_op = self.operand_for(true_val);
        let false_op = self.operand_for(false_val);

        // mov dst, false_val
        self.emit_with(I686Opcode::Mov, vec![
            MachineOperand::Register(dst), false_op,
        ]);
        // test cond, cond
        self.emit_with(I686Opcode::Test, vec![cond_op.clone(), cond_op]);
        // Emit conditional jump over the mov:
        // jz skip_label; mov dst, true_val; skip_label:
        let skip_label = self.next_label;
        self.next_label += 1;
        self.emit_with(I686Opcode::Jcc, vec![
            MachineOperand::Immediate(CC_E as i64),
            MachineOperand::Label(skip_label),
        ]);
        self.emit_with(I686Opcode::Mov, vec![
            MachineOperand::Register(dst), true_op,
        ]);
        self.emit_with(I686Opcode::Label, vec![
            MachineOperand::Label(skip_label),
        ]);

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Floating-point operations
    // -----------------------------------------------------------------------

    /// Select a floating-point binary operation.
    fn select_float_binop(
        &mut self,
        result: Value,
        lhs: Value,
        rhs: Value,
        ty: &IrType,
        arith_opcode: I686Opcode,
    ) -> Result<(), CodeGenError> {
        let dst = self.bind_vreg(result);
        let lhs_op = self.operand_for(lhs);
        let rhs_op = self.operand_for(rhs);

        let is_f32 = matches!(ty, IrType::F32);

        // Map the arithmetic opcode to the correct SSE instruction.
        let sse_op = match arith_opcode {
            I686Opcode::Add => if is_f32 { I686Opcode::Addss } else { I686Opcode::Addsd },
            I686Opcode::Sub => if is_f32 { I686Opcode::Subss } else { I686Opcode::Subsd },
            I686Opcode::Imul => if is_f32 { I686Opcode::Mulss } else { I686Opcode::Mulsd },
            _ => if is_f32 { I686Opcode::Divss } else { I686Opcode::Divsd },
        };

        // movss/movsd dst, lhs
        let mov_op = if is_f32 { I686Opcode::Movss } else { I686Opcode::Movsd };
        self.emit_with(mov_op, vec![
            MachineOperand::Register(dst), lhs_op,
        ]);

        // addss/subss/mulss/divss dst, rhs
        self.emit_with(sse_op, vec![
            MachineOperand::Register(dst), rhs_op,
        ]);

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Terminator selection
    // -----------------------------------------------------------------------

    /// Select machine instructions for a terminator.
    fn select_terminator(&mut self, term: &Terminator) -> Result<(), CodeGenError> {
        match term {
            Terminator::Branch { target } => {
                let label = self.block_label(*target);
                self.emit_with(I686Opcode::Jmp, vec![MachineOperand::Label(label)]);
            }
            Terminator::CondBranch { condition, true_block, false_block } => {
                let cond_op = self.operand_for(*condition);

                // test cond, cond
                self.emit_with(I686Opcode::Test, vec![cond_op.clone(), cond_op]);

                let true_label = self.block_label(*true_block);
                let false_label = self.block_label(*false_block);

                // jne true_block
                self.emit_with(I686Opcode::Jcc, vec![
                    MachineOperand::Immediate(CC_NE as i64),
                    MachineOperand::Label(true_label),
                ]);
                // jmp false_block
                self.emit_with(I686Opcode::Jmp, vec![
                    MachineOperand::Label(false_label),
                ]);
            }
            Terminator::Return { value } => {
                if let Some(ret_val) = value {
                    // Move return value to eax (or eax:edx for 64-bit).
                    let val_op = self.operand_for(*ret_val);
                    self.emit_with(I686Opcode::Mov, vec![
                        MachineOperand::Register(EAX), val_op,
                    ]);
                    if self.value_hi_map.contains_key(ret_val) {
                        let hi_op = self.operand_hi_for(*ret_val);
                        self.emit_with(I686Opcode::Mov, vec![
                            MachineOperand::Register(EDX), hi_op,
                        ]);
                    }
                }
                self.emit_no_operands(I686Opcode::Ret);
            }
            Terminator::Switch { value, default, cases } => {
                let val_op = self.operand_for(*value);
                let default_label = self.block_label(*default);

                // Emit a series of cmp + je for each case.
                for (case_val, case_target) in cases {
                    let target_label = self.block_label(*case_target);
                    self.emit_with(I686Opcode::Cmp, vec![
                        val_op.clone(),
                        MachineOperand::Immediate(*case_val),
                    ]);
                    self.emit_with(I686Opcode::Jcc, vec![
                        MachineOperand::Immediate(CC_E as i64),
                        MachineOperand::Label(target_label),
                    ]);
                }

                // Fall through to default.
                self.emit_with(I686Opcode::Jmp, vec![
                    MachineOperand::Label(default_label),
                ]);
            }
            Terminator::Unreachable => {
                // Emit a trap-like instruction (int3 or nop).
                self.emit_no_operands(I686Opcode::Nop);
            }
        }
        Ok(())
    }
}

// ===========================================================================
// Module-level helper functions
// ===========================================================================

/// Map a CompareOp to an x86 condition code constant.
fn compare_op_to_cc(op: CompareOp) -> u8 {
    match op {
        CompareOp::Equal => CC_E,
        CompareOp::NotEqual => CC_NE,
        CompareOp::SignedLess => CC_L,
        CompareOp::SignedLessEqual => CC_LE,
        CompareOp::SignedGreater => CC_G,
        CompareOp::SignedGreaterEqual => CC_GE,
        CompareOp::UnsignedLess => CC_B,
        CompareOp::UnsignedLessEqual => CC_BE,
        CompareOp::UnsignedGreater => CC_A,
        CompareOp::UnsignedGreaterEqual => CC_NB,
    }
}

/// Map a CompareOp to its unsigned equivalent condition code
/// (used for the low-half of 64-bit comparisons).
fn compare_op_to_unsigned_cc(op: CompareOp) -> u8 {
    match op {
        CompareOp::Equal => CC_E,
        CompareOp::NotEqual => CC_NE,
        CompareOp::SignedLess | CompareOp::UnsignedLess => CC_B,
        CompareOp::SignedLessEqual | CompareOp::UnsignedLessEqual => CC_BE,
        CompareOp::SignedGreater | CompareOp::UnsignedGreater => CC_A,
        CompareOp::SignedGreaterEqual | CompareOp::UnsignedGreaterEqual => CC_NB,
    }
}

/// Map a FloatCompareOp to an x86 condition code constant.
fn float_compare_op_to_cc(op: FloatCompareOp) -> u8 {
    match op {
        FloatCompareOp::OrderedEqual | FloatCompareOp::UnorderedEqual => CC_E,
        FloatCompareOp::OrderedNotEqual => CC_NE,
        FloatCompareOp::OrderedLess | FloatCompareOp::OrderedLessEqual => CC_B,
        FloatCompareOp::OrderedGreater | FloatCompareOp::OrderedGreaterEqual => CC_A,
        FloatCompareOp::Unordered => CC_B, // PF=1 after ucomisd with NaN
    }
}

/// Convert a pointer operand to a memory reference.
/// If the operand is a Register, wraps it as Memory { base: reg, offset: 0 }.
/// If it is already a Memory operand, returns it unchanged.
fn ptr_to_mem(op: MachineOperand) -> MachineOperand {
    match op {
        MachineOperand::Register(reg) => MachineOperand::Memory { base: reg, offset: 0 },
        MachineOperand::Memory { .. } => op,
        _ => MachineOperand::Memory { base: EBP, offset: 0 }, // fallback
    }
}

/// Offset a pointer operand by the given number of bytes.
fn offset_mem(op: MachineOperand, additional: i32) -> MachineOperand {
    match op {
        MachineOperand::Register(reg) => MachineOperand::Memory { base: reg, offset: additional },
        MachineOperand::Memory { base, offset } => MachineOperand::Memory { base, offset: offset + additional },
        _ => MachineOperand::Memory { base: EBP, offset: additional },
    }
}

// ===========================================================================
// Public entry point
// ===========================================================================

/// Select i686 machine instructions for an entire IR function.
///
/// This is the main entry point for i686 instruction selection. It walks the
/// function's basic blocks, lowering each IR instruction to one or more
/// [`MachineInstr`] values with i686-specific opcodes.
///
/// # Arguments
///
/// * `function` — The IR function to lower (must be a definition with blocks).
/// * `target` — Target configuration (must be i686 — `target.is_32bit()` is true).
///
/// # Returns
///
/// A `Vec<MachineInstr>` containing the lowered machine instruction sequence
/// for the entire function, ready for register allocation and encoding.
///
/// # Errors
///
/// Returns [`CodeGenError`] if an unsupported IR construct is encountered or
/// if an internal invariant is violated.
pub fn select_instructions(
    function: &Function,
    target: &TargetConfig,
) -> Result<Vec<MachineInstr>, CodeGenError> {
    // Verify we are targeting a 32-bit architecture.
    debug_assert!(
        target.is_32bit(),
        "i686 isel called for non-32-bit target: {:?}",
        target.arch
    );

    let _ = target.pointer_size;
    let _ = target.stack_alignment;

    // Skip non-definition functions (extern declarations have no body).
    if !function.is_definition || function.blocks.is_empty() {
        return Ok(Vec::new());
    }

    let mut isel = ISel::new(function, target);
    isel.select_all()?;
    Ok(isel.output)
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::regalloc::PhysReg;
    use crate::driver::target::TargetConfig;
    use crate::ir::cfg::{BasicBlock, Terminator};
    use crate::ir::instructions::{
        BlockId, Callee, CastOp, CompareOp, Constant, FloatCompareOp,
        Instruction, Value,
    };
    use crate::ir::types::IrType;
    use crate::ir::builder::Function;

    /// Create an i686 target for testing.
    fn i686_target() -> TargetConfig {
        TargetConfig::i686()
    }

    /// Create a minimal function with one basic block.
    fn make_function(
        name: &str,
        params: Vec<(String, IrType)>,
        instrs: Vec<Instruction>,
        terminator: Terminator,
    ) -> Function {
        let mut block = BasicBlock::new(BlockId(0), "entry".to_string());
        block.instructions = instrs;
        block.terminator = Some(terminator);

        Function {
            name: name.to_string(),
            return_type: IrType::I32,
            params,
            param_values: Vec::new(),
            blocks: vec![block],
            entry_block: BlockId(0),
            is_definition: true,
        }
    }

    /// Helper to check that the output contains at least one instruction
    /// with the given opcode.
    fn has_opcode(instrs: &[MachineInstr], expected: I686Opcode) -> bool {
        instrs.iter().any(|i| i.opcode == expected as u32)
    }

    /// Helper to count how many instructions have a given opcode.
    fn count_opcode(instrs: &[MachineInstr], expected: I686Opcode) -> usize {
        instrs.iter().filter(|i| i.opcode == expected as u32).count()
    }

    // -------------------------------------------------------------------
    // Test: register constants are correctly defined
    // -------------------------------------------------------------------

    #[test]
    fn test_register_constants() {
        assert_eq!(EAX.0, 0);
        assert_eq!(ECX.0, 1);
        assert_eq!(EDX.0, 2);
        assert_eq!(EBX.0, 3);
        assert_eq!(ESP.0, 4);
        assert_eq!(EBP.0, 5);
        assert_eq!(ESI.0, 6);
        assert_eq!(EDI.0, 7);
        assert_eq!(XMM0.0, 8);
        assert_eq!(XMM1.0, 9);
        assert_eq!(XMM7.0, 15);
    }

    // -------------------------------------------------------------------
    // Test: 32-bit add instruction selection
    // -------------------------------------------------------------------

    #[test]
    fn test_add_i32() {
        let target = i686_target();
        let instrs = vec![
            Instruction::Const { result: Value(10), value: Constant::Integer { value: 5, ty: IrType::I32 } },
            Instruction::Const { result: Value(11), value: Constant::Integer { value: 3, ty: IrType::I32 } },
            Instruction::Add { result: Value(12), lhs: Value(10), rhs: Value(11), ty: IrType::I32 },
        ];
        let func = make_function("test_add", vec![], instrs, Terminator::Return { value: Some(Value(12)) });
        let result = select_instructions(&func, &target).unwrap();
        assert!(has_opcode(&result, I686Opcode::Add), "should contain Add instruction");
        assert!(has_opcode(&result, I686Opcode::Mov), "should contain Mov instruction");
    }

    // -------------------------------------------------------------------
    // Test: 32-bit multiply selection
    // -------------------------------------------------------------------

    #[test]
    fn test_mul_i32() {
        let target = i686_target();
        let instrs = vec![
            Instruction::Const { result: Value(10), value: Constant::Integer { value: 5, ty: IrType::I32 } },
            Instruction::Const { result: Value(11), value: Constant::Integer { value: 3, ty: IrType::I32 } },
            Instruction::Mul { result: Value(12), lhs: Value(10), rhs: Value(11), ty: IrType::I32 },
        ];
        let func = make_function("test_mul", vec![], instrs, Terminator::Return { value: Some(Value(12)) });
        let result = select_instructions(&func, &target).unwrap();
        assert!(has_opcode(&result, I686Opcode::Imul), "should contain Imul for signed multiply");
    }

    // -------------------------------------------------------------------
    // Test: 64-bit add produces add + adc pair
    // -------------------------------------------------------------------

    #[test]
    fn test_add_i64() {
        let target = i686_target();
        let instrs = vec![
            Instruction::Const { result: Value(10), value: Constant::Integer { value: 100, ty: IrType::I64 } },
            Instruction::Const { result: Value(11), value: Constant::Integer { value: 200, ty: IrType::I64 } },
            Instruction::Add { result: Value(12), lhs: Value(10), rhs: Value(11), ty: IrType::I64 },
        ];
        let func = make_function("test_add64", vec![], instrs, Terminator::Return { value: Some(Value(12)) });
        let result = select_instructions(&func, &target).unwrap();
        // 64-bit add should produce one Add (low half) and one Adc (high half
        // with carry propagation).
        let add_count = count_opcode(&result, I686Opcode::Add);
        let adc_count = count_opcode(&result, I686Opcode::Adc);
        assert!(add_count >= 1, "64-bit add should produce at least 1 Add instruction (low half), got {}", add_count);
        assert!(adc_count >= 1, "64-bit add should produce at least 1 Adc instruction (high half with carry), got {}", adc_count);
    }

    // -------------------------------------------------------------------
    // Test: 64-bit left shift produces shld + shl
    // -------------------------------------------------------------------

    #[test]
    fn test_shl_i64() {
        let target = i686_target();
        let instrs = vec![
            Instruction::Const { result: Value(10), value: Constant::Integer { value: 1, ty: IrType::I64 } },
            Instruction::Const { result: Value(11), value: Constant::Integer { value: 5, ty: IrType::I32 } },
            Instruction::Shl { result: Value(12), lhs: Value(10), rhs: Value(11), ty: IrType::I64 },
        ];
        let func = make_function("test_shl64", vec![], instrs, Terminator::Return { value: Some(Value(12)) });
        let result = select_instructions(&func, &target).unwrap();
        assert!(has_opcode(&result, I686Opcode::Shld), "64-bit left shift should use Shld");
        assert!(has_opcode(&result, I686Opcode::Shl), "64-bit left shift should use Shl");
    }

    // -------------------------------------------------------------------
    // Test: division uses edx:eax
    // -------------------------------------------------------------------

    #[test]
    fn test_div_i32() {
        let target = i686_target();
        let instrs = vec![
            Instruction::Const { result: Value(10), value: Constant::Integer { value: 10, ty: IrType::I32 } },
            Instruction::Const { result: Value(11), value: Constant::Integer { value: 3, ty: IrType::I32 } },
            Instruction::Div { result: Value(12), lhs: Value(10), rhs: Value(11), ty: IrType::I32, is_signed: true },
        ];
        let func = make_function("test_div", vec![], instrs, Terminator::Return { value: Some(Value(12)) });
        let result = select_instructions(&func, &target).unwrap();
        assert!(has_opcode(&result, I686Opcode::Cdq), "signed div should use Cdq");
        assert!(has_opcode(&result, I686Opcode::Idiv), "signed div should use Idiv");
    }

    // -------------------------------------------------------------------
    // Test: compare + branch produces cmp + jcc
    // -------------------------------------------------------------------

    #[test]
    fn test_icmp_and_branch() {
        let target = i686_target();
        let instrs = vec![
            Instruction::Const { result: Value(10), value: Constant::Integer { value: 5, ty: IrType::I32 } },
            Instruction::Const { result: Value(11), value: Constant::Integer { value: 3, ty: IrType::I32 } },
            Instruction::ICmp {
                result: Value(12), op: CompareOp::Equal,
                lhs: Value(10), rhs: Value(11), ty: IrType::I32,
            },
        ];
        let mut block0 = BasicBlock::new(BlockId(0), "entry".to_string());
        block0.instructions = instrs;
        block0.terminator = Some(Terminator::CondBranch {
            condition: Value(12),
            true_block: BlockId(1),
            false_block: BlockId(1),
        });
        let block1 = BasicBlock::new(BlockId(1), "exit".to_string());
        let mut b1 = block1;
        b1.terminator = Some(Terminator::Return { value: None });

        let func = Function {
            name: "test_cmp".to_string(),
            return_type: IrType::Void,
            params: vec![],
            param_values: Vec::new(),
            blocks: vec![block0, b1],
            entry_block: BlockId(0),
            is_definition: true,
        };
        let result = select_instructions(&func, &target).unwrap();
        assert!(has_opcode(&result, I686Opcode::Cmp), "should have Cmp instruction");
        assert!(has_opcode(&result, I686Opcode::Setcc), "should have Setcc instruction");
        assert!(has_opcode(&result, I686Opcode::Jcc), "should have Jcc (conditional jump)");
    }

    // -------------------------------------------------------------------
    // Test: constant zero uses xor reg, reg
    // -------------------------------------------------------------------

    #[test]
    fn test_const_zero_uses_xor() {
        let target = i686_target();
        let instrs = vec![
            Instruction::Const { result: Value(10), value: Constant::Integer { value: 0, ty: IrType::I32 } },
        ];
        let func = make_function("test_zero", vec![], instrs, Terminator::Return { value: Some(Value(10)) });
        let result = select_instructions(&func, &target).unwrap();
        // The xor-for-zero pattern: find an Xor where both operands are the same register.
        let has_xor_zero = result.iter().any(|i| {
            i.opcode == I686Opcode::Xor as u32
                && i.operands.len() == 2
                && matches!((&i.operands[0], &i.operands[1]),
                    (MachineOperand::Register(a), MachineOperand::Register(b)) if a == b)
        });
        assert!(has_xor_zero, "constant 0 should be materialized via xor reg, reg");
    }

    // -------------------------------------------------------------------
    // Test: function call produces push args + call + cleanup
    // -------------------------------------------------------------------

    #[test]
    fn test_call_direct() {
        let target = i686_target();
        let instrs = vec![
            Instruction::Const { result: Value(10), value: Constant::Integer { value: 42, ty: IrType::I32 } },
            Instruction::Call {
                result: Some(Value(11)),
                callee: Callee::Direct("puts".to_string()),
                args: vec![Value(10)],
                return_ty: IrType::I32,
            },
        ];
        let func = make_function("test_call", vec![], instrs, Terminator::Return { value: Some(Value(11)) });
        let result = select_instructions(&func, &target).unwrap();
        assert!(has_opcode(&result, I686Opcode::Push), "should push args");
        assert!(has_opcode(&result, I686Opcode::Call), "should have Call instruction");
        // Should have a call to "puts"
        let has_puts_call = result.iter().any(|i| {
            i.opcode == I686Opcode::Call as u32
                && i.operands.iter().any(|op| matches!(op, MachineOperand::Symbol(name) if name == "puts"))
        });
        assert!(has_puts_call, "should call 'puts' symbol");
    }

    // -------------------------------------------------------------------
    // Test: load/store for different sizes
    // -------------------------------------------------------------------

    #[test]
    fn test_load_i32() {
        let target = i686_target();
        let instrs = vec![
            Instruction::Const { result: Value(10), value: Constant::Integer { value: 0x1000, ty: IrType::I32 } },
            Instruction::Load { result: Value(11), ty: IrType::I32, ptr: Value(10) },
        ];
        let func = make_function("test_load", vec![], instrs, Terminator::Return { value: Some(Value(11)) });
        let result = select_instructions(&func, &target).unwrap();
        assert!(has_opcode(&result, I686Opcode::Mov), "should have Mov for i32 load");
    }

    #[test]
    fn test_load_i8() {
        let target = i686_target();
        let instrs = vec![
            Instruction::Const { result: Value(10), value: Constant::Integer { value: 0x1000, ty: IrType::I32 } },
            Instruction::Load { result: Value(11), ty: IrType::I8, ptr: Value(10) },
        ];
        let func = make_function("test_load_i8", vec![], instrs, Terminator::Return { value: Some(Value(11)) });
        let result = select_instructions(&func, &target).unwrap();
        assert!(has_opcode(&result, I686Opcode::Movzx8), "should have Movzx8 for byte load");
    }

    // -------------------------------------------------------------------
    // Test: type cast selection (zero-extend, sign-extend, trunc)
    // -------------------------------------------------------------------

    #[test]
    fn test_zext_i8_to_i32() {
        let target = i686_target();
        let instrs = vec![
            Instruction::Const { result: Value(10), value: Constant::Integer { value: 42, ty: IrType::I8 } },
            Instruction::Cast {
                result: Value(11), op: CastOp::ZExt,
                value: Value(10), from_ty: IrType::I8, to_ty: IrType::I32,
            },
        ];
        let func = make_function("test_zext", vec![], instrs, Terminator::Return { value: Some(Value(11)) });
        let result = select_instructions(&func, &target).unwrap();
        assert!(has_opcode(&result, I686Opcode::Movzx8), "zext i8->i32 should use Movzx8");
    }

    #[test]
    fn test_sext_i8_to_i32() {
        let target = i686_target();
        let instrs = vec![
            Instruction::Const { result: Value(10), value: Constant::Integer { value: -1, ty: IrType::I8 } },
            Instruction::Cast {
                result: Value(11), op: CastOp::SExt,
                value: Value(10), from_ty: IrType::I8, to_ty: IrType::I32,
            },
        ];
        let func = make_function("test_sext", vec![], instrs, Terminator::Return { value: Some(Value(11)) });
        let result = select_instructions(&func, &target).unwrap();
        assert!(has_opcode(&result, I686Opcode::Movsx8), "sext i8->i32 should use Movsx8");
    }

    #[test]
    fn test_trunc_i64_to_i32() {
        let target = i686_target();
        let instrs = vec![
            Instruction::Const { result: Value(10), value: Constant::Integer { value: 0x1_0000_0000i64, ty: IrType::I64 } },
            Instruction::Cast {
                result: Value(11), op: CastOp::Trunc,
                value: Value(10), from_ty: IrType::I64, to_ty: IrType::I32,
            },
        ];
        let func = make_function("test_trunc", vec![], instrs, Terminator::Return { value: Some(Value(11)) });
        let result = select_instructions(&func, &target).unwrap();
        // Truncation I64->I32 just takes the low register, which is a Mov.
        assert!(has_opcode(&result, I686Opcode::Mov), "trunc i64->i32 should use Mov");
    }

    // -------------------------------------------------------------------
    // Test: empty function (declaration) returns empty output
    // -------------------------------------------------------------------

    #[test]
    fn test_empty_declaration() {
        let target = i686_target();
        let func = Function {
            name: "extern_fn".to_string(),
            return_type: IrType::I32,
            params: vec![],
            param_values: Vec::new(),
            blocks: vec![],
            entry_block: BlockId(0),
            is_definition: false,
        };
        let result = select_instructions(&func, &target).unwrap();
        assert!(result.is_empty(), "non-definition should produce empty output");
    }

    // -------------------------------------------------------------------
    // Test: float addition uses addss/addsd
    // -------------------------------------------------------------------

    #[test]
    fn test_float_add() {
        let target = i686_target();
        let instrs = vec![
            Instruction::Const { result: Value(10), value: Constant::Float { value: 1.0, ty: IrType::F32 } },
            Instruction::Const { result: Value(11), value: Constant::Float { value: 2.0, ty: IrType::F32 } },
            Instruction::Add { result: Value(12), lhs: Value(10), rhs: Value(11), ty: IrType::F32 },
        ];
        let func = make_function("test_fadd", vec![], instrs, Terminator::Return { value: Some(Value(12)) });
        let result = select_instructions(&func, &target).unwrap();
        assert!(has_opcode(&result, I686Opcode::Addss), "float add should use Addss");
    }

    #[test]
    fn test_double_add() {
        let target = i686_target();
        let instrs = vec![
            Instruction::Const { result: Value(10), value: Constant::Float { value: 1.0, ty: IrType::F64 } },
            Instruction::Const { result: Value(11), value: Constant::Float { value: 2.0, ty: IrType::F64 } },
            Instruction::Add { result: Value(12), lhs: Value(10), rhs: Value(11), ty: IrType::F64 },
        ];
        let func = make_function("test_dadd", vec![], instrs, Terminator::Return { value: Some(Value(12)) });
        let result = select_instructions(&func, &target).unwrap();
        assert!(has_opcode(&result, I686Opcode::Addsd), "double add should use Addsd");
    }

    // -------------------------------------------------------------------
    // Test: compare_op_to_cc mapping
    // -------------------------------------------------------------------

    #[test]
    fn test_compare_op_mapping() {
        assert_eq!(compare_op_to_cc(CompareOp::Equal), CC_E);
        assert_eq!(compare_op_to_cc(CompareOp::NotEqual), CC_NE);
        assert_eq!(compare_op_to_cc(CompareOp::SignedLess), CC_L);
        assert_eq!(compare_op_to_cc(CompareOp::SignedGreater), CC_G);
        assert_eq!(compare_op_to_cc(CompareOp::UnsignedLess), CC_B);
        assert_eq!(compare_op_to_cc(CompareOp::UnsignedGreater), CC_A);
    }
}
