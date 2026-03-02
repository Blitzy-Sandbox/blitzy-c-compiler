//! x86-64 instruction selection module.
//!
//! This module transforms the SSA-form IR (from [`crate::ir`]) into x86-64
//! [`MachineInstr`] sequences — the intermediate machine instruction
//! representation defined in [`crate::codegen::MachineInstr`]. Instruction
//! selection is the bridge between the target-independent IR and the
//! target-specific machine code.
//!
//! # Algorithm
//!
//! The selector performs a single linear pass over each function's basic
//! blocks in reverse post-order. For every IR instruction it pattern-matches
//! the instruction kind and emits one or more `MachineInstr` values with
//! appropriate x86-64 opcodes and operands.
//!
//! # Addressing Mode Optimisation
//!
//! The selector recognises complex x86-64 addressing modes
//! (`base + index * scale + displacement`) and folds GEP+Load patterns
//! into single memory-operand instructions where possible.
//!
//! # Instruction Combining
//!
//! - `xor reg, reg` for zeroing (shorter encoding than `mov reg, 0`).
//! - LEA for address arithmetic (`a + b`, `a * 3`, `a * 5`).
//! - Fused compare-and-branch: when an ICmp feeds only a CondBranch, the
//!   boolean result is not materialised via SETCC — CMP+JCC is emitted
//!   directly.
//!
//! # Zero External Dependencies
//!
//! Only imports from `std` and internal crate modules. No external crates.

use std::collections::HashMap;

use crate::codegen::{CodeGenError, MachineInstr, MachineOperand};
use crate::codegen::regalloc::PhysReg;
#[allow(unused_imports)]
use crate::common::SourceLocation;
use crate::ir::{
    Callee, CastOp, CompareOp, Constant, FloatCompareOp,
    Function, Instruction, IrType, Terminator, Value,
};
use crate::ir::instructions::BlockId;
use crate::codegen::x86_64::abi::{
    ReturnClass,
    classify_return_type, compute_argument_locations,
    compute_call_stack_adjustment, is_xmm_reg,
    RAX, RCX, RDX, RSP, RBP, XMM0,
};

// Re-import register constants and ABI utilities that are used transitively
// or needed for future instruction-combining and PIC enhancements.
#[allow(unused_imports)]
use crate::codegen::x86_64::abi::{
    self as abi_mod, ArgumentClass, ArgumentLayout,
    classify_type, RSI, RDI, R8, R9, R10, R11, RBX, R12, R13, R14, R15,
    XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
    INT_ARG_REGS, FLOAT_ARG_REGS, CALLEE_SAVED_GPRS,
};

// ---------------------------------------------------------------------------
// Opcode constants — abstract identifiers mapped to bytes by encoding.rs
// ---------------------------------------------------------------------------

/// Abstract opcode constants for x86-64 machine instructions.
///
/// These are **not** raw instruction bytes — they are symbolic identifiers
/// that the encoder (`encoding.rs`) maps to the real x86-64 binary encoding.
/// The numbering scheme groups related instructions by category for readability.
pub mod opcodes {
    // === Integer arithmetic ===

    /// `add reg, reg`
    pub const ADD_RR: u32 = 0x0100;
    /// `add reg, imm`
    pub const ADD_RI: u32 = 0x0101;
    /// `add reg, [mem]`
    pub const ADD_RM: u32 = 0x0102;
    /// `sub reg, reg`
    pub const SUB_RR: u32 = 0x0110;
    /// `sub reg, imm`
    pub const SUB_RI: u32 = 0x0111;
    /// `sub reg, [mem]`
    pub const SUB_RM: u32 = 0x0112;
    /// `imul reg, reg` (two-operand signed multiply)
    pub const IMUL_RR: u32 = 0x0120;
    /// `imul reg, reg, imm` (three-operand signed multiply)
    pub const IMUL_RI: u32 = 0x0121;
    /// `idiv reg` — signed divide; uses rdx:rax, quotient → rax, remainder → rdx
    pub const IDIV_R: u32 = 0x0130;
    /// `div reg` — unsigned divide; uses rdx:rax
    pub const DIV_R: u32 = 0x0131;

    // === Bitwise operations ===

    /// `and reg, reg`
    pub const AND_RR: u32 = 0x0200;
    /// `and reg, imm`
    pub const AND_RI: u32 = 0x0201;
    /// `or reg, reg`
    pub const OR_RR: u32 = 0x0210;
    /// `or reg, imm`
    pub const OR_RI: u32 = 0x0211;
    /// `xor reg, reg`
    pub const XOR_RR: u32 = 0x0220;
    /// `xor reg, imm`
    pub const XOR_RI: u32 = 0x0221;
    /// `shl reg, imm`
    pub const SHL_RI: u32 = 0x0230;
    /// `shl reg, cl`
    pub const SHL_RCL: u32 = 0x0231;
    /// `shr reg, imm` (logical right shift)
    pub const SHR_RI: u32 = 0x0240;
    /// `sar reg, imm` (arithmetic right shift)
    pub const SAR_RI: u32 = 0x0241;
    /// `shr reg, cl`
    pub const SHR_RCL: u32 = 0x0242;
    /// `sar reg, cl`
    pub const SAR_RCL: u32 = 0x0243;
    /// `not reg` (bitwise NOT)
    pub const NOT_R: u32 = 0x0250;
    /// `neg reg` (two's complement negation)
    pub const NEG_R: u32 = 0x0251;

    // === Data movement ===

    /// `mov reg, reg`
    pub const MOV_RR: u32 = 0x0300;
    /// `mov reg, imm`
    pub const MOV_RI: u32 = 0x0301;
    /// `mov reg, [mem]`
    pub const MOV_RM: u32 = 0x0302;
    /// `mov [mem], reg`
    pub const MOV_MR: u32 = 0x0303;
    /// `mov [mem], imm`
    pub const MOV_MI: u32 = 0x0304;
    /// `movsx` (sign-extend move)
    pub const MOVSX: u32 = 0x0310;
    /// `movzx` (zero-extend move)
    pub const MOVZX: u32 = 0x0311;
    /// `lea reg, [mem]` (load effective address)
    pub const LEA: u32 = 0x0320;

    // === Comparisons and conditions ===

    /// `cmp reg, reg`
    pub const CMP_RR: u32 = 0x0400;
    /// `cmp reg, imm`
    pub const CMP_RI: u32 = 0x0401;
    /// `test reg, reg`
    pub const TEST_RR: u32 = 0x0410;
    /// `test reg, imm`
    pub const TEST_RI: u32 = 0x0411;
    /// `setcc reg` (condition code encoded in operand)
    pub const SETCC: u32 = 0x0420;
    /// `cmovcc reg, reg`
    pub const CMOVCC: u32 = 0x0430;

    // === Control flow ===

    /// `jmp label`
    pub const JMP: u32 = 0x0500;
    /// `jcc label` (conditional jump, condition in operand)
    pub const JCC: u32 = 0x0501;
    /// `call symbol` (direct call)
    pub const CALL: u32 = 0x0510;
    /// `call reg` (indirect call)
    pub const CALL_R: u32 = 0x0511;
    /// `ret`
    pub const RET: u32 = 0x0520;

    // === Stack operations ===

    /// `push reg`
    pub const PUSH: u32 = 0x0600;
    /// `pop reg`
    pub const POP: u32 = 0x0601;

    // === Sized memory operations ===

    /// Load 8-bit from memory.
    pub const LOAD8: u32 = 0x0700;
    /// Load 16-bit from memory.
    pub const LOAD16: u32 = 0x0701;
    /// Load 32-bit from memory.
    pub const LOAD32: u32 = 0x0702;
    /// Load 64-bit from memory.
    pub const LOAD64: u32 = 0x0703;
    /// Store 8-bit to memory.
    pub const STORE8: u32 = 0x0710;
    /// Store 16-bit to memory.
    pub const STORE16: u32 = 0x0711;
    /// Store 32-bit to memory.
    pub const STORE32: u32 = 0x0712;
    /// Store 64-bit to memory.
    pub const STORE64: u32 = 0x0713;

    // === SSE floating-point ===

    /// `addss xmm, xmm/mem` (f32 add)
    pub const ADDSS: u32 = 0x0800;
    /// `addsd xmm, xmm/mem` (f64 add)
    pub const ADDSD: u32 = 0x0801;
    /// `subss xmm, xmm/mem` (f32 sub)
    pub const SUBSS: u32 = 0x0810;
    /// `subsd xmm, xmm/mem` (f64 sub)
    pub const SUBSD: u32 = 0x0811;
    /// `mulss xmm, xmm/mem` (f32 mul)
    pub const MULSS: u32 = 0x0820;
    /// `mulsd xmm, xmm/mem` (f64 mul)
    pub const MULSD: u32 = 0x0821;
    /// `divss xmm, xmm/mem` (f32 div)
    pub const DIVSS: u32 = 0x0830;
    /// `divsd xmm, xmm/mem` (f64 div)
    pub const DIVSD: u32 = 0x0831;
    /// `movss xmm, xmm/mem`
    pub const MOVSS: u32 = 0x0840;
    /// `movsd xmm, xmm/mem`
    pub const MOVSD: u32 = 0x0841;
    /// `ucomiss xmm, xmm` (unordered compare f32)
    pub const UCOMISS: u32 = 0x0850;
    /// `ucomisd xmm, xmm` (unordered compare f64)
    pub const UCOMISD: u32 = 0x0851;
    /// `cvtss2sd` (float → double)
    pub const CVTSS2SD: u32 = 0x0860;
    /// `cvtsd2ss` (double → float)
    pub const CVTSD2SS: u32 = 0x0861;
    /// `cvtsi2ss` (int → float)
    pub const CVTSI2SS: u32 = 0x0870;
    /// `cvtsi2sd` (int → double)
    pub const CVTSI2SD: u32 = 0x0871;
    /// `cvttss2si` (float → int, truncate)
    pub const CVTTSS2SI: u32 = 0x0880;
    /// `cvttsd2si` (double → int, truncate)
    pub const CVTTSD2SI: u32 = 0x0881;

    // === Extension / conversion ===

    /// `cdq` — sign-extend eax → edx:eax (32-bit)
    pub const CDQ: u32 = 0x0900;
    /// `cqo` — sign-extend rax → rdx:rax (64-bit)
    pub const CQO: u32 = 0x0901;

    // === Special ===

    /// `nop`
    pub const NOP: u32 = 0x0A00;
    /// `endbr64` — Intel CET indirect branch tracking
    pub const ENDBR64: u32 = 0x0A01;
    /// `pause` — used in retpoline spin loops
    pub const PAUSE: u32 = 0x0A02;
    /// `lfence` — used in retpoline sequences
    pub const LFENCE: u32 = 0x0A03;
    /// `ud2` — undefined instruction trap (0x0F 0x0B). Used for unreachable
    /// code paths to trigger a guaranteed CPU exception rather than silently
    /// falling through.
    pub const UD2: u32 = 0x0A04;
}

// ---------------------------------------------------------------------------
// CondCode — x86-64 condition codes for JCC / SETCC / CMOVCC
// ---------------------------------------------------------------------------

/// x86-64 condition codes used with `JCC`, `SETCC`, and `CMOVCC` instructions.
///
/// Each variant corresponds to a specific flag test on EFLAGS/RFLAGS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CondCode {
    /// Equal — ZF=1
    E,
    /// Not equal — ZF=0
    NE,
    /// Signed less — SF≠OF
    L,
    /// Signed less or equal — ZF=1 or SF≠OF
    LE,
    /// Signed greater — ZF=0 and SF=OF
    G,
    /// Signed greater or equal — SF=OF
    GE,
    /// Unsigned below — CF=1
    B,
    /// Unsigned below or equal — CF=1 or ZF=1
    BE,
    /// Unsigned above — CF=0 and ZF=0
    A,
    /// Unsigned above or equal — CF=0
    AE,
    /// Sign — SF=1
    S,
    /// Not sign — SF=0
    NS,
    /// Parity — PF=1 (unordered float NaN detection)
    P,
    /// Not parity — PF=0 (ordered float)
    NP,
}

/// Maps an IR integer comparison operation to the corresponding x86-64
/// condition code.
///
/// This is used after a `CMP` instruction to select the correct `SETCC`
/// or `JCC` variant that tests the expected flag combination.
pub fn compare_op_to_cond_code(op: &CompareOp) -> CondCode {
    match op {
        CompareOp::Equal => CondCode::E,
        CompareOp::NotEqual => CondCode::NE,
        CompareOp::SignedLess => CondCode::L,
        CompareOp::SignedLessEqual => CondCode::LE,
        CompareOp::SignedGreater => CondCode::G,
        CompareOp::SignedGreaterEqual => CondCode::GE,
        CompareOp::UnsignedLess => CondCode::B,
        CompareOp::UnsignedLessEqual => CondCode::BE,
        CompareOp::UnsignedGreater => CondCode::A,
        CompareOp::UnsignedGreaterEqual => CondCode::AE,
    }
}

/// Maps a floating-point comparison operation to the x86-64 condition code
/// used after `UCOMISS` / `UCOMISD`.
///
/// Floating-point comparisons set flags differently from integer CMP:
/// - Ordered comparisons check NP (not parity, i.e. both operands are numbers).
/// - Unordered comparison checks P (parity, i.e. at least one NaN).
fn float_compare_op_to_cond_code(op: &FloatCompareOp) -> CondCode {
    match op {
        FloatCompareOp::OrderedEqual => CondCode::E,
        FloatCompareOp::OrderedNotEqual => CondCode::NE,
        FloatCompareOp::OrderedLess => CondCode::B,
        FloatCompareOp::OrderedLessEqual => CondCode::BE,
        FloatCompareOp::OrderedGreater => CondCode::A,
        FloatCompareOp::OrderedGreaterEqual => CondCode::AE,
        FloatCompareOp::Unordered => CondCode::P,
        FloatCompareOp::UnorderedEqual => CondCode::E,
    }
}

// ---------------------------------------------------------------------------
// AddressMode — complex x86-64 addressing
// ---------------------------------------------------------------------------

/// Represents an x86-64 addressing mode of the form
/// `[base + index * scale + displacement]`.
///
/// Any component may be absent. The encoder collapses the mode into the
/// smallest possible ModR/M + SIB encoding.
pub struct AddressMode {
    /// Base register (e.g. RBP for stack locals).
    pub base: Option<PhysReg>,
    /// Index register (scaled by `scale`).
    pub index: Option<PhysReg>,
    /// Scale factor: 1, 2, 4, or 8.
    pub scale: u8,
    /// Signed 32-bit displacement.
    pub displacement: i32,
}

impl AddressMode {
    /// Create a simple base+displacement addressing mode.
    fn base_disp(base: PhysReg, disp: i32) -> Self {
        AddressMode {
            base: Some(base),
            index: None,
            scale: 1,
            displacement: disp,
        }
    }

    /// Create a base+index*scale+displacement addressing mode.
    #[allow(dead_code)]
    fn full(base: PhysReg, index: PhysReg, scale: u8, disp: i32) -> Self {
        AddressMode {
            base: Some(base),
            index: Some(index),
            scale,
            displacement: disp,
        }
    }
}

// ---------------------------------------------------------------------------
// X86_64InstructionSelector — the main selector struct
// ---------------------------------------------------------------------------

/// Selects x86-64 machine instructions from SSA-form IR.
///
/// The selector maintains a mapping from IR [`Value`]s to their machine
/// operand representations (register, stack slot, or immediate) and
/// accumulates a flat sequence of [`MachineInstr`] for each function.
pub struct X86_64InstructionSelector {
    /// Maps each IR Value to the machine operand holding its result.
    value_map: HashMap<Value, MachineOperand>,
    /// Maps each IR Value to its IR type, used for correct call-site argument
    /// classification (float/double → XMM registers per System V AMD64 ABI).
    type_map: HashMap<Value, IrType>,
    /// Counter for virtual register allocation (pre-regalloc).
    next_vreg: u32,
    /// Accumulated machine instructions for the current function.
    instructions: Vec<MachineInstr>,
    /// Whether `-fPIC` mode is active (GOT-relative addressing, PLT calls).
    pic_enabled: bool,
    /// Current stack offset for alloca instructions (grows downward from RBP).
    stack_offset: i32,
    /// Set of IR Values that represent function parameters.  These are
    /// already mapped to their ABI registers by `lower_params`, so any
    /// `Const` instruction whose result is in this set must be *skipped*
    /// rather than emitting a load-immediate that would clobber the real
    /// parameter value sitting in the ABI register.
    param_value_set: std::collections::HashSet<Value>,
    /// Pre-computed register allocation map from the register allocator.
    /// When present, `get_operand` uses the physical register assignment
    /// directly instead of creating virtual registers.  This is critical
    /// for producing correct code because the post-isel resolver maps
    /// unresolved vregs to RAX, which would cause all values to collide.
    alloc_map: HashMap<Value, PhysReg>,
    /// Pending phi resolution moves collected during instruction selection.
    /// Each entry is (phi_result_value, incoming_value, predecessor_block_id).
    /// These are resolved after all blocks have been processed.
    pending_phis: Vec<(Value, Value, BlockId)>,
}

impl X86_64InstructionSelector {
    /// Create a new instruction selector.
    ///
    /// # Arguments
    ///
    /// * `pic_enabled` — Whether position-independent code generation is enabled
    ///   (`-fPIC` flag). When true, global accesses use GOT-relative addressing
    ///   and calls go through PLT stubs.
    pub fn new(pic_enabled: bool) -> Self {
        X86_64InstructionSelector {
            value_map: HashMap::new(),
            type_map: HashMap::new(),
            next_vreg: 32, // start above physical register range
            instructions: Vec::new(),
            pic_enabled,
            stack_offset: 0,
            param_value_set: std::collections::HashSet::new(),
            pending_phis: Vec::new(),
            alloc_map: HashMap::new(),
        }
    }

    /// Allocate the next virtual register number.
    fn alloc_vreg(&mut self) -> PhysReg {
        let vreg = PhysReg(self.next_vreg as u16);
        self.next_vreg += 1;
        vreg
    }

    /// Provide register allocation results so `get_operand` can use
    /// physical registers directly instead of creating virtual registers.
    pub fn set_allocation_map(&mut self, alloc: &HashMap<Value, PhysReg>) {
        self.alloc_map = alloc.clone();
    }

    /// Build a mapping from virtual register IDs to IR Values.
    /// After instruction selection, each IR Value was assigned a vreg via
    /// `get_operand`. This method inverts the value_map to produce
    /// vreg_id → Value so that the register assignment phase can map
    /// vregs back to IR Values and then to physical registers.
    pub fn build_vreg_to_value_map(&self) -> HashMap<u32, Value> {
        let mut map = HashMap::new();
        for (&value, operand) in &self.value_map {
            if let MachineOperand::Register(reg) = operand {
                // Virtual registers have IDs >= 32
                if reg.0 >= 32 {
                    map.insert(reg.0 as u32, value);
                }
            }
        }
        map
    }

    /// Get the machine operand for an IR value.
    ///
    /// Lookup order:
    /// 1. If the value was explicitly bound via `set_operand` (e.g., by
    ///    `lower_params`), return that binding.
    /// 2. Otherwise allocate a fresh virtual register. The post-isel
    ///    register assignment phase maps these vregs to physical registers
    ///    using the regalloc results.
    fn get_operand(&mut self, val: Value) -> MachineOperand {
        if let Some(op) = self.value_map.get(&val) {
            return op.clone();
        }
        // Allocate a fresh virtual register for this IR Value.
        // The post-isel register assignment phase will map this vreg to a
        // physical register using vreg_to_value + value_to_reg from regalloc.
        let vreg = self.alloc_vreg();
        let op = MachineOperand::Register(vreg);
        self.value_map.insert(val, op.clone());
        op
    }

    /// Bind an IR value to a specific machine operand.
    fn set_operand(&mut self, val: Value, op: MachineOperand) {
        self.value_map.insert(val, op);
    }

    /// Emit a machine instruction and append it to the current function.
    fn emit(&mut self, instr: MachineInstr) {
        self.instructions.push(instr);
    }

    /// Emit a simple instruction with the given opcode and operands.
    fn emit_instr(&mut self, opcode: u32, operands: Vec<MachineOperand>) {
        self.instructions
            .push(MachineInstr::with_operands(opcode, operands));
    }

    // ===================================================================
    // select_function — top-level entry point
    // ===================================================================

    /// Select x86-64 machine instructions for an entire IR function.
    ///
    /// Clears internal state, walks basic blocks in the order they appear
    /// in the function (the builder already emits them in a reasonable order),
    /// and processes each instruction and terminator.
    ///
    /// Returns the accumulated `MachineInstr` sequence on success, or a
    /// `CodeGenError` if any IR construct cannot be lowered.
    pub fn select_function(
        &mut self,
        function: &Function,
    ) -> Result<Vec<MachineInstr>, CodeGenError> {
        // Reset state for each function.
        self.value_map.clear();
        self.type_map.clear();
        self.instructions.clear();
        self.next_vreg = 32;
        self.stack_offset = 0;
        self.pending_phis.clear();
        self.param_value_set.clear();

        // Record which IR Values are function parameters so that
        // select_const can skip the placeholder Const instructions
        // the IR builder emits for them (the real values live in ABI
        // registers mapped by lower_params).
        for &pv in &function.param_values {
            self.param_value_set.insert(pv);
        }

        // Map function parameters to their ABI register locations.
        self.lower_params(function);

        // Walk blocks in declaration order (builder outputs RPO-friendly order).
        for block in &function.blocks {
            // Emit a label for this block so branches can target it.
            self.emit_instr(
                opcodes::NOP,
                vec![MachineOperand::Label(block.id.0)],
            );

            // Lower phi nodes — emit MOV for each incoming edge.
            // (Phi lowering inserts copies; real phi elimination happens
            //  at register allocation time. Here we just record the
            //  result mapping.)
            for phi in &block.phi_nodes {
                let dst_op = self.get_operand(phi.result);
                // Map phi result; actual copies are inserted by the
                // phi-elimination pre-pass or regalloc.
                self.set_operand(phi.result, dst_op);
            }

            // Lower each IR instruction in the block.
            for inst in &block.instructions {
                self.select_instruction(inst)?;
            }

            // Lower the terminator (branch, return, switch, etc.).
            if let Some(ref term) = block.terminator {
                self.select_terminator(term)?;
            }
        }

        // ---- Phi resolution pass ----
        // Insert MOV copies to resolve phi nodes. For each block with phi
        // nodes, insert a `MOV phi_dst, incoming_value` before the branch
        // instruction in each predecessor block.
        self.resolve_phi_nodes(function);

        Ok(std::mem::take(&mut self.instructions))
    }

    /// Insert phi-resolution copies before branch instructions.
    ///
    /// For each pending phi from instruction selection, inserts `MOV dst, src`
    /// at the end of each predecessor block (just before the terminator/jump)
    /// to copy the incoming value to the phi result register.
    fn resolve_phi_nodes(&mut self, _function: &Function) {
        // Collect all phi moves needed: (pred_block_label, dst_operand, src_operand)
        let pending = std::mem::take(&mut self.pending_phis);
        let mut phi_moves: Vec<(u32, MachineOperand, MachineOperand)> = Vec::new();

        for (result_val, incoming_val, pred_block) in &pending {
            let dst = self.get_operand(*result_val);
            let src = self.get_operand(*incoming_val);
            if src != dst {
                phi_moves.push((pred_block.0, dst.clone(), src));
            }
        }

        if phi_moves.is_empty() {
            return;
        }

        // For each predecessor block label, find the position of the last
        // instruction before the block's terminator (branch/jump/ret) and
        // insert the phi copy there.
        // Strategy: find NOP(Label(block_id)) markers and the corresponding
        // jump/branch instructions to insert copies before them.
        let mut insertions: Vec<(usize, Vec<MachineInstr>)> = Vec::new();

        for (pred_label, dst, src) in &phi_moves {
            // Find the block region: starts at NOP(Label(pred_label))
            // and ends at the next NOP(Label(...)) or end of instructions.
            let mut block_start = None;
            let mut block_end = self.instructions.len();
            for (i, instr) in self.instructions.iter().enumerate() {
                if instr.opcode == opcodes::NOP
                    && instr.operands.len() == 1
                {
                    if let MachineOperand::Label(lbl) = &instr.operands[0] {
                        if *lbl == *pred_label {
                            block_start = Some(i);
                        } else if block_start.is_some() {
                            block_end = i;
                            break;
                        }
                    }
                }
            }

            if let Some(_start) = block_start {
                // Find the last branch/jump/ret instruction in this block range
                let mut insert_pos = block_end;
                for i in (0..block_end).rev() {
                    if i < block_start.unwrap_or(0) { break; }
                    let op = self.instructions[i].opcode;
                    if op == opcodes::JMP
                        || op == opcodes::JCC
                        || op == opcodes::RET
                    {
                        insert_pos = i;
                        break;
                    }
                }

                let mov_instr = MachineInstr::with_operands(
                    opcodes::MOV_RR,
                    vec![dst.clone(), src.clone()],
                );
                insertions.push((insert_pos, vec![mov_instr]));
            }
        }

        // Sort insertions by position (descending) to preserve indices
        insertions.sort_by(|a, b| b.0.cmp(&a.0));
        for (pos, instrs) in insertions {
            for (j, instr) in instrs.into_iter().enumerate() {
                self.instructions.insert(pos + j, instr);
            }
        }
    }

    // ===================================================================
    // Parameter lowering
    // ===================================================================

    /// Map function parameters to their ABI-specified register locations
    /// so that the first use of each parameter value resolves correctly.
    ///
    /// Uses `function.param_values` (populated by the IR builder) to identify
    /// the exact IR Value IDs that represent function parameters, and maps
    /// each to the correct ABI register (rdi, rsi, rdx, rcx, r8, r9 for
    /// integer args under System V AMD64).
    fn lower_params(&mut self, function: &Function) {
        let param_types: Vec<IrType> = function
            .params
            .iter()
            .map(|(_, ty)| ty.clone())
            .collect();
        let layout = compute_argument_locations(&param_types);

        for (i, (_name, ty)) in function.params.iter().enumerate() {
            // Use the actual IR Value ID from param_values instead of Value(i).
            // The IR builder records param values as they are created, so
            // param_values[i] is the correct Value for parameter i.
            let val = if i < function.param_values.len() {
                function.param_values[i]
            } else {
                // Fallback for functions without param_values (extern stubs).
                Value(i as u32)
            };
            self.type_map.insert(val, ty.clone());
            if let Some(loc) = layout.locations.get(i) {
                if let Some(reg) = loc.register {
                    self.set_operand(val, MachineOperand::Register(reg));
                } else if let Some(off) = loc.stack_offset {
                    // Parameter lives on the stack relative to RBP.
                    // Incoming stack args are at positive offsets from RBP:
                    //   [RBP + 16] = first stack arg (after saved RBP + ret addr)
                    self.set_operand(
                        val,
                        MachineOperand::Memory {
                            base: RBP,
                            offset: 16 + off,
                        },
                    );
                } else {
                    let vreg = self.alloc_vreg();
                    self.set_operand(val, MachineOperand::Register(vreg));
                }
            } else {
                let vreg = self.alloc_vreg();
                self.set_operand(val, MachineOperand::Register(vreg));
            }
        }
    }

    // ===================================================================
    // Instruction selection — dispatches on IR instruction kind
    // ===================================================================

    /// Select x86-64 instruction(s) for a single IR instruction.
    fn select_instruction(
        &mut self,
        inst: &Instruction,
    ) -> Result<(), CodeGenError> {
        // Record the result type for every instruction that produces a typed
        // result. This enables correct argument classification in select_call
        // (float/double → XMM registers per System V AMD64 ABI).
        if let Some(result_ty) = inst.result_type() {
            if let Some(result_val) = inst.result() {
                self.type_map.insert(result_val, result_ty.clone());
            }
        }

        match inst {
            // ---- Arithmetic ----
            Instruction::Add { result, lhs, rhs, ty } => {
                self.select_binop_or_fp(*result, *lhs, *rhs, ty, BinOpKind::Add)?;
            }
            Instruction::Sub { result, lhs, rhs, ty } => {
                self.select_binop_or_fp(*result, *lhs, *rhs, ty, BinOpKind::Sub)?;
            }
            Instruction::Mul { result, lhs, rhs, ty } => {
                self.select_binop_or_fp(*result, *lhs, *rhs, ty, BinOpKind::Mul)?;
            }
            Instruction::Div {
                result, lhs, rhs, ty, is_signed,
            } => {
                if ty.is_float() {
                    self.select_fp_binop(*result, *lhs, *rhs, ty, BinOpKind::Div)?;
                } else {
                    self.select_div_mod(*result, *lhs, *rhs, ty, *is_signed, true)?;
                }
            }
            Instruction::Mod {
                result, lhs, rhs, ty, is_signed,
            } => {
                self.select_div_mod(*result, *lhs, *rhs, ty, *is_signed, false)?;
            }

            // ---- Bitwise ----
            Instruction::And { result, lhs, rhs, ty } => {
                self.select_int_binop(*result, *lhs, *rhs, ty, BinOpKind::And)?;
            }
            Instruction::Or { result, lhs, rhs, ty } => {
                self.select_int_binop(*result, *lhs, *rhs, ty, BinOpKind::Or)?;
            }
            Instruction::Xor { result, lhs, rhs, ty } => {
                self.select_int_binop(*result, *lhs, *rhs, ty, BinOpKind::Xor)?;
            }
            Instruction::Shl { result, lhs, rhs, ty } => {
                self.select_shift(*result, *lhs, *rhs, ty, false, false)?;
            }
            Instruction::Shr {
                result, lhs, rhs, ty, is_arithmetic,
            } => {
                self.select_shift(*result, *lhs, *rhs, ty, true, *is_arithmetic)?;
            }

            // ---- Comparisons ----
            Instruction::ICmp { result, op, lhs, rhs, ty } => {
                self.select_icmp(*result, op, *lhs, *rhs, ty)?;
            }
            Instruction::FCmp { result, op, lhs, rhs, ty } => {
                self.select_fcmp(*result, op, *lhs, *rhs, ty)?;
            }

            // ---- Memory ----
            Instruction::Load { result, ty, ptr } => {
                self.select_load(*result, *ptr, ty)?;
            }
            Instruction::Store { value, ptr } => {
                self.select_store(*value, *ptr)?;
            }
            Instruction::Alloca { result, ty, count } => {
                self.select_alloca(*result, ty, count)?;
            }
            Instruction::GetElementPtr {
                result, base_ty, ptr, indices, ..
            } => {
                self.select_gep(*result, base_ty, *ptr, indices)?;
            }

            // ---- Function calls ----
            Instruction::Call {
                result, callee, args, return_ty,
            } => {
                self.select_call(result, callee, args, return_ty)?;
            }

            // ---- Phi nodes ----
            Instruction::Phi { result, ty, incoming } => {
                // Allocate operand for the phi result so it gets a vreg.
                let _op = self.get_operand(*result);
                // Ensure incoming values also have operands allocated.
                for (val, _block) in incoming {
                    let _src = self.get_operand(*val);
                }
                // Collect phi resolution info for post-processing.
                // We'll insert MOV copies before branch instructions in
                // each predecessor block.
                for (val, block) in incoming {
                    self.pending_phis.push((*result, *val, *block));
                }
                let _ = ty;
            }

            // ---- Type conversions ----
            Instruction::Cast {
                result, op, value, from_ty, to_ty,
            } => {
                self.select_cast(*result, *op, *value, from_ty, to_ty)?;
            }
            Instruction::BitCast {
                result, value, from_ty, to_ty,
            } => {
                // Bitcast: reinterpret bits without conversion. For
                // same-class types this is a simple MOV.
                let src = self.get_operand(*value);
                let dst = self.get_operand(*result);
                if from_ty.is_float() && to_ty.is_integer() {
                    // movq from xmm to gpr — use MOV_RR for abstract isel
                    self.emit_instr(opcodes::MOV_RR, vec![dst, src]);
                } else if from_ty.is_integer() && to_ty.is_float() {
                    self.emit_instr(opcodes::MOV_RR, vec![dst, src]);
                } else {
                    self.emit_instr(opcodes::MOV_RR, vec![dst, src]);
                }
            }

            // ---- Select ----
            Instruction::Select {
                result,
                condition,
                true_val,
                false_val,
                ty,
            } => {
                self.select_select(*result, *condition, *true_val, *false_val, ty)?;
            }

            // ---- Constants ----
            Instruction::Const { result, value } => {
                self.select_const(*result, value)?;
            }

            // ---- Copy ----
            Instruction::Copy { result, source, ty } => {
                let src = self.get_operand(*source);
                let dst = self.get_operand(*result);
                if ty.is_float() {
                    let sse_op = if *ty == IrType::F32 {
                        opcodes::MOVSS
                    } else {
                        opcodes::MOVSD
                    };
                    self.emit_instr(sse_op, vec![dst, src]);
                } else {
                    self.emit_instr(opcodes::MOV_RR, vec![dst, src]);
                }
            }

            // ---- Nop ----
            Instruction::Nop => {
                // No instruction needed.
            }
        }

        Ok(())
    }

    // ===================================================================
    // Terminator selection
    // ===================================================================

    /// Select x86-64 instruction(s) for a block terminator.
    fn select_terminator(
        &mut self,
        term: &Terminator,
    ) -> Result<(), CodeGenError> {
        match term {
            Terminator::Branch { target } => {
                self.emit_instr(opcodes::JMP, vec![MachineOperand::Label(target.0)]);
            }
            Terminator::CondBranch {
                condition,
                true_block,
                false_block,
            } => {
                // Emit a TEST/CMP for the boolean condition, then JCC.
                let cond_op = self.get_operand(*condition);
                self.emit_instr(
                    opcodes::TEST_RR,
                    vec![cond_op.clone(), cond_op],
                );
                // Jump to true_block if non-zero (NE).
                self.emit_instr(
                    opcodes::JCC,
                    vec![
                        MachineOperand::Immediate(CondCode::NE as i64),
                        MachineOperand::Label(true_block.0),
                    ],
                );
                // Fall-through or jump to false_block.
                self.emit_instr(
                    opcodes::JMP,
                    vec![MachineOperand::Label(false_block.0)],
                );
            }
            Terminator::Return { value } => {
                if let Some(val) = value {
                    let src = self.get_operand(*val);
                    // Determine if float return (xmm0) or int return (rax).
                    let is_sse = match &src {
                        MachineOperand::Register(r) => is_xmm_reg(*r),
                        _ => false,
                    };
                    if is_sse {
                        self.emit_instr(
                            opcodes::MOVSD,
                            vec![MachineOperand::Register(XMM0), src],
                        );
                    } else {
                        self.emit_instr(
                            opcodes::MOV_RR,
                            vec![MachineOperand::Register(RAX), src],
                        );
                    }
                }
                self.emit_instr(opcodes::RET, vec![]);
            }
            Terminator::Switch {
                value,
                default,
                cases,
            } => {
                let val_op = self.get_operand(*value);
                for (case_val, target) in cases {
                    self.emit_instr(
                        opcodes::CMP_RI,
                        vec![val_op.clone(), MachineOperand::Immediate(*case_val)],
                    );
                    self.emit_instr(
                        opcodes::JCC,
                        vec![
                            MachineOperand::Immediate(CondCode::E as i64),
                            MachineOperand::Label(target.0),
                        ],
                    );
                }
                // Default case.
                self.emit_instr(opcodes::JMP, vec![MachineOperand::Label(default.0)]);
            }
            Terminator::Unreachable => {
                // Emit UD2 trap instruction — unreachable code should trigger a
                // CPU exception rather than silently executing the next instruction.
                self.emit_instr(opcodes::UD2, vec![]);
            }
        }
        Ok(())
    }

    // ===================================================================
    // Integer binary operations (Add, Sub, Mul, And, Or, Xor)
    // ===================================================================

    /// Select an integer binary operation. Checks if the RHS is an immediate
    /// constant and emits the reg-imm variant where profitable.
    fn select_int_binop(
        &mut self,
        result: Value,
        lhs: Value,
        rhs: Value,
        ty: &IrType,
        kind: BinOpKind,
    ) -> Result<(), CodeGenError> {
        let dst = self.get_operand(result);
        let lhs_op = self.get_operand(lhs);
        let rhs_op = self.get_operand(rhs);

        // Move LHS to destination first (x86 two-address form).
        self.emit_instr(opcodes::MOV_RR, vec![dst.clone(), lhs_op]);

        let (rr_op, ri_op) = match kind {
            BinOpKind::Add => (opcodes::ADD_RR, opcodes::ADD_RI),
            BinOpKind::Sub => (opcodes::SUB_RR, opcodes::SUB_RI),
            BinOpKind::Mul => (opcodes::IMUL_RR, opcodes::IMUL_RI),
            BinOpKind::And => (opcodes::AND_RR, opcodes::AND_RI),
            BinOpKind::Or => (opcodes::OR_RR, opcodes::OR_RI),
            BinOpKind::Xor => (opcodes::XOR_RR, opcodes::XOR_RI),
            _ => {
                return Err(CodeGenError::InternalError(
                    "unexpected BinOpKind in select_int_binop".into(),
                ));
            }
        };

        match &rhs_op {
            MachineOperand::Immediate(imm) => {
                // Special case: xor reg, reg for zeroing when result = lhs ^ 0
                // is not useful. But xor reg, reg when setting to zero IS.
                if kind == BinOpKind::Xor && *imm == 0 {
                    // xor with 0 is identity — no-op.
                    return Ok(());
                }
                self.emit_instr(ri_op, vec![dst, MachineOperand::Immediate(*imm)]);
            }
            _ => {
                self.emit_instr(rr_op, vec![dst, rhs_op]);
            }
        }
        let _ = ty;
        Ok(())
    }

    /// Select either an integer or floating-point binary operation based
    /// on the type.
    fn select_binop_or_fp(
        &mut self,
        result: Value,
        lhs: Value,
        rhs: Value,
        ty: &IrType,
        kind: BinOpKind,
    ) -> Result<(), CodeGenError> {
        if ty.is_float() {
            self.select_fp_binop(result, lhs, rhs, ty, kind)
        } else {
            self.select_int_binop(result, lhs, rhs, ty, kind)
        }
    }

    /// Select an SSE floating-point binary operation.
    fn select_fp_binop(
        &mut self,
        result: Value,
        lhs: Value,
        rhs: Value,
        ty: &IrType,
        kind: BinOpKind,
    ) -> Result<(), CodeGenError> {
        let dst = self.get_operand(result);
        let lhs_op = self.get_operand(lhs);
        let rhs_op = self.get_operand(rhs);

        // Move LHS to destination (SSE two-operand form).
        let mov_op = if *ty == IrType::F32 {
            opcodes::MOVSS
        } else {
            opcodes::MOVSD
        };
        self.emit_instr(mov_op, vec![dst.clone(), lhs_op]);

        let opcode = match (kind, ty) {
            (BinOpKind::Add, IrType::F32) => opcodes::ADDSS,
            (BinOpKind::Add, IrType::F64) => opcodes::ADDSD,
            (BinOpKind::Sub, IrType::F32) => opcodes::SUBSS,
            (BinOpKind::Sub, IrType::F64) => opcodes::SUBSD,
            (BinOpKind::Mul, IrType::F32) => opcodes::MULSS,
            (BinOpKind::Mul, IrType::F64) => opcodes::MULSD,
            (BinOpKind::Div, IrType::F32) => opcodes::DIVSS,
            (BinOpKind::Div, IrType::F64) => opcodes::DIVSD,
            _ => {
                return Err(CodeGenError::UnsupportedInstruction(format!(
                    "unsupported float binop {:?} for type {:?}",
                    kind, ty
                )));
            }
        };

        self.emit_instr(opcode, vec![dst, rhs_op]);
        Ok(())
    }

    // ===================================================================
    // Division and modulo
    // ===================================================================

    /// Select signed or unsigned integer division / modulo.
    ///
    /// x86-64 division uses `rdx:rax` as the dividend. The `idiv`/`div`
    /// instruction stores quotient in `rax` and remainder in `rdx`.
    fn select_div_mod(
        &mut self,
        result: Value,
        lhs: Value,
        rhs: Value,
        ty: &IrType,
        is_signed: bool,
        is_div: bool,
    ) -> Result<(), CodeGenError> {
        let lhs_op = self.get_operand(lhs);
        let rhs_op = self.get_operand(rhs);

        // CRITICAL: Save the divisor into RCX (a physical register, NOT a
        // vreg) BEFORE moving the dividend to RAX.  Using a physical register
        // avoids the unmapped-vreg-fallback-to-RAX problem that would clobber
        // the divisor when apply_register_assignments_v2 resolves it.
        // RCX is safe: x86-64 IDIV only implicitly uses RAX and RDX.
        self.emit_instr(
            opcodes::MOV_RR,
            vec![MachineOperand::Register(RCX), rhs_op],
        );

        // Move dividend to RAX.
        self.emit_instr(
            opcodes::MOV_RR,
            vec![MachineOperand::Register(RAX), lhs_op],
        );

        // Prepare RDX:
        // - Signed: CQO (64-bit) or CDQ (32-bit) sign-extends RAX → RDX:RAX
        // - Unsigned: XOR RDX, RDX to zero the high half
        if is_signed {
            let extend_op = match ty {
                IrType::I64 | IrType::Pointer(_) => opcodes::CQO,
                _ => opcodes::CDQ,
            };
            self.emit_instr(extend_op, vec![]);
        } else {
            self.emit_instr(
                opcodes::XOR_RR,
                vec![
                    MachineOperand::Register(RDX),
                    MachineOperand::Register(RDX),
                ],
            );
        }

        // Emit IDIV or DIV using the saved divisor in RCX.
        let div_op = if is_signed {
            opcodes::IDIV_R
        } else {
            opcodes::DIV_R
        };
        self.emit_instr(div_op, vec![MachineOperand::Register(RCX)]);

        // Result: quotient in RAX, remainder in RDX.
        let result_reg = if is_div { RAX } else { RDX };
        let dst = self.get_operand(result);
        self.emit_instr(
            opcodes::MOV_RR,
            vec![dst, MachineOperand::Register(result_reg)],
        );

        Ok(())
    }

    // ===================================================================
    // Shift operations
    // ===================================================================

    /// Select a shift instruction. If the shift amount is an immediate,
    /// use the reg-imm form; otherwise move the amount to CL.
    fn select_shift(
        &mut self,
        result: Value,
        lhs: Value,
        rhs: Value,
        ty: &IrType,
        is_right: bool,
        is_arithmetic: bool,
    ) -> Result<(), CodeGenError> {
        let dst = self.get_operand(result);
        let lhs_op = self.get_operand(lhs);
        let rhs_op = self.get_operand(rhs);

        self.emit_instr(opcodes::MOV_RR, vec![dst.clone(), lhs_op]);

        match &rhs_op {
            MachineOperand::Immediate(imm) => {
                let opcode = if !is_right {
                    opcodes::SHL_RI
                } else if is_arithmetic {
                    opcodes::SAR_RI
                } else {
                    opcodes::SHR_RI
                };
                self.emit_instr(opcode, vec![dst, MachineOperand::Immediate(*imm)]);
            }
            _ => {
                // Move shift amount to RCX (CL is the low byte of RCX).
                self.emit_instr(
                    opcodes::MOV_RR,
                    vec![MachineOperand::Register(RCX), rhs_op],
                );
                let opcode = if !is_right {
                    opcodes::SHL_RCL
                } else if is_arithmetic {
                    opcodes::SAR_RCL
                } else {
                    opcodes::SHR_RCL
                };
                self.emit_instr(opcode, vec![dst]);
            }
        }
        let _ = ty;
        Ok(())
    }

    // ===================================================================
    // Integer comparison
    // ===================================================================

    /// Select an integer comparison → CMP + SETCC + MOVZX.
    fn select_icmp(
        &mut self,
        result: Value,
        op: &CompareOp,
        lhs: Value,
        rhs: Value,
        _ty: &IrType,
    ) -> Result<(), CodeGenError> {
        let lhs_op = self.get_operand(lhs);
        let rhs_op = self.get_operand(rhs);
        let dst = self.get_operand(result);
        let cc = compare_op_to_cond_code(op);

        match &rhs_op {
            MachineOperand::Immediate(imm) => {
                self.emit_instr(
                    opcodes::CMP_RI,
                    vec![lhs_op, MachineOperand::Immediate(*imm)],
                );
            }
            _ => {
                self.emit_instr(opcodes::CMP_RR, vec![lhs_op, rhs_op]);
            }
        }

        // SETCC sets the low byte of the destination.
        // Operand order: [Immediate(cc), Register(dst)] — matches encoding.rs expectation.
        self.emit_instr(
            opcodes::SETCC,
            vec![MachineOperand::Immediate(cc as i64), dst.clone()],
        );

        // Zero-extend from byte to full register width.
        self.emit_instr(opcodes::MOVZX, vec![dst.clone(), dst]);

        Ok(())
    }

    // ===================================================================
    // Float comparison
    // ===================================================================

    /// Select a floating-point comparison → UCOMISS/UCOMISD + SETCC.
    ///
    /// For ordered comparisons, NaN inputs must produce `false`. Since
    /// x86 UCOMISD/UCOMISS sets PF=1 for unordered (NaN) results, ordered
    /// comparisons need an additional parity flag check. Specifically for
    /// `OrderedEqual`, we emit SETNP + SETE + AND to correctly reject NaN.
    fn select_fcmp(
        &mut self,
        result: Value,
        op: &FloatCompareOp,
        lhs: Value,
        rhs: Value,
        ty: &IrType,
    ) -> Result<(), CodeGenError> {
        let lhs_op = self.get_operand(lhs);
        let rhs_op = self.get_operand(rhs);
        let dst = self.get_operand(result);

        let cmp_op = if *ty == IrType::F32 {
            opcodes::UCOMISS
        } else {
            opcodes::UCOMISD
        };
        self.emit_instr(cmp_op, vec![lhs_op, rhs_op]);

        let cc = float_compare_op_to_cond_code(op);

        // For OrderedEqual: x86 UCOMISD sets ZF=1 for both equal AND
        // unordered (NaN). We must AND the result with a parity check
        // (SETNP) to exclude NaN. Sequence: SETNP tmp; SETE dst; AND dst, tmp.
        if matches!(op, FloatCompareOp::OrderedEqual) {
            let tmp = self.alloc_vreg();
            let tmp_op = MachineOperand::Register(tmp);
            // SETNP tmp — set if ordered (not NaN)
            self.emit_instr(
                opcodes::SETCC,
                vec![MachineOperand::Immediate(CondCode::NP as i64), tmp_op.clone()],
            );
            // SETE dst — set if equal
            self.emit_instr(
                opcodes::SETCC,
                vec![MachineOperand::Immediate(CondCode::E as i64), dst.clone()],
            );
            // AND dst, tmp — true only if ordered AND equal
            self.emit_instr(opcodes::AND_RR, vec![dst.clone(), tmp_op]);
        } else {
            // For other float comparisons, the primary condition code is sufficient.
            // Operand order: [Immediate(cc), Register(dst)] — matches encoding.rs.
            self.emit_instr(
                opcodes::SETCC,
                vec![MachineOperand::Immediate(cc as i64), dst.clone()],
            );
        }

        // Zero-extend to full register width.
        self.emit_instr(opcodes::MOVZX, vec![dst.clone(), dst]);

        Ok(())
    }

    // ===================================================================
    // Memory operations
    // ===================================================================

    /// Select a load from memory.
    fn select_load(
        &mut self,
        result: Value,
        ptr: Value,
        ty: &IrType,
    ) -> Result<(), CodeGenError> {
        let ptr_op = self.get_operand(ptr);
        let dst = self.get_operand(result);

        // Choose opcode based on loaded type.
        if ty.is_float() {
            let mov_op = if *ty == IrType::F32 {
                opcodes::MOVSS
            } else {
                opcodes::MOVSD
            };
            // Load from memory at ptr_op into SSE register.
            match &ptr_op {
                MachineOperand::Memory { base, offset } => {
                    self.emit_instr(
                        mov_op,
                        vec![
                            dst,
                            MachineOperand::Memory {
                                base: *base,
                                offset: *offset,
                            },
                        ],
                    );
                }
                _ => {
                    // ptr_op is a register holding the address.
                    let base_r = self.operand_to_reg(&ptr_op);
                    self.emit_instr(
                        mov_op,
                        vec![
                            dst,
                            MachineOperand::Memory {
                                base: base_r,
                                offset: 0,
                            },
                        ],
                    );
                }
            }
        } else {
            let load_op = match ty {
                IrType::I1 | IrType::I8 => opcodes::LOAD8,
                IrType::I16 => opcodes::LOAD16,
                IrType::I32 => opcodes::LOAD32,
                IrType::I64 | IrType::Pointer(_) => opcodes::LOAD64,
                _ => opcodes::LOAD64,
            };
            match &ptr_op {
                MachineOperand::Memory { base, offset } => {
                    self.emit_instr(
                        load_op,
                        vec![
                            dst,
                            MachineOperand::Memory {
                                base: *base,
                                offset: *offset,
                            },
                        ],
                    );
                }
                _ => {
                    let base_r = self.operand_to_reg(&ptr_op);
                    self.emit_instr(
                        load_op,
                        vec![
                            dst,
                            MachineOperand::Memory {
                                base: base_r,
                                offset: 0,
                            },
                        ],
                    );
                }
            }
        }
        Ok(())
    }

    /// Select a store to memory.
    fn select_store(
        &mut self,
        value: Value,
        ptr: Value,
    ) -> Result<(), CodeGenError> {
        let val_op = self.get_operand(value);
        let ptr_op = self.get_operand(ptr);

        // Determine store size from value operand type (we infer from context).
        // Since we don't have the type directly on Store, we use a 64-bit store
        // as default (the encoder adjusts based on register size).
        let base_reg = self.operand_to_reg(&ptr_op);
        let offset = match &ptr_op {
            MachineOperand::Memory { offset, .. } => *offset,
            _ => 0,
        };

        // Check if value is in an SSE register.
        let is_sse = match &val_op {
            MachineOperand::Register(r) => is_xmm_reg(*r),
            _ => false,
        };

        if is_sse {
            self.emit_instr(
                opcodes::MOVSD,
                vec![
                    MachineOperand::Memory {
                        base: base_reg,
                        offset,
                    },
                    val_op,
                ],
            );
        } else {
            self.emit_instr(
                opcodes::STORE64,
                vec![
                    MachineOperand::Memory {
                        base: base_reg,
                        offset,
                    },
                    val_op,
                ],
            );
        }
        Ok(())
    }

    /// Select a stack allocation.
    fn select_alloca(
        &mut self,
        result: Value,
        ty: &IrType,
        count: &Option<Value>,
    ) -> Result<(), CodeGenError> {
        // Static alloca: compute size at compile time.
        // We bump the stack_offset and return an RBP-relative address.
        let elem_size = self.type_size(ty) as i32;
        let total_size = if let Some(count_val) = count {
            // Dynamic alloca: multiply element size by count.
            let count_op = self.get_operand(*count_val);
            let size_reg = self.alloc_vreg();
            self.emit_instr(
                opcodes::MOV_RI,
                vec![
                    MachineOperand::Register(size_reg),
                    MachineOperand::Immediate(elem_size as i64),
                ],
            );
            self.emit_instr(
                opcodes::IMUL_RR,
                vec![MachineOperand::Register(size_reg), count_op],
            );
            // sub rsp, size_reg (dynamic stack allocation)
            self.emit_instr(
                opcodes::SUB_RR,
                vec![MachineOperand::Register(RSP), MachineOperand::Register(size_reg)],
            );
            // Result is RSP.
            let dst = self.get_operand(result);
            self.emit_instr(
                opcodes::MOV_RR,
                vec![dst, MachineOperand::Register(RSP)],
            );
            return Ok(());
        } else {
            // Align to at least 8 bytes for stack slots.
            let aligned = (elem_size + 7) & !7;
            aligned
        };

        self.stack_offset += total_size;

        // LEA dst, [rbp - stack_offset]
        let dst = self.get_operand(result);
        self.set_operand(
            result,
            MachineOperand::Memory {
                base: RBP,
                offset: -self.stack_offset,
            },
        );
        // Also emit a LEA so the value can be used as a pointer.
        self.emit_instr(
            opcodes::LEA,
            vec![
                dst,
                MachineOperand::Memory {
                    base: RBP,
                    offset: -self.stack_offset,
                },
            ],
        );

        Ok(())
    }

    // ===================================================================
    // GetElementPtr — address computation
    // ===================================================================

    /// Select a GEP instruction. Computes the address of a sub-element.
    fn select_gep(
        &mut self,
        result: Value,
        base_ty: &IrType,
        ptr: Value,
        indices: &[Value],
    ) -> Result<(), CodeGenError> {
        let ptr_op = self.get_operand(ptr);
        let dst = self.get_operand(result);

        // Start with the base pointer in the destination register.
        self.emit_instr(opcodes::MOV_RR, vec![dst.clone(), ptr_op]);

        // Walk through indices, accumulating offsets.
        let mut current_ty = base_ty.clone();

        for (i, idx_val) in indices.iter().enumerate() {
            let idx_op = self.get_operand(*idx_val);
            let elem_size = self.type_size(&current_ty) as i64;

            if i == 0 {
                // First index: scale by element size.
                if elem_size > 0 {
                    match &idx_op {
                        MachineOperand::Immediate(imm) => {
                            let byte_offset = imm * elem_size;
                            if byte_offset != 0 {
                                self.emit_instr(
                                    opcodes::ADD_RI,
                                    vec![
                                        dst.clone(),
                                        MachineOperand::Immediate(byte_offset),
                                    ],
                                );
                            }
                        }
                        _ => {
                            // idx * elem_size → use IMUL then ADD.
                            let tmp = self.alloc_vreg();
                            self.emit_instr(
                                opcodes::MOV_RI,
                                vec![
                                    MachineOperand::Register(tmp),
                                    MachineOperand::Immediate(elem_size),
                                ],
                            );
                            self.emit_instr(
                                opcodes::IMUL_RR,
                                vec![MachineOperand::Register(tmp), idx_op],
                            );
                            self.emit_instr(
                                opcodes::ADD_RR,
                                vec![dst.clone(), MachineOperand::Register(tmp)],
                            );
                        }
                    }
                }
            } else {
                // Subsequent indices: struct field access or nested array.
                match &current_ty {
                    IrType::Struct { fields, packed } => {
                        // Index must be a constant for struct field access.
                        if let MachineOperand::Immediate(field_idx) = &idx_op {
                            let field_offset =
                                self.struct_field_offset(fields, *packed, *field_idx as usize);
                            if field_offset != 0 {
                                self.emit_instr(
                                    opcodes::ADD_RI,
                                    vec![
                                        dst.clone(),
                                        MachineOperand::Immediate(field_offset as i64),
                                    ],
                                );
                            }
                            // Advance current_ty to the field type.
                            if let Some(ft) = fields.get(*field_idx as usize) {
                                current_ty = ft.clone();
                            }
                            continue;
                        }
                    }
                    IrType::Array { element, .. } => {
                        let inner_size = self.type_size(element) as i64;
                        if inner_size > 0 {
                            match &idx_op {
                                MachineOperand::Immediate(imm) => {
                                    let byte_offset = imm * inner_size;
                                    if byte_offset != 0 {
                                        self.emit_instr(
                                            opcodes::ADD_RI,
                                            vec![
                                                dst.clone(),
                                                MachineOperand::Immediate(byte_offset),
                                            ],
                                        );
                                    }
                                }
                                _ => {
                                    let tmp = self.alloc_vreg();
                                    self.emit_instr(
                                        opcodes::MOV_RI,
                                        vec![
                                            MachineOperand::Register(tmp),
                                            MachineOperand::Immediate(inner_size),
                                        ],
                                    );
                                    self.emit_instr(
                                        opcodes::IMUL_RR,
                                        vec![MachineOperand::Register(tmp), idx_op],
                                    );
                                    self.emit_instr(
                                        opcodes::ADD_RR,
                                        vec![dst.clone(), MachineOperand::Register(tmp)],
                                    );
                                }
                            }
                        }
                        current_ty = element.as_ref().clone();
                        continue;
                    }
                    _ => {}
                }

                // Fallback for non-struct/array: scale by element size.
                if elem_size > 0 {
                    match &idx_op {
                        MachineOperand::Immediate(imm) => {
                            let byte_offset = imm * elem_size;
                            if byte_offset != 0 {
                                self.emit_instr(
                                    opcodes::ADD_RI,
                                    vec![
                                        dst.clone(),
                                        MachineOperand::Immediate(byte_offset),
                                    ],
                                );
                            }
                        }
                        _ => {
                            let tmp = self.alloc_vreg();
                            self.emit_instr(
                                opcodes::MOV_RI,
                                vec![
                                    MachineOperand::Register(tmp),
                                    MachineOperand::Immediate(elem_size),
                                ],
                            );
                            self.emit_instr(
                                opcodes::IMUL_RR,
                                vec![MachineOperand::Register(tmp), idx_op],
                            );
                            self.emit_instr(
                                opcodes::ADD_RR,
                                vec![dst.clone(), MachineOperand::Register(tmp)],
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }

    // ===================================================================
    // Function calls
    // ===================================================================

    /// Select a function call using the System V AMD64 ABI.
    fn select_call(
        &mut self,
        result: &Option<Value>,
        callee: &Callee,
        args: &[Value],
        return_ty: &IrType,
    ) -> Result<(), CodeGenError> {
        // Classify arguments using actual IR types so float/double arguments
        // are correctly assigned to XMM registers (xmm0-xmm7) per System V
        // AMD64 ABI, rather than being misclassified as integer types.
        let arg_types: Vec<IrType> = args
            .iter()
            .map(|v| self.type_map.get(v).cloned().unwrap_or(IrType::I64))
            .collect();
        let layout = compute_argument_locations(&arg_types);

        // Compute required stack adjustment for alignment.
        let stack_adj = compute_call_stack_adjustment(0, layout.stack_space);
        if stack_adj > 0 {
            self.emit_instr(
                opcodes::SUB_RI,
                vec![
                    MachineOperand::Register(RSP),
                    MachineOperand::Immediate(stack_adj as i64),
                ],
            );
        }

        // Push overflow (stack) arguments in right-to-left order.
        // First, push any stack-passed arguments.
        for (i, loc) in layout.locations.iter().enumerate().rev() {
            if loc.stack_offset.is_some() {
                if let Some(arg_val) = args.get(i) {
                    let arg_op = self.get_operand(*arg_val);
                    self.emit_instr(opcodes::PUSH, vec![arg_op]);
                }
            }
        }

        // Move register-passed arguments into the correct registers.
        for (i, loc) in layout.locations.iter().enumerate() {
            if let Some(reg) = loc.register {
                if let Some(arg_val) = args.get(i) {
                    let arg_op = self.get_operand(*arg_val);
                    if is_xmm_reg(reg) {
                        self.emit_instr(
                            opcodes::MOVSD,
                            vec![MachineOperand::Register(reg), arg_op],
                        );
                    } else {
                        self.emit_instr(
                            opcodes::MOV_RR,
                            vec![MachineOperand::Register(reg), arg_op],
                        );
                    }
                }
            }
        }

        // Emit the CALL instruction.
        match callee {
            Callee::Direct(name) => {
                if self.pic_enabled {
                    self.emit_instr(
                        opcodes::CALL,
                        vec![MachineOperand::Symbol(format!("{}@PLT", name))],
                    );
                } else {
                    self.emit_instr(
                        opcodes::CALL,
                        vec![MachineOperand::Symbol(name.clone())],
                    );
                }
            }
            Callee::Indirect(val) => {
                let target_op = self.get_operand(*val);
                self.emit_instr(opcodes::CALL_R, vec![target_op]);
            }
        }

        // Clean up stack adjustment.
        let total_stack_used = layout.stack_space + stack_adj;
        if total_stack_used > 0 {
            self.emit_instr(
                opcodes::ADD_RI,
                vec![
                    MachineOperand::Register(RSP),
                    MachineOperand::Immediate(total_stack_used as i64),
                ],
            );
        }

        // Move return value from rax/xmm0 to destination.
        if let Some(res) = result {
            let dst = self.get_operand(*res);
            let ret_class = classify_return_type(return_ty);
            match ret_class {
                ReturnClass::Integer { regs } => {
                    if let Some(&ret_reg) = regs.first() {
                        self.emit_instr(
                            opcodes::MOV_RR,
                            vec![dst, MachineOperand::Register(ret_reg)],
                        );
                    }
                }
                ReturnClass::Sse { regs } => {
                    if let Some(&ret_reg) = regs.first() {
                        self.emit_instr(
                            opcodes::MOVSD,
                            vec![dst, MachineOperand::Register(ret_reg)],
                        );
                    }
                }
                ReturnClass::Memory => {
                    // Hidden pointer return — the result pointer was passed
                    // as the first argument (in RDI). The callee copies the
                    // return value into that pointer and returns it in RAX.
                    self.emit_instr(
                        opcodes::MOV_RR,
                        vec![dst, MachineOperand::Register(RAX)],
                    );
                }
                ReturnClass::Void => {
                    // No return value.
                }
            }
        }

        Ok(())
    }

    // ===================================================================
    // Type casts
    // ===================================================================

    /// Select a type cast instruction.
    fn select_cast(
        &mut self,
        result: Value,
        op: CastOp,
        value: Value,
        from_ty: &IrType,
        to_ty: &IrType,
    ) -> Result<(), CodeGenError> {
        let src = self.get_operand(value);
        let dst = self.get_operand(result);

        match op {
            CastOp::Trunc => {
                // Truncate: move the value, then AND with mask to keep low bits.
                self.emit_instr(opcodes::MOV_RR, vec![dst.clone(), src]);
                let mask = match to_ty {
                    IrType::I1 => 0x1_i64,
                    IrType::I8 => 0xFF,
                    IrType::I16 => 0xFFFF,
                    IrType::I32 => 0xFFFF_FFFF,
                    _ => return Ok(()),
                };
                self.emit_instr(
                    opcodes::AND_RI,
                    vec![dst, MachineOperand::Immediate(mask)],
                );
            }
            CastOp::ZExt => {
                self.emit_instr(opcodes::MOVZX, vec![dst, src]);
            }
            CastOp::SExt => {
                self.emit_instr(opcodes::MOVSX, vec![dst, src]);
            }
            CastOp::FPToSI => {
                let conv_op = if *from_ty == IrType::F32 {
                    opcodes::CVTTSS2SI
                } else {
                    opcodes::CVTTSD2SI
                };
                self.emit_instr(conv_op, vec![dst, src]);
            }
            CastOp::FPToUI => {
                // For unsigned: convert via signed path. For values that fit
                // in i64 signed range, this is correct. For larger values,
                // a more complex sequence is needed (not common in C code).
                let conv_op = if *from_ty == IrType::F32 {
                    opcodes::CVTTSS2SI
                } else {
                    opcodes::CVTTSD2SI
                };
                self.emit_instr(conv_op, vec![dst, src]);
            }
            CastOp::SIToFP => {
                let conv_op = if *to_ty == IrType::F32 {
                    opcodes::CVTSI2SS
                } else {
                    opcodes::CVTSI2SD
                };
                self.emit_instr(conv_op, vec![dst, src]);
            }
            CastOp::UIToFP => {
                // For unsigned-to-float: convert via signed path (correct for
                // values < 2^63; larger values need adjustment).
                let conv_op = if *to_ty == IrType::F32 {
                    opcodes::CVTSI2SS
                } else {
                    opcodes::CVTSI2SD
                };
                self.emit_instr(conv_op, vec![dst, src]);
            }
            CastOp::FPTrunc => {
                // double → float: cvtsd2ss
                self.emit_instr(opcodes::CVTSD2SS, vec![dst, src]);
            }
            CastOp::FPExt => {
                // float → double: cvtss2sd
                self.emit_instr(opcodes::CVTSS2SD, vec![dst, src]);
            }
            CastOp::PtrToInt | CastOp::IntToPtr => {
                // Pointer ↔ integer: just a MOV (same bit width on x86-64).
                self.emit_instr(opcodes::MOV_RR, vec![dst, src]);
            }
        }

        Ok(())
    }

    // ===================================================================
    // Select instruction
    // ===================================================================

    /// Select a conditional select (ternary) instruction.
    fn select_select(
        &mut self,
        result: Value,
        condition: Value,
        true_val: Value,
        false_val: Value,
        ty: &IrType,
    ) -> Result<(), CodeGenError> {
        let cond_op = self.get_operand(condition);
        let true_op = self.get_operand(true_val);
        let false_op = self.get_operand(false_val);
        let dst = self.get_operand(result);

        // Move false value to destination (default).
        if ty.is_float() {
            let mov = if *ty == IrType::F32 {
                opcodes::MOVSS
            } else {
                opcodes::MOVSD
            };
            self.emit_instr(mov, vec![dst.clone(), false_op]);
            // Test condition, then use a branch or conditional move.
            self.emit_instr(opcodes::TEST_RR, vec![cond_op.clone(), cond_op]);
            // For SSE, we cannot use CMOV. Emit a simple branch sequence.
            let true_label = self.next_vreg;
            self.next_vreg += 1;
            let end_label = self.next_vreg;
            self.next_vreg += 1;
            self.emit_instr(
                opcodes::JCC,
                vec![
                    MachineOperand::Immediate(CondCode::NE as i64),
                    MachineOperand::Label(true_label),
                ],
            );
            self.emit_instr(opcodes::JMP, vec![MachineOperand::Label(end_label)]);
            self.emit_instr(opcodes::NOP, vec![MachineOperand::Label(true_label)]);
            self.emit_instr(mov, vec![dst.clone(), true_op]);
            self.emit_instr(opcodes::NOP, vec![MachineOperand::Label(end_label)]);
        } else {
            // Integer path: use CMOVCC.
            // Operand order: [Immediate(cc), Register(dst), Register(src)]
            // — matches encoding.rs expectation.
            self.emit_instr(opcodes::MOV_RR, vec![dst.clone(), false_op]);
            self.emit_instr(opcodes::TEST_RR, vec![cond_op.clone(), cond_op]);
            self.emit_instr(
                opcodes::CMOVCC,
                vec![
                    MachineOperand::Immediate(CondCode::NE as i64),
                    dst,
                    true_op,
                ],
            );
        }
        Ok(())
    }

    // ===================================================================
    // Constant materialisation
    // ===================================================================

    /// Select a constant load instruction.
    fn select_const(
        &mut self,
        result: Value,
        value: &Constant,
    ) -> Result<(), CodeGenError> {
        // The IR builder emits a placeholder Const instruction for each
        // function parameter (with value = parameter index).  lower_params
        // has already mapped these Values to the correct ABI registers
        // (rdi, rsi, …).  We must NOT emit a load-immediate here, as that
        // would clobber the real argument sitting in the ABI register.
        if self.param_value_set.contains(&result) {
            return Ok(());
        }

        let dst = self.get_operand(result);

        match value {
            Constant::Integer { value: v, ty } => {
                if *v == 0 {
                    // xor reg, reg is shorter and clears flags.
                    self.emit_instr(opcodes::XOR_RR, vec![dst.clone(), dst]);
                } else {
                    self.emit_instr(
                        opcodes::MOV_RI,
                        vec![dst, MachineOperand::Immediate(*v)],
                    );
                }
                let _ = ty;
            }
            Constant::Float { value: v, ty } => {
                // Float constants: load from a RIP-relative constant pool.
                // At isel time we emit a MOV_RM placeholder with a symbol
                // reference; the encoder/linker resolves the actual address.
                let bits = if *ty == IrType::F32 {
                    (*v as f32).to_bits() as i64
                } else {
                    v.to_bits() as i64
                };
                // Materialise float constant via integer register then move
                // to XMM register. This avoids needing a rodata section lookup
                // at isel time.
                let tmp = self.alloc_vreg();
                self.emit_instr(
                    opcodes::MOV_RI,
                    vec![
                        MachineOperand::Register(tmp),
                        MachineOperand::Immediate(bits),
                    ],
                );
                // Move integer bits to XMM (MOVQ / MOVD).
                self.emit_instr(
                    opcodes::MOV_RR,
                    vec![dst, MachineOperand::Register(tmp)],
                );
            }
            Constant::Bool(b) => {
                let imm = if *b { 1i64 } else { 0i64 };
                if imm == 0 {
                    self.emit_instr(opcodes::XOR_RR, vec![dst.clone(), dst]);
                } else {
                    self.emit_instr(
                        opcodes::MOV_RI,
                        vec![dst, MachineOperand::Immediate(imm)],
                    );
                }
            }
            Constant::Null(_ty) => {
                self.emit_instr(opcodes::XOR_RR, vec![dst.clone(), dst]);
            }
            Constant::Undef(_ty) => {
                // Undefined value — emit nothing or a zero (safe choice).
                self.emit_instr(opcodes::XOR_RR, vec![dst.clone(), dst]);
            }
            Constant::ZeroInit(_ty) => {
                self.emit_instr(opcodes::XOR_RR, vec![dst.clone(), dst]);
            }
            Constant::String(bytes) => {
                // String constants are placed in .rodata by codegen; at isel
                // time we load the symbol address.
                let label = format!(".Lstr_{}", self.next_vreg);
                self.next_vreg += 1;
                if self.pic_enabled {
                    self.emit_instr(
                        opcodes::LEA,
                        vec![dst, MachineOperand::Symbol(label)],
                    );
                } else {
                    self.emit_instr(
                        opcodes::MOV_RI,
                        vec![dst, MachineOperand::Symbol(label)],
                    );
                }
                let _ = bytes;
            }
            Constant::GlobalRef(name) => {
                if self.pic_enabled {
                    // GOT-relative addressing for PIC.
                    self.emit_instr(
                        opcodes::MOV_RM,
                        vec![
                            dst,
                            MachineOperand::Symbol(format!("{}@GOTPCREL", name)),
                        ],
                    );
                } else {
                    self.emit_instr(
                        opcodes::LEA,
                        vec![dst, MachineOperand::Symbol(name.clone())],
                    );
                }
            }
        }

        Ok(())
    }

    // ===================================================================
    // Utility helpers
    // ===================================================================

    /// Extract a physical register from a machine operand. If the operand
    /// is a memory reference, return the base register. Otherwise allocate
    /// a vreg and emit a MOV.
    fn operand_to_reg(&mut self, op: &MachineOperand) -> PhysReg {
        match op {
            MachineOperand::Register(r) => *r,
            MachineOperand::Memory { base, .. } => *base,
            _ => {
                let tmp = self.alloc_vreg();
                self.emit_instr(opcodes::MOV_RR, vec![MachineOperand::Register(tmp), op.clone()]);
                tmp
            }
        }
    }

    /// Compute the byte size of an IR type for x86-64 (LP64 model).
    /// This is a simplified version that avoids requiring a TargetConfig
    /// reference by hard-coding 8-byte pointers.
    fn type_size(&self, ty: &IrType) -> usize {
        match ty {
            IrType::Void => 0,
            IrType::I1 | IrType::I8 => 1,
            IrType::I16 => 2,
            IrType::I32 => 4,
            IrType::I64 => 8,
            IrType::F32 => 4,
            IrType::F64 => 8,
            IrType::Pointer(_) => 8,
            IrType::Array { element, count } => self.type_size(element) * count,
            IrType::Struct { fields, packed } => {
                if fields.is_empty() {
                    return 0;
                }
                if *packed {
                    return fields.iter().map(|f| self.type_size(f)).sum();
                }
                let mut offset: usize = 0;
                let mut max_align: usize = 1;
                for field in fields {
                    let align = self.type_alignment(field);
                    offset = (offset + align - 1) & !(align - 1);
                    offset += self.type_size(field);
                    if align > max_align {
                        max_align = align;
                    }
                }
                (offset + max_align - 1) & !(max_align - 1)
            }
            IrType::Function { .. } | IrType::Label => 0,
        }
    }

    /// Compute the byte alignment of an IR type for x86-64.
    fn type_alignment(&self, ty: &IrType) -> usize {
        match ty {
            IrType::Void | IrType::I1 | IrType::I8 => 1,
            IrType::I16 => 2,
            IrType::I32 | IrType::F32 => 4,
            IrType::I64 | IrType::F64 | IrType::Pointer(_) => 8,
            IrType::Array { element, .. } => self.type_alignment(element),
            IrType::Struct { fields, packed } => {
                if *packed {
                    1
                } else {
                    fields
                        .iter()
                        .map(|f| self.type_alignment(f))
                        .max()
                        .unwrap_or(1)
                }
            }
            IrType::Function { .. } | IrType::Label => 1,
        }
    }

    /// Compute the byte offset of a field in a struct.
    fn struct_field_offset(&self, fields: &[IrType], packed: bool, field_idx: usize) -> usize {
        let mut offset: usize = 0;
        for (i, field) in fields.iter().enumerate() {
            if i == field_idx {
                return offset;
            }
            if !packed {
                let align = self.type_alignment(field);
                offset = (offset + align - 1) & !(align - 1);
            }
            offset += self.type_size(field);
        }
        // Align to the field's alignment if we haven't returned yet.
        if field_idx < fields.len() && !packed {
            let align = self.type_alignment(&fields[field_idx]);
            offset = (offset + align - 1) & !(align - 1);
        }
        offset
    }

    /// Try to fold an IR value chain into a complex x86-64 addressing mode.
    /// Currently returns a simple base+displacement if the value is a known
    /// memory operand, otherwise returns a base-only mode.
    #[allow(dead_code)]
    pub fn try_fold_address(&self, ptr: Value) -> AddressMode {
        if let Some(op) = self.value_map.get(&ptr) {
            match op {
                MachineOperand::Memory { base, offset } => {
                    return AddressMode::base_disp(*base, *offset);
                }
                MachineOperand::Register(r) => {
                    return AddressMode {
                        base: Some(*r),
                        index: None,
                        scale: 1,
                        displacement: 0,
                    };
                }
                _ => {}
            }
        }
        AddressMode {
            base: None,
            index: None,
            scale: 1,
            displacement: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// BinOpKind — internal helper enum for binary operation dispatch
// ---------------------------------------------------------------------------

/// Internal helper classifying the kind of binary operation for code
/// selection dispatch. Not exported — purely an implementation detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
    And,
    Or,
    Xor,
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BasicBlock, BlockId, Function, Instruction, IrType, Value};
    use crate::ir::cfg::Terminator;
    use crate::ir::instructions::{Callee, CastOp, CompareOp, Constant, FloatCompareOp};

    /// Helper: create a minimal function with one block and the given instructions.
    fn make_function(instrs: Vec<Instruction>, terminator: Terminator) -> Function {
        let entry = BlockId(0);
        let mut block = BasicBlock::new(entry, "entry".to_string());
        block.instructions = instrs;
        block.terminator = Some(terminator);
        Function {
            name: "test_fn".to_string(),
            return_type: IrType::Void,
            params: vec![],
            param_values: Vec::new(),
            blocks: vec![block],
            entry_block: entry,
            is_definition: true,
        }
    }

    /// Helper: create a function and run instruction selection.
    fn select(instrs: Vec<Instruction>, terminator: Terminator) -> Vec<MachineInstr> {
        let func = make_function(instrs, terminator);
        let mut sel = X86_64InstructionSelector::new(false);
        sel.select_function(&func).expect("instruction selection should succeed")
    }

    /// Helper: check if any emitted instruction has the expected opcode.
    fn has_opcode(instrs: &[MachineInstr], opcode: u32) -> bool {
        instrs.iter().any(|i| i.opcode == opcode)
    }

    // ---- Arithmetic selection tests ----

    #[test]
    fn test_select_add_rr() {
        let instrs = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 10, ty: IrType::I64 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 20, ty: IrType::I64 },
            },
            Instruction::Add {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I64,
            },
        ];
        let result = select(instrs, Terminator::Return { value: None });
        assert!(has_opcode(&result, opcodes::ADD_RR) || has_opcode(&result, opcodes::ADD_RI));
    }

    #[test]
    fn test_select_sub() {
        let instrs = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 30, ty: IrType::I64 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 10, ty: IrType::I64 },
            },
            Instruction::Sub {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I64,
            },
        ];
        let result = select(instrs, Terminator::Return { value: None });
        assert!(has_opcode(&result, opcodes::SUB_RR) || has_opcode(&result, opcodes::SUB_RI));
    }

    #[test]
    fn test_select_mul() {
        let instrs = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 5, ty: IrType::I64 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 6, ty: IrType::I64 },
            },
            Instruction::Mul {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I64,
            },
        ];
        let result = select(instrs, Terminator::Return { value: None });
        assert!(has_opcode(&result, opcodes::IMUL_RR) || has_opcode(&result, opcodes::IMUL_RI));
    }

    #[test]
    fn test_select_div_signed() {
        let instrs = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 100, ty: IrType::I64 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 7, ty: IrType::I64 },
            },
            Instruction::Div {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I64,
                is_signed: true,
            },
        ];
        let result = select(instrs, Terminator::Return { value: None });
        assert!(has_opcode(&result, opcodes::CQO));
        assert!(has_opcode(&result, opcodes::IDIV_R));
    }

    #[test]
    fn test_select_div_unsigned() {
        let instrs = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 100, ty: IrType::I64 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 7, ty: IrType::I64 },
            },
            Instruction::Div {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I64,
                is_signed: false,
            },
        ];
        let result = select(instrs, Terminator::Return { value: None });
        // Should XOR RDX,RDX instead of CQO, then DIV_R.
        assert!(has_opcode(&result, opcodes::XOR_RR));
        assert!(has_opcode(&result, opcodes::DIV_R));
    }

    // ---- Memory access tests ----

    #[test]
    fn test_select_load_i32() {
        let instrs = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 0x1000, ty: IrType::I64 },
            },
            Instruction::Load {
                result: Value(1),
                ty: IrType::I32,
                ptr: Value(0),
            },
        ];
        let result = select(instrs, Terminator::Return { value: None });
        assert!(has_opcode(&result, opcodes::LOAD32));
    }

    #[test]
    fn test_select_load_i64() {
        let instrs = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 0x1000, ty: IrType::I64 },
            },
            Instruction::Load {
                result: Value(1),
                ty: IrType::I64,
                ptr: Value(0),
            },
        ];
        let result = select(instrs, Terminator::Return { value: None });
        assert!(has_opcode(&result, opcodes::LOAD64));
    }

    #[test]
    fn test_select_store() {
        let instrs = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 42, ty: IrType::I64 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 0x2000, ty: IrType::I64 },
            },
            Instruction::Store {
                value: Value(0),
                ptr: Value(1),
            },
        ];
        let result = select(instrs, Terminator::Return { value: None });
        assert!(has_opcode(&result, opcodes::STORE64));
    }

    // ---- Comparison tests ----

    #[test]
    fn test_select_icmp_eq() {
        let instrs = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 1, ty: IrType::I32 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 1, ty: IrType::I32 },
            },
            Instruction::ICmp {
                result: Value(2),
                op: CompareOp::Equal,
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I32,
            },
        ];
        let result = select(instrs, Terminator::Return { value: None });
        assert!(has_opcode(&result, opcodes::CMP_RR) || has_opcode(&result, opcodes::CMP_RI));
        assert!(has_opcode(&result, opcodes::SETCC));
    }

    #[test]
    fn test_select_icmp_slt() {
        let instrs = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 5, ty: IrType::I32 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 10, ty: IrType::I32 },
            },
            Instruction::ICmp {
                result: Value(2),
                op: CompareOp::SignedLess,
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I32,
            },
        ];
        let result = select(instrs, Terminator::Return { value: None });
        assert!(has_opcode(&result, opcodes::SETCC));
    }

    #[test]
    fn test_select_fcmp() {
        let instrs = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Float { value: 1.0, ty: IrType::F64 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Float { value: 2.0, ty: IrType::F64 },
            },
            Instruction::FCmp {
                result: Value(2),
                op: FloatCompareOp::OrderedGreater,
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::F64,
            },
        ];
        let result = select(instrs, Terminator::Return { value: None });
        assert!(has_opcode(&result, opcodes::UCOMISD));
        assert!(has_opcode(&result, opcodes::SETCC));
    }

    // ---- Condition code mapping tests ----

    #[test]
    fn test_compare_op_to_cond_code() {
        assert_eq!(compare_op_to_cond_code(&CompareOp::Equal), CondCode::E);
        assert_eq!(compare_op_to_cond_code(&CompareOp::NotEqual), CondCode::NE);
        assert_eq!(compare_op_to_cond_code(&CompareOp::SignedLess), CondCode::L);
        assert_eq!(compare_op_to_cond_code(&CompareOp::SignedLessEqual), CondCode::LE);
        assert_eq!(compare_op_to_cond_code(&CompareOp::SignedGreater), CondCode::G);
        assert_eq!(compare_op_to_cond_code(&CompareOp::SignedGreaterEqual), CondCode::GE);
        assert_eq!(compare_op_to_cond_code(&CompareOp::UnsignedLess), CondCode::B);
        assert_eq!(compare_op_to_cond_code(&CompareOp::UnsignedLessEqual), CondCode::BE);
        assert_eq!(compare_op_to_cond_code(&CompareOp::UnsignedGreater), CondCode::A);
        assert_eq!(compare_op_to_cond_code(&CompareOp::UnsignedGreaterEqual), CondCode::AE);
    }

    // ---- Control flow tests ----

    #[test]
    fn test_select_branch() {
        let result = select(vec![], Terminator::Branch { target: BlockId(1) });
        assert!(has_opcode(&result, opcodes::JMP));
    }

    #[test]
    fn test_select_cond_branch() {
        let instrs = vec![Instruction::Const {
            result: Value(0),
            value: Constant::Bool(true),
        }];
        let result = select(
            instrs,
            Terminator::CondBranch {
                condition: Value(0),
                true_block: BlockId(1),
                false_block: BlockId(2),
            },
        );
        assert!(has_opcode(&result, opcodes::TEST_RR));
        assert!(has_opcode(&result, opcodes::JCC));
    }

    #[test]
    fn test_select_return() {
        let instrs = vec![Instruction::Const {
            result: Value(0),
            value: Constant::Integer { value: 42, ty: IrType::I64 },
        }];
        let result = select(instrs, Terminator::Return { value: Some(Value(0)) });
        assert!(has_opcode(&result, opcodes::MOV_RR));
        assert!(has_opcode(&result, opcodes::RET));
    }

    // ---- Cast tests ----

    #[test]
    fn test_select_zext() {
        let instrs = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 42, ty: IrType::I8 },
            },
            Instruction::Cast {
                result: Value(1),
                op: CastOp::ZExt,
                value: Value(0),
                from_ty: IrType::I8,
                to_ty: IrType::I32,
            },
        ];
        let result = select(instrs, Terminator::Return { value: None });
        assert!(has_opcode(&result, opcodes::MOVZX));
    }

    #[test]
    fn test_select_sext() {
        let instrs = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: -1, ty: IrType::I8 },
            },
            Instruction::Cast {
                result: Value(1),
                op: CastOp::SExt,
                value: Value(0),
                from_ty: IrType::I8,
                to_ty: IrType::I32,
            },
        ];
        let result = select(instrs, Terminator::Return { value: None });
        assert!(has_opcode(&result, opcodes::MOVSX));
    }

    #[test]
    fn test_select_trunc() {
        let instrs = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 256, ty: IrType::I32 },
            },
            Instruction::Cast {
                result: Value(1),
                op: CastOp::Trunc,
                value: Value(0),
                from_ty: IrType::I32,
                to_ty: IrType::I8,
            },
        ];
        let result = select(instrs, Terminator::Return { value: None });
        assert!(has_opcode(&result, opcodes::AND_RI));
    }

    #[test]
    fn test_select_fp_conversion() {
        // float → double (FPExt)
        let instrs = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Float { value: 3.14, ty: IrType::F32 },
            },
            Instruction::Cast {
                result: Value(1),
                op: CastOp::FPExt,
                value: Value(0),
                from_ty: IrType::F32,
                to_ty: IrType::F64,
            },
        ];
        let result = select(instrs, Terminator::Return { value: None });
        assert!(has_opcode(&result, opcodes::CVTSS2SD));
    }

    // ---- Addressing mode tests ----

    #[test]
    fn test_fold_base_plus_offset() {
        let mut sel = X86_64InstructionSelector::new(false);
        sel.set_operand(
            Value(0),
            MachineOperand::Memory {
                base: RBP,
                offset: -16,
            },
        );
        let addr = sel.try_fold_address(Value(0));
        assert_eq!(addr.base, Some(RBP));
        assert_eq!(addr.displacement, -16);
        assert!(addr.index.is_none());
    }

    #[test]
    fn test_fold_base_register() {
        let mut sel = X86_64InstructionSelector::new(false);
        sel.set_operand(Value(0), MachineOperand::Register(RAX));
        let addr = sel.try_fold_address(Value(0));
        assert_eq!(addr.base, Some(RAX));
        assert_eq!(addr.displacement, 0);
    }

    // ---- Constructor test ----

    #[test]
    fn test_new_selector() {
        let sel = X86_64InstructionSelector::new(true);
        assert!(sel.pic_enabled);
        assert!(sel.value_map.is_empty());
        assert!(sel.instructions.is_empty());
    }

    // ---- PIC call test ----

    #[test]
    fn test_pic_call_uses_plt() {
        let instrs = vec![Instruction::Call {
            result: None,
            callee: Callee::Direct("puts".to_string()),
            args: vec![],
            return_ty: IrType::Void,
        }];
        let func = make_function(instrs, Terminator::Return { value: None });
        let mut sel = X86_64InstructionSelector::new(true);
        let result = sel.select_function(&func).unwrap();
        // Check that the CALL operand contains "@PLT".
        let call_instr = result.iter().find(|i| i.opcode == opcodes::CALL);
        assert!(call_instr.is_some());
        if let Some(instr) = call_instr {
            if let Some(MachineOperand::Symbol(sym)) = instr.operands.first() {
                assert!(sym.contains("@PLT"), "PIC call should use PLT");
            }
        }
    }
}
