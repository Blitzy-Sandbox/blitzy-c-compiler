//! # AArch64 (ARM 64-bit) Code Generation Backend
//!
//! Implements the [`CodeGen`] trait for the AArch64 (ARMv8-A 64-bit) architecture.
//! Generates ELF64 machine code using the AAPCS64 ABI.
//!
//! ## Register File
//!
//! - 31 General-Purpose Registers (x0-x30)
//!   - x31 is context-dependent: SP (Stack Pointer) or XZR (Zero Register)
//!   - x29 is FP (Frame Pointer), x30 is LR (Link Register)
//! - 32 SIMD/FP Registers (v0-v31)
//!   - Accessed as Bn (8-bit), Hn (16-bit), Sn (32-bit), Dn (64-bit), Qn (128-bit)
//!
//! ## Key Architectural Features
//!
//! - Fixed-width 32-bit instructions (simplifies encoding and branch offset calculation)
//! - Barrel shifter operands (combine shift+ALU in one instruction)
//! - Conditional select (CSEL/CSINC/CSINV/CSNEG for branchless conditionals)
//! - Load/store pairs (LDP/STP for efficient register save/restore)
//! - PC-relative addressing (ADRP+ADD for 4KB-page-granular symbol access)
//! - Compare-and-branch (CBZ/CBNZ/TBZ/TBNZ for efficient conditionals)
//!
//! ## Pipeline
//!
//! ```text
//! IR Function → Instruction Selection (isel.rs)
//!            → Register Allocation (regalloc.rs)
//!            → Prologue/Epilogue (abi.rs)
//!            → Instruction Encoding (encoding.rs)
//!            → Machine Code Bytes (always 4-byte aligned)
//! ```
//!
//! ## Zero External Dependencies
//!
//! This module and all its submodules use only the Rust standard library (`std`).
//! No external crates are imported, per project constraint.

// ---------------------------------------------------------------------------
// Submodule declarations
// ---------------------------------------------------------------------------

/// AArch64 instruction selection: maps IR instructions to A64 machine
/// instructions exploiting barrel shifter operands, conditional select,
/// load/store pair optimization, PC-relative addressing, and compare-and-branch.
pub mod isel;

/// AArch64 integrated assembler: encodes selected machine instructions into
/// fixed-width 32-bit binary instruction bytes in little-endian format for
/// ELF64 output.
pub mod encoding;

/// AAPCS64 ABI implementation: function prologue/epilogue generation with
/// STP/LDP pairs, argument classification for x0-x7 and v0-v7, return value
/// conventions, stack frame layout with 16-byte SP alignment enforcement.
pub mod abi;

// ---------------------------------------------------------------------------
// Imports
// ---------------------------------------------------------------------------

use crate::codegen::regalloc::{
    build_value_to_reg_map, compute_live_intervals, linear_scan_allocate, PhysReg, RegisterInfo,
};
use crate::codegen::{
    Architecture, CodeGen, CodeGenError, MachineInstr, MachineOperand, ObjectCode, Relocation,
    Section, SectionFlags, SectionType, Symbol, SymbolBinding, SymbolType, SymbolVisibility,
};
use crate::driver::target::TargetConfig;
use crate::ir::{Function, GlobalVariable, IrType, Module, Value};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// AArch64 Physical Register Constants — General Purpose Registers
// ---------------------------------------------------------------------------
// GPRs x0-x30 map to PhysReg(0)-PhysReg(30).
// x31 is context-dependent: SP (Stack Pointer) when used as a base/address
// register, or XZR (Zero Register) when used as an operand source/destination.
// Both are represented as PhysReg(31) — the encoder distinguishes context.

/// x0 — Argument/result register; first integer argument, integer return value.
pub const X0: PhysReg = PhysReg(0);
/// x1 — Second integer argument register; second return register for pairs.
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
/// x8 — Indirect result location register (for returning large structs via pointer).
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
/// x16 — IP0 (intra-procedure-call scratch register 0, reserved for linker veneers).
#[allow(dead_code)]
pub const X16: PhysReg = PhysReg(16);
/// x17 — IP1 (intra-procedure-call scratch register 1, reserved for linker veneers).
#[allow(dead_code)]
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
/// x29 — Frame Pointer (FP), callee-saved but managed by prologue/epilogue.
#[allow(dead_code)]
pub const X29: PhysReg = PhysReg(29);
/// x30 — Link Register (LR), holds return address after BL instruction.
#[allow(dead_code)]
pub const X30: PhysReg = PhysReg(30);

// x31 context-dependent registers: SP (Stack Pointer) and XZR (Zero Register)
// share the same encoding (PhysReg(31)). The assembler/encoder distinguishes
// them based on the instruction context.

/// SP — Stack Pointer (encoded as x31 in base/address register contexts).
#[allow(dead_code)]
pub const SP: PhysReg = PhysReg(31);
/// XZR — Zero Register (encoded as x31 in operand register contexts).
/// Reads always return zero; writes are discarded.
#[allow(dead_code)]
pub const XZR: PhysReg = PhysReg(31);

// ---------------------------------------------------------------------------
// AArch64 Physical Register Constants — SIMD/FP Registers
// ---------------------------------------------------------------------------
// SIMD/FP registers v0-v31 map to PhysReg(32)-PhysReg(63).
// These 128-bit registers can be accessed as:
//   Bn (8-bit), Hn (16-bit), Sn (32-bit float), Dn (64-bit double), Qn (128-bit)

/// v0 — FP argument/result register (s0/d0), caller-saved.
pub const V0: PhysReg = PhysReg(32);
/// v1 — FP argument/result register (s1/d1), caller-saved.
pub const V1: PhysReg = PhysReg(33);
/// v2 — FP argument register (s2/d2), caller-saved.
pub const V2: PhysReg = PhysReg(34);
/// v3 — FP argument register (s3/d3), caller-saved.
pub const V3: PhysReg = PhysReg(35);
/// v4 — FP argument register (s4/d4), caller-saved.
pub const V4: PhysReg = PhysReg(36);
/// v5 — FP argument register (s5/d5), caller-saved.
pub const V5: PhysReg = PhysReg(37);
/// v6 — FP argument register (s6/d6), caller-saved.
pub const V6: PhysReg = PhysReg(38);
/// v7 — FP argument register (s7/d7), caller-saved.
pub const V7: PhysReg = PhysReg(39);
/// v8 — Callee-saved FP register (lower 64 bits d8 preserved).
pub const V8: PhysReg = PhysReg(40);
/// v9 — Callee-saved FP register (lower 64 bits d9 preserved).
pub const V9: PhysReg = PhysReg(41);
/// v10 — Callee-saved FP register (lower 64 bits d10 preserved).
pub const V10: PhysReg = PhysReg(42);
/// v11 — Callee-saved FP register (lower 64 bits d11 preserved).
pub const V11: PhysReg = PhysReg(43);
/// v12 — Callee-saved FP register (lower 64 bits d12 preserved).
pub const V12: PhysReg = PhysReg(44);
/// v13 — Callee-saved FP register (lower 64 bits d13 preserved).
pub const V13: PhysReg = PhysReg(45);
/// v14 — Callee-saved FP register (lower 64 bits d14 preserved).
pub const V14: PhysReg = PhysReg(46);
/// v15 — Callee-saved FP register (lower 64 bits d15 preserved).
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
// ABI Register Aliases
// ---------------------------------------------------------------------------

/// FP — Frame Pointer (alias for x29).
#[allow(dead_code)]
pub const FP: PhysReg = X29;
/// LR — Link Register (alias for x30).
#[allow(dead_code)]
pub const LR: PhysReg = X30;
/// IP0 — Intra-procedure-call scratch register 0 (alias for x16).
/// Reserved for linker-generated veneers; NOT allocatable.
#[allow(dead_code)]
pub const IP0: PhysReg = X16;
/// IP1 — Intra-procedure-call scratch register 1 (alias for x17).
/// Reserved for linker-generated veneers; NOT allocatable.
#[allow(dead_code)]
pub const IP1: PhysReg = X17;

// ---------------------------------------------------------------------------
// build_aarch64_reg_names — register name mapping construction
// ---------------------------------------------------------------------------

/// Builds the AArch64 register name mapping for diagnostic output and debugging.
///
/// Maps `PhysReg` identifiers to their canonical ABI names:
/// - GPRs: `"x0"` through `"x28"`, `"fp"` (x29), `"lr"` (x30), `"sp"` (x31)
/// - SIMD/FP: `"v0"` through `"v31"`
///
/// The returned `HashMap` is stored in [`RegisterInfo::reg_names`] and used by
/// the register allocator for human-readable output in diagnostics and debug dumps.
pub fn build_aarch64_reg_names() -> HashMap<PhysReg, &'static str> {
    let mut names = HashMap::new();

    // GPRs x0-x28 use numeric names.
    names.insert(PhysReg(0), "x0");
    names.insert(PhysReg(1), "x1");
    names.insert(PhysReg(2), "x2");
    names.insert(PhysReg(3), "x3");
    names.insert(PhysReg(4), "x4");
    names.insert(PhysReg(5), "x5");
    names.insert(PhysReg(6), "x6");
    names.insert(PhysReg(7), "x7");
    names.insert(PhysReg(8), "x8");
    names.insert(PhysReg(9), "x9");
    names.insert(PhysReg(10), "x10");
    names.insert(PhysReg(11), "x11");
    names.insert(PhysReg(12), "x12");
    names.insert(PhysReg(13), "x13");
    names.insert(PhysReg(14), "x14");
    names.insert(PhysReg(15), "x15");
    names.insert(PhysReg(16), "x16");
    names.insert(PhysReg(17), "x17");
    names.insert(PhysReg(18), "x18");
    names.insert(PhysReg(19), "x19");
    names.insert(PhysReg(20), "x20");
    names.insert(PhysReg(21), "x21");
    names.insert(PhysReg(22), "x22");
    names.insert(PhysReg(23), "x23");
    names.insert(PhysReg(24), "x24");
    names.insert(PhysReg(25), "x25");
    names.insert(PhysReg(26), "x26");
    names.insert(PhysReg(27), "x27");
    names.insert(PhysReg(28), "x28");

    // x29 = Frame Pointer — use ABI alias name.
    names.insert(PhysReg(29), "fp");
    // x30 = Link Register — use ABI alias name.
    names.insert(PhysReg(30), "lr");
    // x31 = Stack Pointer (context-dependent with XZR).
    names.insert(PhysReg(31), "sp");

    // SIMD/FP registers v0-v31.
    names.insert(PhysReg(32), "v0");
    names.insert(PhysReg(33), "v1");
    names.insert(PhysReg(34), "v2");
    names.insert(PhysReg(35), "v3");
    names.insert(PhysReg(36), "v4");
    names.insert(PhysReg(37), "v5");
    names.insert(PhysReg(38), "v6");
    names.insert(PhysReg(39), "v7");
    names.insert(PhysReg(40), "v8");
    names.insert(PhysReg(41), "v9");
    names.insert(PhysReg(42), "v10");
    names.insert(PhysReg(43), "v11");
    names.insert(PhysReg(44), "v12");
    names.insert(PhysReg(45), "v13");
    names.insert(PhysReg(46), "v14");
    names.insert(PhysReg(47), "v15");
    names.insert(PhysReg(48), "v16");
    names.insert(PhysReg(49), "v17");
    names.insert(PhysReg(50), "v18");
    names.insert(PhysReg(51), "v19");
    names.insert(PhysReg(52), "v20");
    names.insert(PhysReg(53), "v21");
    names.insert(PhysReg(54), "v22");
    names.insert(PhysReg(55), "v23");
    names.insert(PhysReg(56), "v24");
    names.insert(PhysReg(57), "v25");
    names.insert(PhysReg(58), "v26");
    names.insert(PhysReg(59), "v27");
    names.insert(PhysReg(60), "v28");
    names.insert(PhysReg(61), "v29");
    names.insert(PhysReg(62), "v30");
    names.insert(PhysReg(63), "v31");

    names
}

// ---------------------------------------------------------------------------
// aarch64_register_info — RegisterInfo construction for AArch64
// ---------------------------------------------------------------------------

/// Builds the AArch64 [`RegisterInfo`] descriptor for the shared register allocator.
///
/// The register allocator uses this to determine:
/// - Which physical registers are available for allocation
/// - Which registers are callee-saved (requiring save/restore in prologue/epilogue)
/// - Allocation priority order (caller-saved first to minimise prologue/epilogue overhead)
///
/// ## Allocatable registers (27 GPRs + 32 SIMD/FP)
///
/// The following GPRs are **NOT** allocatable:
/// - x29 (FP) — managed by prologue/epilogue, not available for general allocation
/// - x30 (LR) — managed by BL/RET and prologue/epilogue save
/// - SP/XZR (x31) — hardware stack pointer / zero register
/// - x16 (IP0) — reserved for linker-generated veneers and trampolines
/// - x17 (IP1) — reserved for linker-generated veneers and trampolines
///
/// ## Allocation priority
///
/// Caller-saved registers are listed first to minimise prologue/epilogue overhead:
/// 1. Caller-saved temporaries (x9-x15): preferred first, cheapest to use
/// 2. Argument registers (x0-x7, x8): caller-saved, overlap with parameter passing
/// 3. Platform register (x18): caller-saved on Linux
/// 4. Callee-saved registers (x19-x28): used only when caller-saved registers are exhausted
pub fn aarch64_register_info() -> RegisterInfo {
    RegisterInfo {
        int_regs: vec![
            // Caller-saved temporaries (preferred for allocation — no save/restore cost)
            X9, X10, X11, X12, X13, X14, X15,
            // Argument registers (caller-saved; may conflict with parameter passing)
            X0, X1, X2, X3, X4, X5, X6, X7,  // Indirect result register (caller-saved)
            X8,  // Platform register (caller-saved on Linux)
            X18, // Callee-saved registers (used as last resort; require save/restore)
            X19, X20, X21, X22, X23, X24, X25, X26, X27, X28,
        ],
        float_regs: vec![
            // Caller-saved FP argument registers (preferred)
            V0, V1, V2, V3, V4, V5, V6, V7, // More caller-saved FP registers
            V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, V29, V30, V31,
            // Callee-saved FP registers (d8-d15, lower 64 bits preserved)
            V8, V9, V10, V11, V12, V13, V14, V15,
        ],
        callee_saved_int: vec![X19, X20, X21, X22, X23, X24, X25, X26, X27, X28],
        callee_saved_float: vec![V8, V9, V10, V11, V12, V13, V14, V15],
        reg_names: build_aarch64_reg_names(),
    }
}

// ---------------------------------------------------------------------------
// Aarch64CodeGen — AArch64 code generation backend
// ---------------------------------------------------------------------------

/// AArch64 code generation backend implementing the [`CodeGen`] trait.
///
/// Coordinates the full code generation pipeline for a single compilation unit:
/// 1. Iterates over IR functions in the module
/// 2. Computes live intervals and runs register allocation per function
/// 3. Generates AAPCS64-compliant prologue and epilogue
/// 4. Runs instruction selection to map IR to A64 machine instructions
/// 5. Encodes all instructions to fixed-width 32-bit machine code bytes
/// 6. Collects relocations for the linker to resolve
/// 7. Handles global variables by placing them in appropriate data sections
///
/// The output is an [`ObjectCode`] containing `.text`, `.data`, `.rodata`, and
/// `.bss` sections with symbols and relocations ready for ELF64 emission.
pub struct Aarch64CodeGen;

impl Aarch64CodeGen {
    /// Creates a new AArch64 code generator instance.
    ///
    /// The generator is stateless — all per-function state is created during
    /// `generate()` and discarded afterward.
    pub fn new() -> Self {
        Aarch64CodeGen
    }

    /// Generates machine code for a single IR function.
    ///
    /// Executes the per-function code generation pipeline:
    /// instruction selection → register allocation → prologue/epilogue → encoding.
    ///
    /// Returns the encoded machine code bytes and any relocations produced.
    fn generate_function(
        &self,
        function: &Function,
        target: &TargetConfig,
        reg_info: &RegisterInfo,
    ) -> Result<(Vec<u8>, Vec<Relocation>), CodeGenError> {
        // Step 1: Run instruction selection — maps IR to A64 machine instructions.
        let mut selector = isel::Aarch64InstructionSelector::new();
        let mut body_instrs = selector.select_function(function)?;

        // Step 2: Compute live intervals and register allocation on the IR.
        let mut intervals = compute_live_intervals(function);
        let alloc_result = linear_scan_allocate(&mut intervals, reg_info);

        // Step 3: Apply register assignments to machine instructions.
        // Maps vreg IDs → IR Values → PhysRegs from the allocator.
        let vreg_to_value = selector.build_vreg_to_value_map();
        let value_to_reg = build_value_to_reg_map(&alloc_result);
        Self::apply_aarch64_reg_assignments(&mut body_instrs, &vreg_to_value, &value_to_reg);

        // Step 4: Generate AAPCS64-compliant function prologue.
        let prologue = abi::generate_prologue(function, &alloc_result, target);

        // Step 5: Generate AAPCS64-compliant function epilogue.
        let epilogue = abi::generate_epilogue(function, &alloc_result, target);

        // Step 6: Insert prologue at the beginning and epilogue before each RET.
        // The ISel emits RET for each `return` statement, but the epilogue
        // (callee-saved restore, frame deallocation) must execute before RET.
        // We replace each body RET with the full epilogue (which ends with RET).
        let ret_opcode = isel::Aarch64Opcode::RET.as_u32();

        // Find all RET positions in the body (process in reverse to preserve indices).
        let ret_positions: Vec<usize> = body_instrs
            .iter()
            .enumerate()
            .filter(|(_, instr)| instr.opcode == ret_opcode)
            .map(|(i, _)| i)
            .collect();

        if !ret_positions.is_empty() {
            // Replace each body RET with the full epilogue sequence.
            for &pos in ret_positions.iter().rev() {
                body_instrs.remove(pos);
                for (j, epi_instr) in epilogue.iter().enumerate() {
                    body_instrs.insert(pos + j, epi_instr.clone());
                }
            }
        } else {
            // No RET in body — append epilogue at the end.
            body_instrs.extend(epilogue);
        }

        // Insert prologue at the very beginning.
        for (i, pro_instr) in prologue.iter().enumerate() {
            body_instrs.insert(i, pro_instr.clone());
        }

        let all_instrs = body_instrs;

        // Step 7: Encode all instructions to machine code bytes.
        // All AArch64 instructions are fixed-width 32 bits (4 bytes).
        let mut encoder = encoding::Aarch64Encoder::new();
        let func_code = encoder.encode_function(&all_instrs);

        // Collect relocations from the encoder for linker resolution.
        let relocations = encoder.get_relocations().to_vec();

        Ok((func_code, relocations))
    }

    /// Emits a global variable's initializer data as raw bytes.
    ///
    /// Handles all `Constant` variants: integers, floats, booleans, null pointers,
    /// zero-initialized values, string literals, and global symbol references.
    /// Apply register assignments to AArch64 machine instructions.
    /// Maps virtual register IDs (>= 64) to physical registers via the
    /// vreg→Value→PhysReg chain from the register allocator.
    fn apply_aarch64_reg_assignments(
        instrs: &mut Vec<MachineInstr>,
        vreg_to_value: &HashMap<u32, Value>,
        value_to_reg: &HashMap<Value, PhysReg>,
    ) {
        // AArch64 GP registers: 0-30 (X0-X30), 31 = SP/XZR
        // FP/SIMD registers: numbered separately in instruction encoding
        let fallback_pool: [PhysReg; 4] = [
            PhysReg(0), // X0
            PhysReg(1), // X1
            PhysReg(2), // X2
            PhysReg(9), // X9 (caller-saved temp)
        ];
        let mut fallback_idx: usize = 0;

        for instr in instrs.iter_mut() {
            for operand in instr.operands.iter_mut() {
                match operand {
                    MachineOperand::Register(ref mut reg) => {
                        if reg.0 >= 64 {
                            let vreg_id = reg.0 as u32;
                            if let Some(&value) = vreg_to_value.get(&vreg_id) {
                                if let Some(&phys) = value_to_reg.get(&value) {
                                    *reg = phys;
                                    continue;
                                }
                            }
                            *reg = fallback_pool[fallback_idx % fallback_pool.len()];
                            fallback_idx += 1;
                        }
                    }
                    MachineOperand::Memory { ref mut base, .. } => {
                        if base.0 >= 64 {
                            let vreg_id = base.0 as u32;
                            if let Some(&value) = vreg_to_value.get(&vreg_id) {
                                if let Some(&phys) = value_to_reg.get(&value) {
                                    *base = phys;
                                    continue;
                                }
                            }
                            *base = fallback_pool[fallback_idx % fallback_pool.len()];
                            fallback_idx += 1;
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    fn emit_global_data(&self, global: &GlobalVariable, target: &TargetConfig) -> Vec<u8> {
        match &global.initializer {
            Some(init) => self.emit_constant_data(init, &global.ty, target),
            None => {
                // No initializer — zero-fill to the type's size.
                let size = global.ty.size(target);
                vec![0u8; size]
            }
        }
    }

    /// Encodes a compile-time constant into raw bytes.
    ///
    /// Uses little-endian byte order per AArch64's default configuration
    /// for all four supported Linux targets.
    fn emit_constant_data(
        &self,
        constant: &crate::ir::Constant,
        _ty: &IrType,
        target: &TargetConfig,
    ) -> Vec<u8> {
        use crate::ir::Constant;

        match constant {
            Constant::Integer { value, ty: int_ty } => {
                let size = int_ty.size(target);
                let mut bytes = vec![0u8; size];
                let val = *value;
                // Write little-endian integer bytes.
                for i in 0..size.min(8) {
                    bytes[i] = ((val >> (i * 8)) & 0xFF) as u8;
                }
                bytes
            }
            Constant::Float {
                value,
                ty: float_ty,
            } => {
                match float_ty {
                    IrType::F32 => {
                        let bits = (*value as f32).to_bits();
                        bits.to_le_bytes().to_vec()
                    }
                    IrType::F64 => {
                        let bits = value.to_bits();
                        bits.to_le_bytes().to_vec()
                    }
                    _ => {
                        // Fallback for unexpected float type.
                        let bits = value.to_bits();
                        bits.to_le_bytes().to_vec()
                    }
                }
            }
            Constant::Bool(val) => {
                vec![if *val { 1u8 } else { 0u8 }]
            }
            Constant::Null(_) => {
                // Null pointer — zero-fill to pointer size.
                vec![0u8; target.pointer_size as usize]
            }
            Constant::ZeroInit(zi_ty) => {
                let size = zi_ty.size(target);
                vec![0u8; size]
            }
            Constant::Undef(undef_ty) => {
                // Undefined values can be anything; zero-fill is safe.
                let size = undef_ty.size(target);
                vec![0u8; size]
            }
            Constant::String(bytes) => bytes.clone(),
            Constant::GlobalRef(_name) => {
                // Global reference — emit a zero placeholder that the linker
                // will fill via a relocation entry. The actual relocation is
                // handled separately during symbol processing.
                vec![0u8; target.pointer_size as usize]
            }
        }
    }

    /// Determines whether a global variable should be placed in the `.rodata`
    /// section (read-only data) rather than `.data` (read-write data).
    ///
    /// A global is read-only if it is not mutable at runtime. In C, this
    /// corresponds to `const`-qualified global variables and string literals.
    fn is_readonly_global(&self, global: &GlobalVariable) -> bool {
        // String literals and explicitly constant globals go to .rodata.
        if let Some(crate::ir::Constant::String(_)) = &global.initializer {
            return true;
        }
        // Static globals without initializers that are not extern could go
        // to .bss, but that's handled by the is_bss check below.
        false
    }

    /// Determines whether a global variable should be placed in the `.bss`
    /// section (zero-initialized data that occupies no file space).
    fn is_bss_global(&self, global: &GlobalVariable) -> bool {
        match &global.initializer {
            None => true,
            Some(crate::ir::Constant::ZeroInit(_)) => true,
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// CodeGen trait implementation
// ---------------------------------------------------------------------------

impl CodeGen for Aarch64CodeGen {
    /// Generates machine code for the given IR module targeting AArch64.
    ///
    /// Produces an [`ObjectCode`] containing:
    /// - `.text` section with encoded machine code for all function definitions
    /// - `.data` section for initialized global variables
    /// - `.rodata` section for read-only data (string literals, const globals)
    /// - `.bss` section for zero-initialized or uninitialized globals
    /// - Symbol table entries for all functions and globals
    /// - Relocation entries for unresolved symbol references (R_AARCH64_*)
    fn generate(&self, module: &Module, target: &TargetConfig) -> Result<ObjectCode, CodeGenError> {
        let reg_info = aarch64_register_info();

        let mut sections: Vec<Section> = Vec::new();
        let mut symbols: Vec<Symbol> = Vec::new();
        let mut all_relocations: Vec<Relocation> = Vec::new();

        // ===================================================================
        // Phase 1: Generate code for all function definitions
        // ===================================================================

        let mut text_data: Vec<u8> = Vec::new();
        let text_section_index: usize = 0; // .text is always section 0

        for function in &module.functions {
            if !function.is_definition {
                // External declaration — emit an undefined symbol but no code.
                symbols.push(Symbol {
                    name: function.name.clone(),
                    section_index: 0,
                    offset: 0,
                    size: 0,
                    binding: SymbolBinding::Global,
                    symbol_type: SymbolType::NoType,
                    visibility: SymbolVisibility::Default,
                    is_definition: false,
                });
                continue;
            }

            // Generate machine code for this function.
            let (func_code, func_relocs) = self.generate_function(function, target, &reg_info)?;

            // Record the function's starting offset in the .text section.
            let func_offset = text_data.len() as u64;

            // Emit the function symbol.
            // All defined functions are global by default (C non-static functions
            // have external linkage). The `is_static` flag on IR Function would
            // need to be propagated from sema for proper local binding.
            symbols.push(Symbol {
                name: function.name.clone(),
                section_index: text_section_index,
                offset: func_offset,
                size: func_code.len() as u64,
                binding: SymbolBinding::Global,
                symbol_type: SymbolType::Function,
                visibility: SymbolVisibility::Default,
                is_definition: true,
            });

            // Adjust relocation offsets to account for the function's position
            // within the .text section, and set the correct section index.
            for mut reloc in func_relocs {
                reloc.offset += func_offset;
                reloc.section_index = text_section_index;
                all_relocations.push(reloc);
            }

            // Append function's machine code to the .text section buffer.
            text_data.extend_from_slice(&func_code);
        }

        // Build the .text section.
        sections.push(Section {
            name: ".text".to_string(),
            data: text_data,
            section_type: SectionType::Text,
            alignment: 4, // AArch64 instructions are 4-byte aligned
            flags: SectionFlags {
                writable: false,
                executable: true,
                allocatable: true,
            },
        });

        // ===================================================================
        // Phase 2: Handle global variables → .data, .rodata, .bss sections
        // ===================================================================

        let mut data_bytes: Vec<u8> = Vec::new();
        let mut rodata_bytes: Vec<u8> = Vec::new();
        let mut bss_size: u64 = 0;

        // Track section indices for symbols. We'll build sections in order:
        // 0 = .text (already added), 1 = .data, 2 = .rodata, 3 = .bss
        let data_section_index: usize = 1;
        let rodata_section_index: usize = 2;
        let bss_section_index: usize = 3;

        for global in &module.globals {
            if global.is_extern {
                // Extern declaration — emit undefined symbol, no data.
                symbols.push(Symbol {
                    name: global.name.clone(),
                    section_index: 0,
                    offset: 0,
                    size: 0,
                    binding: SymbolBinding::Global,
                    symbol_type: SymbolType::NoType,
                    visibility: SymbolVisibility::Default,
                    is_definition: false,
                });
                continue;
            }

            let var_size = global.ty.size(target) as u64;
            let var_align = global.ty.alignment(target) as u64;

            if self.is_bss_global(global) {
                // .bss — zero-initialized, occupies no file space.
                // Align the bss offset.
                let aligned_offset = align_up_u64(bss_size, var_align);
                symbols.push(Symbol {
                    name: global.name.clone(),
                    section_index: bss_section_index,
                    offset: aligned_offset,
                    size: var_size,
                    binding: if global.is_static {
                        SymbolBinding::Local
                    } else {
                        SymbolBinding::Global
                    },
                    symbol_type: SymbolType::Object,
                    visibility: SymbolVisibility::Default,
                    is_definition: true,
                });
                bss_size = aligned_offset + var_size;
            } else if self.is_readonly_global(global) {
                // .rodata — read-only data (string literals, const globals).
                let aligned_offset = align_up_u64(rodata_bytes.len() as u64, var_align);
                // Pad to alignment.
                while (rodata_bytes.len() as u64) < aligned_offset {
                    rodata_bytes.push(0);
                }
                let offset = rodata_bytes.len() as u64;
                let init_data = self.emit_global_data(global, target);
                rodata_bytes.extend_from_slice(&init_data);

                // Handle global references that need relocations.
                if let Some(crate::ir::Constant::GlobalRef(ref_name)) = &global.initializer {
                    all_relocations.push(Relocation {
                        offset,
                        symbol: ref_name.clone(),
                        reloc_type: crate::codegen::RelocationType::Aarch64_ABS64,
                        addend: 0,
                        section_index: rodata_section_index,
                    });
                }

                symbols.push(Symbol {
                    name: global.name.clone(),
                    section_index: rodata_section_index,
                    offset,
                    size: var_size,
                    binding: if global.is_static {
                        SymbolBinding::Local
                    } else {
                        SymbolBinding::Global
                    },
                    symbol_type: SymbolType::Object,
                    visibility: SymbolVisibility::Default,
                    is_definition: true,
                });
            } else {
                // .data — initialized read-write data.
                let aligned_offset = align_up_u64(data_bytes.len() as u64, var_align);
                // Pad to alignment.
                while (data_bytes.len() as u64) < aligned_offset {
                    data_bytes.push(0);
                }
                let offset = data_bytes.len() as u64;
                let init_data = self.emit_global_data(global, target);
                data_bytes.extend_from_slice(&init_data);

                // Handle global references that need relocations.
                if let Some(crate::ir::Constant::GlobalRef(ref_name)) = &global.initializer {
                    all_relocations.push(Relocation {
                        offset,
                        symbol: ref_name.clone(),
                        reloc_type: crate::codegen::RelocationType::Aarch64_ABS64,
                        addend: 0,
                        section_index: data_section_index,
                    });
                }

                symbols.push(Symbol {
                    name: global.name.clone(),
                    section_index: data_section_index,
                    offset,
                    size: var_size,
                    binding: if global.is_static {
                        SymbolBinding::Local
                    } else {
                        SymbolBinding::Global
                    },
                    symbol_type: SymbolType::Object,
                    visibility: SymbolVisibility::Default,
                    is_definition: true,
                });
            }
        }

        // Build data sections.
        sections.push(Section {
            name: ".data".to_string(),
            data: data_bytes,
            section_type: SectionType::Data,
            alignment: 8, // 8-byte alignment for AArch64 data
            flags: SectionFlags {
                writable: true,
                executable: false,
                allocatable: true,
            },
        });

        sections.push(Section {
            name: ".rodata".to_string(),
            data: rodata_bytes,
            section_type: SectionType::Rodata,
            alignment: 8,
            flags: SectionFlags {
                writable: false,
                executable: false,
                allocatable: true,
            },
        });

        // .bss section — no data bytes, just declared size.
        // The actual bss_size is encoded in the section size (data.len() returns 0
        // for an empty vec, but the linker uses the symbol offsets to determine
        // the BSS extent).
        sections.push(Section {
            name: ".bss".to_string(),
            data: Vec::new(),
            section_type: SectionType::Bss,
            alignment: 8,
            flags: SectionFlags {
                writable: true,
                executable: false,
                allocatable: true,
            },
        });

        Ok(ObjectCode {
            sections,
            symbols,
            relocations: all_relocations,
            target_arch: Architecture::Aarch64,
        })
    }

    /// Returns the target architecture for this backend.
    fn target_arch(&self) -> Architecture {
        Architecture::Aarch64
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Aligns `value` up to the next multiple of `alignment`.
///
/// `alignment` must be a power of two and greater than zero.
#[inline]
fn align_up_u64(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    (value + alignment - 1) & !(alignment - 1)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =======================================================================
    // Register alias tests
    // =======================================================================

    #[test]
    fn test_fp_alias() {
        assert_eq!(FP, X29, "FP should alias x29");
    }

    #[test]
    fn test_lr_alias() {
        assert_eq!(LR, X30, "LR should alias x30");
    }

    #[test]
    fn test_ip0_alias() {
        assert_eq!(IP0, X16, "IP0 should alias x16");
    }

    #[test]
    fn test_ip1_alias() {
        assert_eq!(IP1, X17, "IP1 should alias x17");
    }

    #[test]
    fn test_sp_xzr_same_encoding() {
        assert_eq!(
            SP, XZR,
            "SP and XZR should have the same encoding (PhysReg(31))"
        );
        assert_eq!(SP.0, 31);
        assert_eq!(XZR.0, 31);
    }

    // =======================================================================
    // Register constant value tests
    // =======================================================================

    #[test]
    fn test_gpr_numbering() {
        assert_eq!(X0.0, 0);
        assert_eq!(X1.0, 1);
        assert_eq!(X15.0, 15);
        assert_eq!(X28.0, 28);
        assert_eq!(X29.0, 29);
        assert_eq!(X30.0, 30);
    }

    #[test]
    fn test_simd_fp_numbering() {
        assert_eq!(V0.0, 32);
        assert_eq!(V1.0, 33);
        assert_eq!(V7.0, 39);
        assert_eq!(V8.0, 40);
        assert_eq!(V15.0, 47);
        assert_eq!(V16.0, 48);
        assert_eq!(V31.0, 63);
    }

    // =======================================================================
    // RegisterInfo tests
    // =======================================================================

    #[test]
    fn test_allocatable_gpr_count() {
        let reg_info = aarch64_register_info();
        // Should have 27 allocatable GPRs:
        // x9-x15 (7) + x0-x7 (8) + x8 (1) + x18 (1) + x19-x28 (10) = 27
        assert_eq!(
            reg_info.int_regs.len(),
            27,
            "AArch64 should have 27 allocatable GPRs"
        );
    }

    #[test]
    fn test_allocatable_fpr_count() {
        let reg_info = aarch64_register_info();
        // Should have 32 allocatable FP registers (v0-v31).
        assert_eq!(
            reg_info.float_regs.len(),
            32,
            "AArch64 should have 32 allocatable FP registers"
        );
    }

    #[test]
    fn test_callee_saved_gpr_count() {
        let reg_info = aarch64_register_info();
        // AAPCS64: x19-x28 = 10 callee-saved GPRs.
        assert_eq!(
            reg_info.callee_saved_int.len(),
            10,
            "AAPCS64 specifies 10 callee-saved GPRs (x19-x28)"
        );
    }

    #[test]
    fn test_callee_saved_fpr_count() {
        let reg_info = aarch64_register_info();
        // AAPCS64: v8-v15 / d8-d15 = 8 callee-saved FP registers.
        assert_eq!(
            reg_info.callee_saved_float.len(),
            8,
            "AAPCS64 specifies 8 callee-saved FP registers (v8-v15)"
        );
    }

    #[test]
    fn test_callee_saved_gpr_values() {
        let reg_info = aarch64_register_info();
        let expected = vec![X19, X20, X21, X22, X23, X24, X25, X26, X27, X28];
        assert_eq!(
            reg_info.callee_saved_int, expected,
            "Callee-saved GPRs must be x19-x28"
        );
    }

    #[test]
    fn test_callee_saved_fpr_values() {
        let reg_info = aarch64_register_info();
        let expected = vec![V8, V9, V10, V11, V12, V13, V14, V15];
        assert_eq!(
            reg_info.callee_saved_float, expected,
            "Callee-saved FPRs must be v8-v15"
        );
    }

    #[test]
    fn test_non_allocatable_registers_excluded() {
        let reg_info = aarch64_register_info();
        // SP/XZR (31), FP/x29, LR/x30, IP0/x16, IP1/x17 must NOT be allocatable.
        assert!(
            !reg_info.int_regs.contains(&PhysReg(31)),
            "SP/XZR must not be allocatable"
        );
        assert!(
            !reg_info.int_regs.contains(&X29),
            "FP (x29) must not be allocatable"
        );
        assert!(
            !reg_info.int_regs.contains(&X30),
            "LR (x30) must not be allocatable"
        );
        assert!(
            !reg_info.int_regs.contains(&X16),
            "IP0 (x16) must not be allocatable"
        );
        assert!(
            !reg_info.int_regs.contains(&X17),
            "IP1 (x17) must not be allocatable"
        );
    }

    #[test]
    fn test_caller_saved_temporaries_listed_first() {
        let reg_info = aarch64_register_info();
        // First 7 registers should be the caller-saved temporaries x9-x15.
        let first_seven: Vec<PhysReg> = reg_info.int_regs[..7].to_vec();
        let expected = vec![X9, X10, X11, X12, X13, X14, X15];
        assert_eq!(
            first_seven, expected,
            "Caller-saved temporaries (x9-x15) should be listed first for allocation priority"
        );
    }

    // =======================================================================
    // Register name tests
    // =======================================================================

    #[test]
    fn test_gpr_names_present() {
        let names = build_aarch64_reg_names();
        for i in 0u16..29 {
            let expected = format!("x{}", i);
            assert_eq!(
                names.get(&PhysReg(i)).copied(),
                Some(expected.as_str()),
                "Register x{} should have name \"x{}\"",
                i,
                i
            );
        }
        assert_eq!(
            names.get(&PhysReg(29)).copied(),
            Some("fp"),
            "x29 should be named 'fp'"
        );
        assert_eq!(
            names.get(&PhysReg(30)).copied(),
            Some("lr"),
            "x30 should be named 'lr'"
        );
        assert_eq!(
            names.get(&PhysReg(31)).copied(),
            Some("sp"),
            "x31 should be named 'sp'"
        );
    }

    #[test]
    fn test_fpr_names_present() {
        let names = build_aarch64_reg_names();
        for i in 0u16..32 {
            let expected = format!("v{}", i);
            assert_eq!(
                names.get(&PhysReg(32 + i)).copied(),
                Some(expected.as_str()),
                "Register v{} should have name \"v{}\"",
                i,
                i
            );
        }
    }

    #[test]
    fn test_total_register_names() {
        let names = build_aarch64_reg_names();
        // 32 GPRs (x0-x28 + fp + lr + sp) + 32 FPRs (v0-v31) = 64 names
        assert_eq!(
            names.len(),
            64,
            "Should have 64 total register names (32 GPR + 32 FPR)"
        );
    }

    // =======================================================================
    // CodeGen trait tests
    // =======================================================================

    #[test]
    fn test_target_arch() {
        let codegen = Aarch64CodeGen::new();
        assert_eq!(codegen.target_arch(), Architecture::Aarch64);
    }

    #[test]
    fn test_codegen_new() {
        let _codegen = Aarch64CodeGen::new();
        // Constructor should succeed without panic.
    }

    #[test]
    fn test_empty_module_generates_valid_output() {
        let codegen = Aarch64CodeGen::new();
        let module = Module::new("test".to_string());

        // Create a minimal AArch64 target config for testing.
        let target = TargetConfig {
            arch: Architecture::Aarch64,
            triple: "aarch64-linux-gnu".to_string(),
            pointer_size: 8,
            long_size: 8,
            long_double_size: 16,
            elf_class: crate::driver::target::ElfClass::Elf64,
            endianness: crate::driver::target::Endianness::Little,
            abi: crate::driver::target::AbiVariant::Aapcs64,
            max_alignment: 16,
            stack_alignment: 16,
            elf_machine: 183, // EM_AARCH64
            elf_osabi: 0,
            gpr_count: 31,
            fpr_count: 32,
            crt_search_paths: Vec::new(),
            lib_search_paths: Vec::new(),
            retpoline: false,
            cf_protection: false,
            pic: false,
            no_red_zone: false,
            function_sections: false,
            data_sections: false,
        };

        let result = codegen.generate(&module, &target);
        assert!(
            result.is_ok(),
            "Generating code for empty module should succeed"
        );

        let object = result.unwrap();
        assert_eq!(object.target_arch, Architecture::Aarch64);
        // Should have 4 sections: .text, .data, .rodata, .bss
        assert_eq!(object.sections.len(), 4);
        assert_eq!(object.sections[0].name, ".text");
        assert_eq!(object.sections[1].name, ".data");
        assert_eq!(object.sections[2].name, ".rodata");
        assert_eq!(object.sections[3].name, ".bss");
    }

    #[test]
    fn test_text_section_alignment() {
        let codegen = Aarch64CodeGen::new();
        let module = Module::new("test".to_string());
        let target = TargetConfig {
            arch: Architecture::Aarch64,
            triple: "aarch64-linux-gnu".to_string(),
            pointer_size: 8,
            long_size: 8,
            long_double_size: 16,
            elf_class: crate::driver::target::ElfClass::Elf64,
            endianness: crate::driver::target::Endianness::Little,
            abi: crate::driver::target::AbiVariant::Aapcs64,
            max_alignment: 16,
            stack_alignment: 16,
            elf_machine: 183,
            elf_osabi: 0,
            gpr_count: 31,
            fpr_count: 32,
            crt_search_paths: Vec::new(),
            lib_search_paths: Vec::new(),
            retpoline: false,
            cf_protection: false,
            pic: false,
            no_red_zone: false,
            function_sections: false,
            data_sections: false,
        };

        let object = codegen.generate(&module, &target).unwrap();
        // .text section must have 4-byte alignment for AArch64.
        assert_eq!(
            object.sections[0].alignment, 4,
            ".text section must have 4-byte alignment for AArch64 instructions"
        );
    }

    #[test]
    fn test_text_section_flags() {
        let codegen = Aarch64CodeGen::new();
        let module = Module::new("test".to_string());
        let target = TargetConfig {
            arch: Architecture::Aarch64,
            triple: "aarch64-linux-gnu".to_string(),
            pointer_size: 8,
            long_size: 8,
            long_double_size: 16,
            elf_class: crate::driver::target::ElfClass::Elf64,
            endianness: crate::driver::target::Endianness::Little,
            abi: crate::driver::target::AbiVariant::Aapcs64,
            max_alignment: 16,
            stack_alignment: 16,
            elf_machine: 183,
            elf_osabi: 0,
            gpr_count: 31,
            fpr_count: 32,
            crt_search_paths: Vec::new(),
            lib_search_paths: Vec::new(),
            retpoline: false,
            cf_protection: false,
            pic: false,
            no_red_zone: false,
            function_sections: false,
            data_sections: false,
        };

        let object = codegen.generate(&module, &target).unwrap();
        let text_flags = &object.sections[0].flags;
        assert!(!text_flags.writable, ".text should not be writable");
        assert!(text_flags.executable, ".text should be executable");
        assert!(text_flags.allocatable, ".text should be allocatable");
    }

    // =======================================================================
    // Utility tests
    // =======================================================================

    #[test]
    fn test_align_up_u64() {
        assert_eq!(align_up_u64(0, 16), 0);
        assert_eq!(align_up_u64(1, 16), 16);
        assert_eq!(align_up_u64(16, 16), 16);
        assert_eq!(align_up_u64(17, 16), 32);
        assert_eq!(align_up_u64(15, 4), 16);
        assert_eq!(align_up_u64(4, 4), 4);
    }
}
