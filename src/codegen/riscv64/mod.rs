//! # RISC-V 64-bit Code Generation Backend
//!
//! Implements the [`CodeGen`] trait for the RISC-V 64 (RV64GC) architecture.
//! Generates ELF64 machine code using the LP64D ABI.
//!
//! ## Register File
//! - 32 General-Purpose Registers (x0-x31), x0 hardwired to zero
//! - 32 Floating-Point Registers (f0-f31) for D extension
//!
//! ## ISA Extensions
//! - RV64I: Base integer instruction set
//! - M: Integer multiply/divide
//! - A: Atomic operations
//! - F: Single-precision floating-point
//! - D: Double-precision floating-point
//! - C: Compressed instructions (optional, for code size reduction)
//!
//! ## Pipeline
//! ```text
//! IR Function → Instruction Selection (isel.rs)
//!            → Register Allocation (regalloc.rs)
//!            → Prologue/Epilogue generation
//!            → Instruction Encoding (encoding.rs)
//!            → Machine Code Bytes
//! ```
//!
//! ## Zero External Dependencies
//!
//! This module uses only the Rust standard library (`std`) and internal
//! crate modules. No external crates are imported.

pub mod abi;
pub mod encoding;
pub mod isel;

use std::collections::HashMap;

use crate::codegen::regalloc::{
    compute_live_intervals, linear_scan_allocate, AllocationResult, PhysReg, RegClass, RegisterInfo,
};
use crate::codegen::{
    Architecture, CodeGen, CodeGenError, MachineInstr, MachineOperand, ObjectCode, Relocation,
    Section, SectionFlags, SectionType, Symbol, SymbolBinding, SymbolType, SymbolVisibility,
};
use crate::driver::target::TargetConfig;
use crate::ir::{Function, GlobalVariable, IrType, Module};

// ---------------------------------------------------------------------------
// RISC-V 64 General-Purpose Register Constants (x0-x31)
// ---------------------------------------------------------------------------
// GPRs use PhysReg(0..31). x0 is hardwired to zero.

/// x0 — hardwired zero register (always reads as 0, writes are discarded).
pub const X0: PhysReg = PhysReg(0);
/// x1 — return address register (ra).
pub const X1: PhysReg = PhysReg(1);
/// x2 — stack pointer register (sp).
pub const X2: PhysReg = PhysReg(2);
/// x3 — global pointer register (gp).
pub const X3: PhysReg = PhysReg(3);
/// x4 — thread pointer register (tp).
pub const X4: PhysReg = PhysReg(4);
/// x5 — temporary register t0.
pub const X5: PhysReg = PhysReg(5);
/// x6 — temporary register t1.
pub const X6: PhysReg = PhysReg(6);
/// x7 — temporary register t2.
pub const X7: PhysReg = PhysReg(7);
/// x8 — callee-saved register s0 / frame pointer (fp).
pub const X8: PhysReg = PhysReg(8);
/// x9 — callee-saved register s1.
pub const X9: PhysReg = PhysReg(9);
/// x10 — argument/return register a0.
pub const X10: PhysReg = PhysReg(10);
/// x11 — argument/return register a1.
pub const X11: PhysReg = PhysReg(11);
/// x12 — argument register a2.
pub const X12: PhysReg = PhysReg(12);
/// x13 — argument register a3.
pub const X13: PhysReg = PhysReg(13);
/// x14 — argument register a4.
pub const X14: PhysReg = PhysReg(14);
/// x15 — argument register a5.
pub const X15: PhysReg = PhysReg(15);
/// x16 — argument register a6.
pub const X16: PhysReg = PhysReg(16);
/// x17 — argument register a7.
pub const X17: PhysReg = PhysReg(17);
/// x18 — callee-saved register s2.
pub const X18: PhysReg = PhysReg(18);
/// x19 — callee-saved register s3.
pub const X19: PhysReg = PhysReg(19);
/// x20 — callee-saved register s4.
pub const X20: PhysReg = PhysReg(20);
/// x21 — callee-saved register s5.
pub const X21: PhysReg = PhysReg(21);
/// x22 — callee-saved register s6.
pub const X22: PhysReg = PhysReg(22);
/// x23 — callee-saved register s7.
pub const X23: PhysReg = PhysReg(23);
/// x24 — callee-saved register s8.
pub const X24: PhysReg = PhysReg(24);
/// x25 — callee-saved register s9.
pub const X25: PhysReg = PhysReg(25);
/// x26 — callee-saved register s10.
pub const X26: PhysReg = PhysReg(26);
/// x27 — callee-saved register s11.
pub const X27: PhysReg = PhysReg(27);
/// x28 — temporary register t3.
pub const X28: PhysReg = PhysReg(28);
/// x29 — temporary register t4.
pub const X29: PhysReg = PhysReg(29);
/// x30 — temporary register t5.
pub const X30: PhysReg = PhysReg(30);
/// x31 — temporary register t6.
pub const X31: PhysReg = PhysReg(31);

// ---------------------------------------------------------------------------
// RISC-V 64 Floating-Point Register Constants (f0-f31)
// ---------------------------------------------------------------------------
// FPRs use PhysReg(32..63) to share the PhysReg namespace with GPRs.

/// f0 — FP temporary register ft0.
pub const F0: PhysReg = PhysReg(32);
/// f1 — FP temporary register ft1.
pub const F1: PhysReg = PhysReg(33);
/// f2 — FP temporary register ft2.
pub const F2: PhysReg = PhysReg(34);
/// f3 — FP temporary register ft3.
pub const F3: PhysReg = PhysReg(35);
/// f4 — FP temporary register ft4.
pub const F4: PhysReg = PhysReg(36);
/// f5 — FP temporary register ft5.
pub const F5: PhysReg = PhysReg(37);
/// f6 — FP temporary register ft6.
pub const F6: PhysReg = PhysReg(38);
/// f7 — FP temporary register ft7.
pub const F7: PhysReg = PhysReg(39);
/// f8 — FP callee-saved register fs0.
pub const F8: PhysReg = PhysReg(40);
/// f9 — FP callee-saved register fs1.
pub const F9: PhysReg = PhysReg(41);
/// f10 — FP argument/return register fa0.
pub const F10: PhysReg = PhysReg(42);
/// f11 — FP argument/return register fa1.
pub const F11: PhysReg = PhysReg(43);
/// f12 — FP argument register fa2.
pub const F12: PhysReg = PhysReg(44);
/// f13 — FP argument register fa3.
pub const F13: PhysReg = PhysReg(45);
/// f14 — FP argument register fa4.
pub const F14: PhysReg = PhysReg(46);
/// f15 — FP argument register fa5.
pub const F15: PhysReg = PhysReg(47);
/// f16 — FP argument register fa6.
pub const F16: PhysReg = PhysReg(48);
/// f17 — FP argument register fa7.
pub const F17: PhysReg = PhysReg(49);
/// f18 — FP callee-saved register fs2.
pub const F18: PhysReg = PhysReg(50);
/// f19 — FP callee-saved register fs3.
pub const F19: PhysReg = PhysReg(51);
/// f20 — FP callee-saved register fs4.
pub const F20: PhysReg = PhysReg(52);
/// f21 — FP callee-saved register fs5.
pub const F21: PhysReg = PhysReg(53);
/// f22 — FP callee-saved register fs6.
pub const F22: PhysReg = PhysReg(54);
/// f23 — FP callee-saved register fs7.
pub const F23: PhysReg = PhysReg(55);
/// f24 — FP callee-saved register fs8.
pub const F24: PhysReg = PhysReg(56);
/// f25 — FP callee-saved register fs9.
pub const F25: PhysReg = PhysReg(57);
/// f26 — FP callee-saved register fs10.
pub const F26: PhysReg = PhysReg(58);
/// f27 — FP callee-saved register fs11.
pub const F27: PhysReg = PhysReg(59);
/// f28 — FP temporary register ft8.
pub const F28: PhysReg = PhysReg(60);
/// f29 — FP temporary register ft9.
pub const F29: PhysReg = PhysReg(61);
/// f30 — FP temporary register ft10.
pub const F30: PhysReg = PhysReg(62);
/// f31 — FP temporary register ft11.
pub const F31: PhysReg = PhysReg(63);

// ---------------------------------------------------------------------------
// ABI Register Aliases
// ---------------------------------------------------------------------------

/// zero register alias — hardwired to 0 (same as x0).
pub const ZERO: PhysReg = X0;
/// Return address register alias (same as x1).
pub const RA: PhysReg = X1;
/// Stack pointer register alias (same as x2).
pub const SP: PhysReg = X2;
/// Global pointer register alias (same as x3).
pub const GP: PhysReg = X3;
/// Thread pointer register alias (same as x4).
pub const TP: PhysReg = X4;
/// Frame pointer register alias — s0 doubles as fp (same as x8).
pub const FP: PhysReg = X8;

// ---------------------------------------------------------------------------
// RegisterInfo Construction
// ---------------------------------------------------------------------------

/// Constructs a [`RegisterInfo`] descriptor for the RISC-V 64 register file.
///
/// The register sets are ordered with caller-saved (temporary) registers first
/// to minimize callee-saved spill overhead during allocation. Registers x0
/// (zero), x1 (ra), x2 (sp), x3 (gp), and x4 (tp) are excluded from the
/// allocatable set because they have dedicated architectural roles.
///
/// # LP64D ABI Register Classification
///
/// | Registers     | ABI Names   | Role           | Saved By |
/// |---------------|-------------|----------------|----------|
/// | x0            | zero        | Hardwired zero | N/A      |
/// | x1            | ra          | Return address | Caller   |
/// | x2            | sp          | Stack pointer  | Callee   |
/// | x3            | gp          | Global pointer | N/A      |
/// | x4            | tp          | Thread pointer | N/A      |
/// | x5-x7         | t0-t2       | Temporaries    | Caller   |
/// | x8-x9         | s0-s1       | Callee-saved   | Callee   |
/// | x10-x17       | a0-a7       | Arguments      | Caller   |
/// | x18-x27       | s2-s11      | Callee-saved   | Callee   |
/// | x28-x31       | t3-t6       | Temporaries    | Caller   |
/// | f0-f7         | ft0-ft7     | FP Temporaries | Caller   |
/// | f8-f9         | fs0-fs1     | FP Callee-saved| Callee   |
/// | f10-f17       | fa0-fa7     | FP Arguments   | Caller   |
/// | f18-f27       | fs2-fs11    | FP Callee-saved| Callee   |
/// | f28-f31       | ft8-ft11    | FP Temporaries | Caller   |
pub fn riscv64_register_info() -> RegisterInfo {
    RegisterInfo {
        // Allocatable integer registers in allocation priority order.
        // Caller-saved registers listed first to minimize save/restore overhead.
        int_regs: vec![
            // Caller-saved temporaries (preferred — no save/restore cost):
            X5, X6, X7, // t0-t2
            X28, X29, X30, X31, // t3-t6
            // Argument registers (caller-saved):
            X10, X11, X12, X13, X14, X15, X16, X17, // a0-a7
            // Callee-saved registers (used when caller-saved are exhausted):
            X8, X9, // s0-s1
            X18, X19, X20, X21, X22, X23, X24, X25, X26, X27, // s2-s11
        ],
        // Allocatable floating-point registers in allocation priority order.
        float_regs: vec![
            // Caller-saved FP temporaries:
            F0, F1, F2, F3, F4, F5, F6, F7, // ft0-ft7
            F28, F29, F30, F31, // ft8-ft11
            // FP argument registers (caller-saved):
            F10, F11, F12, F13, F14, F15, F16, F17, // fa0-fa7
            // FP callee-saved registers:
            F8, F9, // fs0-fs1
            F18, F19, F20, F21, F22, F23, F24, F25, F26, F27, // fs2-fs11
        ],
        // Integer callee-saved registers (s0-s11 per LP64D ABI).
        callee_saved_int: vec![
            X8, X9, // s0-s1
            X18, X19, X20, X21, X22, X23, X24, X25, X26, X27, // s2-s11
        ],
        // FP callee-saved registers (fs0-fs11 per LP64D ABI).
        callee_saved_float: vec![
            F8, F9, // fs0-fs1
            F18, F19, F20, F21, F22, F23, F24, F25, F26, F27, // fs2-fs11
        ],
        reg_names: build_riscv64_reg_names(),
    }
}

/// Builds the mapping from [`PhysReg`] identifiers to LP64D ABI register
/// name strings for all 64 registers (32 GPRs + 32 FPRs).
fn build_riscv64_reg_names() -> HashMap<PhysReg, &'static str> {
    let mut map = HashMap::new();

    // General-purpose registers x0-x31
    map.insert(PhysReg(0), "zero");
    map.insert(PhysReg(1), "ra");
    map.insert(PhysReg(2), "sp");
    map.insert(PhysReg(3), "gp");
    map.insert(PhysReg(4), "tp");
    map.insert(PhysReg(5), "t0");
    map.insert(PhysReg(6), "t1");
    map.insert(PhysReg(7), "t2");
    map.insert(PhysReg(8), "s0");
    map.insert(PhysReg(9), "s1");
    map.insert(PhysReg(10), "a0");
    map.insert(PhysReg(11), "a1");
    map.insert(PhysReg(12), "a2");
    map.insert(PhysReg(13), "a3");
    map.insert(PhysReg(14), "a4");
    map.insert(PhysReg(15), "a5");
    map.insert(PhysReg(16), "a6");
    map.insert(PhysReg(17), "a7");
    map.insert(PhysReg(18), "s2");
    map.insert(PhysReg(19), "s3");
    map.insert(PhysReg(20), "s4");
    map.insert(PhysReg(21), "s5");
    map.insert(PhysReg(22), "s6");
    map.insert(PhysReg(23), "s7");
    map.insert(PhysReg(24), "s8");
    map.insert(PhysReg(25), "s9");
    map.insert(PhysReg(26), "s10");
    map.insert(PhysReg(27), "s11");
    map.insert(PhysReg(28), "t3");
    map.insert(PhysReg(29), "t4");
    map.insert(PhysReg(30), "t5");
    map.insert(PhysReg(31), "t6");

    // Floating-point registers f0-f31
    map.insert(PhysReg(32), "ft0");
    map.insert(PhysReg(33), "ft1");
    map.insert(PhysReg(34), "ft2");
    map.insert(PhysReg(35), "ft3");
    map.insert(PhysReg(36), "ft4");
    map.insert(PhysReg(37), "ft5");
    map.insert(PhysReg(38), "ft6");
    map.insert(PhysReg(39), "ft7");
    map.insert(PhysReg(40), "fs0");
    map.insert(PhysReg(41), "fs1");
    map.insert(PhysReg(42), "fa0");
    map.insert(PhysReg(43), "fa1");
    map.insert(PhysReg(44), "fa2");
    map.insert(PhysReg(45), "fa3");
    map.insert(PhysReg(46), "fa4");
    map.insert(PhysReg(47), "fa5");
    map.insert(PhysReg(48), "fa6");
    map.insert(PhysReg(49), "fa7");
    map.insert(PhysReg(50), "fs2");
    map.insert(PhysReg(51), "fs3");
    map.insert(PhysReg(52), "fs4");
    map.insert(PhysReg(53), "fs5");
    map.insert(PhysReg(54), "fs6");
    map.insert(PhysReg(55), "fs7");
    map.insert(PhysReg(56), "fs8");
    map.insert(PhysReg(57), "fs9");
    map.insert(PhysReg(58), "fs10");
    map.insert(PhysReg(59), "fs11");
    map.insert(PhysReg(60), "ft8");
    map.insert(PhysReg(61), "ft9");
    map.insert(PhysReg(62), "ft10");
    map.insert(PhysReg(63), "ft11");

    map
}

// ---------------------------------------------------------------------------
// Riscv64CodeGen — Main Code Generation Struct
// ---------------------------------------------------------------------------

/// RISC-V 64-bit code generation backend implementing the [`CodeGen`] trait.
///
/// Coordinates the four phases of RISC-V 64 code generation:
/// 1. **Register allocation** — assigns physical registers to SSA values
/// 2. **Prologue/epilogue generation** — LP64D ABI-compliant stack frame setup
/// 3. **Instruction selection** — translates IR to RV64GC machine instructions
/// 4. **Machine code encoding** — emits binary instruction bytes
///
/// The output is an [`ObjectCode`] containing `.text`, `.data`, `.rodata`,
/// and `.bss` sections with symbol definitions and relocation entries ready
/// for the integrated linker.
pub struct Riscv64CodeGen;

impl Riscv64CodeGen {
    /// Creates a new RISC-V 64 code generator instance.
    pub fn new() -> Self {
        Riscv64CodeGen
    }
}

/// Returns the register class for a given physical register.
///
/// RISC-V 64 uses PhysReg IDs 0-31 for GPRs (integer class) and 32-63 for
/// FPRs (float class). This classification is used during register allocation
/// to ensure values are assigned to the correct register file.
fn classify_register(reg: PhysReg) -> RegClass {
    if reg.0 < 32 {
        RegClass::Integer
    } else {
        RegClass::Float
    }
}

// ---------------------------------------------------------------------------
// Prologue / Epilogue Generation Helpers
// ---------------------------------------------------------------------------
// These produce MachineInstr sequences for LP64D ABI-compliant function
// entry and exit. The prologue saves the return address, frame pointer,
// and any callee-saved registers used by the register allocator, then
// allocates the stack frame. The epilogue reverses these operations.

/// Computes the stack frame size for a function, accounting for:
/// - Return address (ra) save slot: 8 bytes
/// - Frame pointer (s0) save slot: 8 bytes
/// - Callee-saved registers used: 8 bytes each
/// - Spill slots from register allocation: 8 bytes each
/// - 16-byte alignment as required by LP64D ABI
fn compute_frame_size(alloc_result: &AllocationResult) -> i64 {
    // Base frame: ra + s0 = 16 bytes
    let mut frame_size: i64 = 16;

    // Callee-saved registers (excluding s0 which is always saved as part of
    // the base frame when used as the frame pointer)
    for &reg in &alloc_result.used_callee_saved {
        if reg == FP {
            continue; // Already counted in the base 16 bytes
        }
        // Both integer (8 bytes for 64-bit GPR) and floating-point (8 bytes
        // for double-precision FPR) callee-saved registers require 8 bytes.
        let _class = classify_register(reg);
        frame_size += 8;
    }

    // Spill slots from register allocation (8 bytes each)
    frame_size += alloc_result.num_spill_slots as i64 * 8;

    // Align frame size to 16 bytes (LP64D ABI requirement)
    frame_size = (frame_size + 15) & !15;

    frame_size
}

/// Generates the function prologue as a sequence of machine instructions.
///
/// The LP64D prologue layout (growing downward from high to low addresses):
/// ```text
///     [previous frame]
///     +------------------+ ← old sp
///     | ra save          | sp + frame_size - 8
///     +------------------+
///     | s0/fp save       | sp + frame_size - 16
///     +------------------+
///     | callee-saved #1  | sp + frame_size - 24
///     +------------------+
///     | callee-saved #2  | sp + frame_size - 32
///     +------------------+
///     | ...              |
///     +------------------+
///     | spill slots      |
///     +------------------+ ← new sp (16-byte aligned)
/// ```
fn generate_prologue(alloc_result: &AllocationResult, _target: &TargetConfig) -> Vec<MachineInstr> {
    let frame_size = compute_frame_size(alloc_result);
    let mut instrs = Vec::new();

    if frame_size == 0 {
        return instrs;
    }

    // Adjust stack pointer: addi sp, sp, -frame_size
    // If frame_size > 2047 (max 12-bit signed immediate), use LI + ADD sequence.
    if frame_size <= 2047 {
        instrs.push(MachineInstr::with_operands(
            isel::Riscv64Opcode::ADDI.as_u32(),
            vec![
                MachineOperand::Register(SP),
                MachineOperand::Register(SP),
                MachineOperand::Immediate(-frame_size),
            ],
        ));
    } else {
        // Use a temporary register (t0/x5) for large frame sizes
        instrs.push(MachineInstr::with_operands(
            isel::Riscv64Opcode::LI.as_u32(),
            vec![
                MachineOperand::Register(X5),
                MachineOperand::Immediate(-frame_size),
            ],
        ));
        instrs.push(MachineInstr::with_operands(
            isel::Riscv64Opcode::ADD.as_u32(),
            vec![
                MachineOperand::Register(SP),
                MachineOperand::Register(SP),
                MachineOperand::Register(X5),
            ],
        ));
    }

    // Save return address: sd ra, frame_size-8(sp)
    instrs.push(MachineInstr::with_operands(
        isel::Riscv64Opcode::SD.as_u32(),
        vec![
            MachineOperand::Register(RA),
            MachineOperand::Memory {
                base: SP,
                offset: (frame_size - 8) as i32,
            },
        ],
    ));

    // Save frame pointer: sd s0, frame_size-16(sp)
    instrs.push(MachineInstr::with_operands(
        isel::Riscv64Opcode::SD.as_u32(),
        vec![
            MachineOperand::Register(FP),
            MachineOperand::Memory {
                base: SP,
                offset: (frame_size - 16) as i32,
            },
        ],
    ));

    // Save callee-saved registers used by the allocator
    let mut save_offset = frame_size - 24;
    for &reg in &alloc_result.used_callee_saved {
        if reg == FP {
            continue; // Already saved above
        }
        if save_offset < 0 {
            break;
        }
        // Use SD for integer registers, FSD for FP registers
        let store_opcode = match classify_register(reg) {
            RegClass::Float => isel::Riscv64Opcode::FSD.as_u32(),
            RegClass::Integer => isel::Riscv64Opcode::SD.as_u32(),
        };
        instrs.push(MachineInstr::with_operands(
            store_opcode,
            vec![
                MachineOperand::Register(reg),
                MachineOperand::Memory {
                    base: SP,
                    offset: save_offset as i32,
                },
            ],
        ));
        save_offset -= 8;
    }

    // Set up frame pointer: addi s0, sp, frame_size
    if frame_size <= 2047 {
        instrs.push(MachineInstr::with_operands(
            isel::Riscv64Opcode::ADDI.as_u32(),
            vec![
                MachineOperand::Register(FP),
                MachineOperand::Register(SP),
                MachineOperand::Immediate(frame_size),
            ],
        ));
    } else {
        // Large frame: use LI + ADD
        instrs.push(MachineInstr::with_operands(
            isel::Riscv64Opcode::LI.as_u32(),
            vec![
                MachineOperand::Register(X5),
                MachineOperand::Immediate(frame_size),
            ],
        ));
        instrs.push(MachineInstr::with_operands(
            isel::Riscv64Opcode::ADD.as_u32(),
            vec![
                MachineOperand::Register(FP),
                MachineOperand::Register(SP),
                MachineOperand::Register(X5),
            ],
        ));
    }

    instrs
}

/// Generates the function epilogue as a sequence of machine instructions.
///
/// Reverses the prologue: restores callee-saved registers, restores ra and
/// s0, deallocates the stack frame, and returns via the RET pseudo-instruction.
fn generate_epilogue(alloc_result: &AllocationResult, _target: &TargetConfig) -> Vec<MachineInstr> {
    let frame_size = compute_frame_size(alloc_result);
    let mut instrs = Vec::new();

    if frame_size > 0 {
        // Restore callee-saved registers (in reverse order of saving)
        let mut restore_offset = frame_size - 24;
        for &reg in &alloc_result.used_callee_saved {
            if reg == FP {
                continue; // Restored separately below
            }
            if restore_offset < 0 {
                break;
            }
            // Use LD for integer registers, FLD for FP registers
            let load_opcode = match classify_register(reg) {
                RegClass::Float => isel::Riscv64Opcode::FLD.as_u32(),
                RegClass::Integer => isel::Riscv64Opcode::LD.as_u32(),
            };
            instrs.push(MachineInstr::with_operands(
                load_opcode,
                vec![
                    MachineOperand::Register(reg),
                    MachineOperand::Memory {
                        base: SP,
                        offset: restore_offset as i32,
                    },
                ],
            ));
            restore_offset -= 8;
        }

        // Restore return address: ld ra, frame_size-8(sp)
        instrs.push(MachineInstr::with_operands(
            isel::Riscv64Opcode::LD.as_u32(),
            vec![
                MachineOperand::Register(RA),
                MachineOperand::Memory {
                    base: SP,
                    offset: (frame_size - 8) as i32,
                },
            ],
        ));

        // Restore frame pointer: ld s0, frame_size-16(sp)
        instrs.push(MachineInstr::with_operands(
            isel::Riscv64Opcode::LD.as_u32(),
            vec![
                MachineOperand::Register(FP),
                MachineOperand::Memory {
                    base: SP,
                    offset: (frame_size - 16) as i32,
                },
            ],
        ));

        // Deallocate stack frame: addi sp, sp, frame_size
        if frame_size <= 2047 {
            instrs.push(MachineInstr::with_operands(
                isel::Riscv64Opcode::ADDI.as_u32(),
                vec![
                    MachineOperand::Register(SP),
                    MachineOperand::Register(SP),
                    MachineOperand::Immediate(frame_size),
                ],
            ));
        } else {
            instrs.push(MachineInstr::with_operands(
                isel::Riscv64Opcode::LI.as_u32(),
                vec![
                    MachineOperand::Register(X5),
                    MachineOperand::Immediate(frame_size),
                ],
            ));
            instrs.push(MachineInstr::with_operands(
                isel::Riscv64Opcode::ADD.as_u32(),
                vec![
                    MachineOperand::Register(SP),
                    MachineOperand::Register(SP),
                    MachineOperand::Register(X5),
                ],
            ));
        }
    }

    // Return: ret (pseudo for jalr x0, ra, 0)
    instrs.push(MachineInstr::with_operands(
        isel::Riscv64Opcode::RET.as_u32(),
        vec![],
    ));

    instrs
}

// ---------------------------------------------------------------------------
// Global Variable Section Helpers
// ---------------------------------------------------------------------------

/// Emits the `.data` section for initialized, mutable global variables.
fn emit_data_section(globals: &[GlobalVariable]) -> (Section, Vec<Symbol>) {
    let mut data = Vec::new();
    let mut symbols = Vec::new();

    for global in globals {
        // Skip extern declarations and const globals (those go in .rodata)
        if global.is_extern {
            continue;
        }
        let init = match &global.initializer {
            Some(c) => c,
            None => continue, // Uninitialized globals go to .bss
        };
        // String constants go to .rodata, not .data
        {
            use crate::ir::Constant;
            if matches!(init, Constant::String(_)) {
                continue;
            }
        }

        let offset = data.len() as u64;

        // Emit initializer bytes based on the IR type
        let init_bytes = emit_global_initializer(global);
        if init_bytes.is_empty() {
            continue;
        }

        symbols.push(Symbol {
            name: global.name.clone(),
            section_index: 0, // Will be adjusted by caller
            offset,
            size: init_bytes.len() as u64,
            binding: if global.is_static {
                SymbolBinding::Local
            } else {
                SymbolBinding::Global
            },
            symbol_type: SymbolType::Object,
            visibility: SymbolVisibility::Default,
            is_definition: true,
        });

        data.extend_from_slice(&init_bytes);
    }

    let section = Section {
        name: ".data".to_string(),
        data,
        section_type: SectionType::Data,
        alignment: 8,
        flags: SectionFlags::data(),
    };

    (section, symbols)
}

/// Emits the `.rodata` section for read-only global data.
///
/// This handles string literals and const-qualified globals that are placed
/// in the read-only data section. These are distinguished from `.data` by
/// their immutability — the ELF loader maps `.rodata` as non-writable.
fn emit_rodata_section(globals: &[GlobalVariable]) -> (Section, Vec<Symbol>) {
    let mut data = Vec::new();
    let mut symbols = Vec::new();

    for global in globals {
        if global.is_extern {
            continue;
        }
        // Check if this is a string constant or other read-only initializer.
        // String literals (Constant::String) are placed in .rodata.
        if let Some(init) = &global.initializer {
            use crate::ir::Constant;
            let is_rodata = matches!(init, Constant::String(_));
            if !is_rodata {
                continue;
            }
            let offset = data.len() as u64;
            let init_bytes = emit_global_initializer(global);
            if init_bytes.is_empty() {
                continue;
            }
            symbols.push(Symbol {
                name: global.name.clone(),
                section_index: 0,
                offset,
                size: init_bytes.len() as u64,
                binding: if global.is_static {
                    SymbolBinding::Local
                } else {
                    SymbolBinding::Global
                },
                symbol_type: SymbolType::Object,
                visibility: SymbolVisibility::Default,
                is_definition: true,
            });
            data.extend_from_slice(&init_bytes);
        }
    }

    let section = Section {
        name: ".rodata".to_string(),
        data,
        section_type: SectionType::Rodata,
        alignment: 8,
        flags: SectionFlags::rodata(),
    };

    (section, symbols)
}

/// Emits the `.bss` section descriptor for uninitialized global variables.
fn emit_bss_section(globals: &[GlobalVariable]) -> (Section, Vec<Symbol>) {
    let mut bss_size: u64 = 0;
    let mut symbols = Vec::new();

    for global in globals {
        if global.is_extern {
            continue;
        }
        if global.initializer.is_some() {
            continue; // Initialized globals go to .data
        }

        let type_size = ir_type_size(&global.ty);
        let offset = bss_size;
        bss_size += type_size;
        // Align to 8 bytes
        bss_size = (bss_size + 7) & !7;

        symbols.push(Symbol {
            name: global.name.clone(),
            section_index: 0, // Adjusted by caller
            offset,
            size: type_size,
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

    // .bss section has no data bytes — the linker zero-fills at load time
    let section = Section {
        name: ".bss".to_string(),
        data: Vec::new(),
        section_type: SectionType::Bss,
        alignment: 8,
        flags: SectionFlags::bss(),
    };

    (section, symbols)
}

/// Emits initializer bytes for a global variable based on its constant value.
fn emit_global_initializer(global: &GlobalVariable) -> Vec<u8> {
    use crate::ir::Constant;

    let init = match &global.initializer {
        Some(c) => c,
        None => return Vec::new(),
    };

    match init {
        Constant::Integer { value, ty } => {
            let byte_count = ir_type_size(ty) as usize;
            let val_bytes = value.to_le_bytes();
            val_bytes[..byte_count.min(8)].to_vec()
        }
        Constant::Float { value, ty } => {
            match ty {
                IrType::F32 => (*value as f32).to_le_bytes().to_vec(),
                _ => value.to_le_bytes().to_vec(), // F64 or default
            }
        }
        Constant::Bool(val) => vec![if *val { 1u8 } else { 0u8 }],
        Constant::ZeroInit(ty) => {
            let size = ir_type_size(ty) as usize;
            vec![0u8; size]
        }
        Constant::String(bytes) => bytes.clone(),
        Constant::Null(_) | Constant::Undef(_) => {
            let size = ir_type_size(&global.ty) as usize;
            vec![0u8; size]
        }
        Constant::GlobalRef(_) => {
            // Global reference — requires a relocation; emit 8 zero bytes
            // that the linker will patch with the final address.
            vec![0u8; 8]
        }
    }
}

/// Returns the size in bytes of an IR type for the RISC-V 64 target.
fn ir_type_size(ty: &IrType) -> u64 {
    match ty {
        IrType::Void => 0,
        IrType::I1 => 1,
        IrType::I8 => 1,
        IrType::I16 => 2,
        IrType::I32 => 4,
        IrType::I64 => 8,
        IrType::F32 => 4,
        IrType::F64 => 8,
        IrType::Pointer(_) => 8, // 64-bit pointers on RV64
        IrType::Array { element, count } => ir_type_size(element) * (*count as u64),
        IrType::Struct { fields, packed } => {
            let mut size: u64 = 0;
            for field in fields {
                let field_size = ir_type_size(field);
                if !packed {
                    let align = field_size.max(1);
                    size = (size + align - 1) & !(align - 1);
                }
                size += field_size;
            }
            if !packed {
                // Align total struct size to its largest field alignment
                let max_align = fields
                    .iter()
                    .map(|f| ir_type_size(f).max(1))
                    .max()
                    .unwrap_or(1);
                size = (size + max_align - 1) & !(max_align - 1);
            }
            size
        }
        IrType::Function { .. } => 0, // Function types have zero size; use Pointer for fn ptrs
        IrType::Label => 0,
    }
}

// ---------------------------------------------------------------------------
// CodeGen Trait Implementation
// ---------------------------------------------------------------------------

impl CodeGen for Riscv64CodeGen {
    /// Generates RISC-V 64 machine code for the given IR module.
    ///
    /// Iterates over all function definitions in the module, performing:
    /// 1. Live interval computation for register allocation
    /// 2. Linear scan register allocation using the RISC-V 64 register file
    /// 3. LP64D ABI-compliant prologue generation
    /// 4. Instruction selection (IR → RV64GC machine instructions)
    /// 5. LP64D ABI-compliant epilogue generation
    /// 6. Machine code encoding (instructions → bytes)
    ///
    /// Global variables are emitted into `.data`, `.rodata`, and `.bss` sections.
    fn generate(&self, module: &Module, target: &TargetConfig) -> Result<ObjectCode, CodeGenError> {
        let reg_info = riscv64_register_info();
        let mut sections = Vec::new();
        let mut symbols = Vec::new();
        let mut all_relocations: Vec<Relocation> = Vec::new();
        let mut text_data: Vec<u8> = Vec::new();

        // ----- Process function definitions -----
        let functions: &Vec<Function> = &module.functions;
        for function in functions {
            if !function.is_definition {
                // External declaration — emit an undefined symbol reference
                // but no code. The linker resolves this against library objects.
                symbols.push(Symbol {
                    name: function.name.clone(),
                    section_index: 0,
                    offset: 0,
                    size: 0,
                    binding: SymbolBinding::Global,
                    symbol_type: SymbolType::Function,
                    visibility: SymbolVisibility::Default,
                    is_definition: false,
                });
                continue;
            }

            // Step 1: Compute live intervals for the function's SSA values
            let mut intervals = compute_live_intervals(function);

            // Step 2: Perform linear scan register allocation
            let alloc_result = linear_scan_allocate(&mut intervals, &reg_info);

            // Step 3: Generate LP64D prologue (stack frame setup)
            let prologue = generate_prologue(&alloc_result, target);

            // Step 4: Run instruction selection (IR → RV64GC machine instrs)
            let mut selector = isel::Riscv64InstructionSelector::new(&alloc_result, target);
            let body_instrs = selector.select_function(function);

            // Step 5: Generate LP64D epilogue (stack frame teardown + return)
            let epilogue = generate_epilogue(&alloc_result, target);

            // Step 6: Concatenate prologue + body + epilogue
            let mut all_instrs =
                Vec::with_capacity(prologue.len() + body_instrs.len() + epilogue.len());
            all_instrs.extend(prologue);
            all_instrs.extend(body_instrs);
            all_instrs.extend(epilogue);

            // Step 7: Encode machine instructions to binary bytes
            let mut encoder = encoding::Riscv64Encoder::new();
            let func_code = encoder.encode_function(&all_instrs);

            // Step 8: Record the function symbol
            let func_offset = text_data.len() as u64;
            symbols.push(Symbol {
                name: function.name.clone(),
                section_index: 0, // .text section index (temporary, fixed up below)
                offset: func_offset,
                size: func_code.len() as u64,
                binding: SymbolBinding::Global,
                symbol_type: SymbolType::Function,
                visibility: SymbolVisibility::Default,
                is_definition: true,
            });

            // Step 9: Collect relocations from the encoder, adjusting offsets
            // to account for the function's position within the .text section
            for reloc in encoder.get_relocations() {
                all_relocations.push(Relocation {
                    offset: reloc.offset + func_offset,
                    symbol: reloc.symbol.clone(),
                    reloc_type: reloc.reloc_type,
                    addend: reloc.addend,
                    section_index: 0, // .text section
                });
            }

            // Also collect relocations from instruction selection
            for reloc in selector.relocations() {
                all_relocations.push(Relocation {
                    offset: reloc.offset + func_offset,
                    symbol: reloc.symbol.clone(),
                    reloc_type: reloc.reloc_type,
                    addend: reloc.addend,
                    section_index: 0,
                });
            }

            // Append encoded function bytes to the .text data
            text_data.extend_from_slice(&func_code);
        }

        // Build the .text section
        let text_section_idx = sections.len();
        sections.push(Section {
            name: ".text".to_string(),
            data: text_data,
            section_type: SectionType::Text,
            alignment: 4, // RISC-V instructions are 4-byte aligned (2-byte if compressed)
            flags: SectionFlags::text(),
        });

        // Fix up section indices for *defined* function symbols to point at .text.
        // Undefined (extern) function symbols must keep their section_index as-is
        // because they are not bound to any section.
        for sym in &mut symbols {
            if sym.symbol_type == SymbolType::Function && sym.is_definition {
                sym.section_index = text_section_idx;
            }
        }

        // ----- Process global variables -----
        let (data_section, data_symbols) = emit_data_section(&module.globals);
        if !data_section.data.is_empty() {
            let data_idx = sections.len();
            sections.push(data_section);
            for mut sym in data_symbols {
                sym.section_index = data_idx;
                symbols.push(sym);
            }
        }

        let (rodata_section, rodata_symbols) = emit_rodata_section(&module.globals);
        if !rodata_section.data.is_empty() {
            let rodata_idx = sections.len();
            sections.push(rodata_section);
            for mut sym in rodata_symbols {
                sym.section_index = rodata_idx;
                symbols.push(sym);
            }
        }

        let (bss_section, bss_symbols) = emit_bss_section(&module.globals);
        if !bss_symbols.is_empty() {
            let bss_idx = sections.len();
            sections.push(bss_section);
            for mut sym in bss_symbols {
                sym.section_index = bss_idx;
                symbols.push(sym);
            }
        }

        Ok(ObjectCode {
            sections,
            symbols,
            relocations: all_relocations,
            target_arch: Architecture::Riscv64,
        })
    }

    /// Returns the target architecture for this backend.
    fn target_arch(&self) -> Architecture {
        Architecture::Riscv64
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Register alias tests ----

    #[test]
    fn test_zero_is_x0() {
        assert_eq!(ZERO, X0);
        assert_eq!(ZERO.0, 0);
    }

    #[test]
    fn test_ra_is_x1() {
        assert_eq!(RA, X1);
        assert_eq!(RA.0, 1);
    }

    #[test]
    fn test_sp_is_x2() {
        assert_eq!(SP, X2);
        assert_eq!(SP.0, 2);
    }

    #[test]
    fn test_gp_is_x3() {
        assert_eq!(GP, X3);
        assert_eq!(GP.0, 3);
    }

    #[test]
    fn test_tp_is_x4() {
        assert_eq!(TP, X4);
        assert_eq!(TP.0, 4);
    }

    #[test]
    fn test_fp_is_x8() {
        assert_eq!(FP, X8);
        assert_eq!(FP.0, 8);
    }

    // ---- Register numbering tests ----

    #[test]
    fn test_gpr_numbering() {
        assert_eq!(X0.0, 0);
        assert_eq!(X1.0, 1);
        assert_eq!(X10.0, 10);
        assert_eq!(X17.0, 17);
        assert_eq!(X31.0, 31);
    }

    #[test]
    fn test_fpr_numbering() {
        assert_eq!(F0.0, 32);
        assert_eq!(F1.0, 33);
        assert_eq!(F10.0, 42);
        assert_eq!(F17.0, 49);
        assert_eq!(F31.0, 63);
    }

    // ---- RegisterInfo tests ----

    #[test]
    fn test_register_info_excludes_special_regs() {
        let info = riscv64_register_info();
        // x0 (zero), x1 (ra), x2 (sp), x3 (gp), x4 (tp) must NOT be allocatable
        assert!(
            !info.int_regs.contains(&X0),
            "x0 (zero) must not be allocatable"
        );
        assert!(
            !info.int_regs.contains(&X1),
            "x1 (ra) must not be allocatable"
        );
        assert!(
            !info.int_regs.contains(&X2),
            "x2 (sp) must not be allocatable"
        );
        assert!(
            !info.int_regs.contains(&X3),
            "x3 (gp) must not be allocatable"
        );
        assert!(
            !info.int_regs.contains(&X4),
            "x4 (tp) must not be allocatable"
        );
    }

    #[test]
    fn test_allocatable_int_register_count() {
        let info = riscv64_register_info();
        // 32 GPRs - 5 special (x0, x1, x2, x3, x4) = 27 allocatable
        assert_eq!(
            info.int_regs.len(),
            27,
            "Expected 27 allocatable integer registers (32 - 5 special)"
        );
    }

    #[test]
    fn test_allocatable_float_register_count() {
        let info = riscv64_register_info();
        // All 32 FP registers are allocatable
        assert_eq!(
            info.float_regs.len(),
            32,
            "Expected 32 allocatable floating-point registers"
        );
    }

    #[test]
    fn test_callee_saved_int_registers() {
        let info = riscv64_register_info();
        // LP64D callee-saved: s0-s1 (x8-x9), s2-s11 (x18-x27) = 12 registers
        assert_eq!(
            info.callee_saved_int.len(),
            12,
            "Expected 12 callee-saved integer registers (s0-s11)"
        );
        assert!(info.callee_saved_int.contains(&X8)); // s0
        assert!(info.callee_saved_int.contains(&X9)); // s1
        assert!(info.callee_saved_int.contains(&X18)); // s2
        assert!(info.callee_saved_int.contains(&X27)); // s11
    }

    #[test]
    fn test_callee_saved_float_registers() {
        let info = riscv64_register_info();
        // LP64D FP callee-saved: fs0-fs1 (f8-f9), fs2-fs11 (f18-f27) = 12 registers
        assert_eq!(
            info.callee_saved_float.len(),
            12,
            "Expected 12 callee-saved floating-point registers (fs0-fs11)"
        );
        assert!(info.callee_saved_float.contains(&F8)); // fs0
        assert!(info.callee_saved_float.contains(&F9)); // fs1
        assert!(info.callee_saved_float.contains(&F18)); // fs2
        assert!(info.callee_saved_float.contains(&F27)); // fs11
    }

    #[test]
    fn test_caller_saved_first_in_allocation_order() {
        let info = riscv64_register_info();
        // Verify caller-saved temporaries appear before callee-saved in the
        // allocation order (first element should be t0=x5, a caller-saved reg).
        let first_int = info.int_regs[0];
        assert_eq!(first_int, X5, "First allocatable int reg should be t0 (x5)");

        let first_float = info.float_regs[0];
        assert_eq!(
            first_float, F0,
            "First allocatable float reg should be ft0 (f0)"
        );
    }

    #[test]
    fn test_all_gpr_names_present() {
        let info = riscv64_register_info();
        // All 32 GPR names should be in the reg_names map
        for i in 0..32u16 {
            assert!(
                info.reg_names.contains_key(&PhysReg(i)),
                "Missing register name for GPR x{} (PhysReg({}))",
                i,
                i
            );
        }
    }

    #[test]
    fn test_all_fpr_names_present() {
        let info = riscv64_register_info();
        // All 32 FPR names should be in the reg_names map
        for i in 32..64u16 {
            assert!(
                info.reg_names.contains_key(&PhysReg(i)),
                "Missing register name for FPR f{} (PhysReg({}))",
                i - 32,
                i
            );
        }
    }

    #[test]
    fn test_register_name_lookup() {
        let info = riscv64_register_info();
        assert_eq!(info.name(X0), "zero");
        assert_eq!(info.name(X1), "ra");
        assert_eq!(info.name(X2), "sp");
        assert_eq!(info.name(X8), "s0");
        assert_eq!(info.name(X10), "a0");
        assert_eq!(info.name(F10), "fa0");
        assert_eq!(info.name(F0), "ft0");
    }

    // ---- CodeGen trait tests ----

    #[test]
    fn test_target_arch() {
        let codegen = Riscv64CodeGen::new();
        assert_eq!(codegen.target_arch(), Architecture::Riscv64);
    }

    #[test]
    fn test_new_creates_valid_instance() {
        let _codegen = Riscv64CodeGen::new();
        // If this compiles and runs, the constructor works
    }

    #[test]
    fn test_generate_empty_module() {
        let codegen = Riscv64CodeGen::new();
        let module = Module::new("test".to_string());
        let target = TargetConfig::riscv64();
        let result = codegen.generate(&module, &target);
        assert!(
            result.is_ok(),
            "Code generation should succeed for empty module"
        );
        let obj = result.unwrap();
        assert_eq!(obj.target_arch, Architecture::Riscv64);
    }

    // ---- Frame size computation tests ----

    #[test]
    fn test_frame_size_no_callee_saved() {
        let alloc = AllocationResult {
            intervals: Vec::new(),
            num_spill_slots: 0,
            used_callee_saved: Vec::new(),
        };
        // Base frame: ra(8) + s0(8) = 16, aligned to 16
        let size = compute_frame_size(&alloc);
        assert_eq!(size, 16);
    }

    #[test]
    fn test_frame_size_with_spill_slots() {
        let alloc = AllocationResult {
            intervals: Vec::new(),
            num_spill_slots: 3,
            used_callee_saved: Vec::new(),
        };
        // 16 (base) + 24 (3 spill slots × 8) = 40, aligned to 48
        let size = compute_frame_size(&alloc);
        assert_eq!(size, 48);
    }

    #[test]
    fn test_frame_size_alignment() {
        let alloc = AllocationResult {
            intervals: Vec::new(),
            num_spill_slots: 1,
            used_callee_saved: vec![X18], // s2
        };
        // 16 (base) + 8 (s2) + 8 (1 spill) = 32, aligned to 32
        let size = compute_frame_size(&alloc);
        assert_eq!(size, 32);
    }
}
