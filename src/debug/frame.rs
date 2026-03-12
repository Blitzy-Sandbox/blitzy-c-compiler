// .debug_frame Section — Call Frame Information (CFI) for DWARF v4
//
// This module generates the `.debug_frame` ELF section content, providing
// Call Frame Information (CFI) that describes how to unwind the call stack
// at any given program counter value. The CFI data consists of:
//
// - **Common Information Entry (CIE)**: Defines the default unwinding rules
//   for an architecture, including the initial CFA definition, return address
//   register, and alignment factors.
//
// - **Frame Description Entry (FDE)**: Describes per-function register
//   save/restore rules relative to the CIE's initial state.
//
// Together, CIEs and FDEs enable debuggers (GDB, LLDB) and runtime unwinders
// to produce accurate stack traces at any instruction within a function.
//
// # Architecture Support
//
// All four target architectures are supported:
// - **x86-64**: System V AMD64 ABI, 16 GP registers + return address (RIP)
// - **i686**: cdecl ABI, 8 GP registers + return address (EIP)
// - **AArch64**: AAPCS64, 31 GP registers + SP, return in x30 (LR)
// - **RISC-V 64**: LP64D ABI, 32 GP registers + 32 FP, return in x1 (RA)
//
// # DWARF v4 Compliance
//
// All CFI encoding conforms to the DWARF Debugging Information Format
// Version 4 specification, using 32-bit DWARF format throughout
// (CIE_id = 0xFFFFFFFF).
//
// # Constraints
// - Zero external dependencies: only `std` and internal crate imports.
// - No `unsafe` code: pure data structure generation and byte encoding.

use crate::codegen::Architecture;
use crate::debug::dwarf;

// ===========================================================================
// DWARF CFA Instruction Opcodes
// ===========================================================================
//
// High-2-bit opcodes encode the opcode in the upper 2 bits of the byte,
// with the low 6 bits used as an operand (register number or location delta).

/// Advance location counter by delta encoded in low 6 bits.
/// The delta is factored by `code_alignment_factor`.
const DW_CFA_ADVANCE_LOC: u8 = 0x40;

/// Register saved at CFA + factored offset. Low 6 bits encode the register;
/// the factored offset follows as ULEB128.
const DW_CFA_OFFSET: u8 = 0x80;

/// Restore register to initial rule from CIE. Low 6 bits encode the register.
const DW_CFA_RESTORE: u8 = 0xc0;

// ---------------------------------------------------------------------------
// Zero-operand opcodes
// ---------------------------------------------------------------------------

/// No operation (used for alignment padding).
const DW_CFA_NOP: u8 = 0x00;

/// Push current register rule set onto an implicit stack.
const DW_CFA_REMEMBER_STATE: u8 = 0x0a;

/// Pop register rule set from the implicit stack.
const DW_CFA_RESTORE_STATE: u8 = 0x0b;

// ---------------------------------------------------------------------------
// One/two-operand opcodes
// ---------------------------------------------------------------------------

/// Set location to an absolute address (target-width operand).
#[allow(dead_code)]
const DW_CFA_SET_LOC: u8 = 0x01;

/// Advance location by a 1-byte unsigned delta.
const DW_CFA_ADVANCE_LOC1: u8 = 0x02;

/// Advance location by a 2-byte unsigned delta (little-endian).
const DW_CFA_ADVANCE_LOC2: u8 = 0x03;

/// Advance location by a 4-byte unsigned delta (little-endian).
const DW_CFA_ADVANCE_LOC4: u8 = 0x04;

/// Like `DW_CFA_offset` but the register is encoded as ULEB128 (for regs >= 64).
const DW_CFA_OFFSET_EXTENDED: u8 = 0x05;

/// Like `DW_CFA_restore` but the register is encoded as ULEB128.
const DW_CFA_RESTORE_EXTENDED: u8 = 0x06;

/// Register has no recoverable value.
#[allow(dead_code)]
const DW_CFA_UNDEFINED: u8 = 0x07;

/// Register retains its value (unchanged across calls).
const DW_CFA_SAME_VALUE: u8 = 0x08;

/// Register's value is currently held in another register.
/// Operands: ULEB128(register), ULEB128(other_register).
#[allow(dead_code)]
const DW_CFA_REGISTER: u8 = 0x09;

/// Define the Canonical Frame Address as `register + offset`.
/// Operands: ULEB128(register), ULEB128(offset).
const DW_CFA_DEF_CFA: u8 = 0x0c;

/// Change the CFA register, keeping the current offset.
/// Operand: ULEB128(register).
const DW_CFA_DEF_CFA_REGISTER: u8 = 0x0d;

/// Change the CFA offset, keeping the current register.
/// Operand: ULEB128(offset).
const DW_CFA_DEF_CFA_OFFSET: u8 = 0x0e;

/// CFA is computed by a DWARF expression.
#[allow(dead_code)]
const DW_CFA_DEF_CFA_EXPRESSION: u8 = 0x0f;

/// Register saved at location described by a DWARF expression.
#[allow(dead_code)]
const DW_CFA_EXPRESSION: u8 = 0x10;

/// Signed variant of `DW_CFA_def_cfa_offset` (SLEB128 operand).
#[allow(dead_code)]
const DW_CFA_DEF_CFA_OFFSET_SF: u8 = 0x13;

/// Register value *is* CFA + factored offset (not a pointer to saved value).
#[allow(dead_code)]
const DW_CFA_VAL_OFFSET: u8 = 0x14;

// ===========================================================================
// Architecture-Specific DWARF Register Numbers
// ===========================================================================

/// DWARF register numbers for x86-64 (System V AMD64 ABI).
///
/// Register numbering follows the System V AMD64 ABI DWARF register mapping:
/// - GP registers: 0..15 (RAX, RDX, RCX, RBX, RSI, RDI, RBP, RSP, R8..R15)
/// - Return address (RIP): 16
/// - SSE registers: 17..32 (XMM0..XMM15)
pub mod x86_64_regs {
    /// RAX — accumulator, return value register.
    pub const RAX: u16 = 0;
    /// RDX — data register, second return value.
    pub const RDX: u16 = 1;
    /// RCX — counter register, 4th integer argument.
    pub const RCX: u16 = 2;
    /// RBX — callee-saved base register.
    pub const RBX: u16 = 3;
    /// RSI — 2nd integer argument.
    pub const RSI: u16 = 4;
    /// RDI — 1st integer argument.
    pub const RDI: u16 = 5;
    /// RBP — frame pointer (callee-saved).
    pub const RBP: u16 = 6;
    /// RSP — stack pointer.
    pub const RSP: u16 = 7;
    /// R8 — 5th integer argument.
    pub const R8: u16 = 8;
    /// R9 — 6th integer argument.
    pub const R9: u16 = 9;
    /// R10 — temporary register.
    pub const R10: u16 = 10;
    /// R11 — temporary register.
    pub const R11: u16 = 11;
    /// R12 — callee-saved.
    pub const R12: u16 = 12;
    /// R13 — callee-saved.
    pub const R13: u16 = 13;
    /// R14 — callee-saved.
    pub const R14: u16 = 14;
    /// R15 — callee-saved.
    pub const R15: u16 = 15;
    /// Return address register (RIP).
    pub const RA: u16 = 16;
    /// XMM0 — first SSE/floating-point register.
    pub const XMM0: u16 = 17;
    /// XMM1
    pub const XMM1: u16 = 18;
    /// XMM2
    pub const XMM2: u16 = 19;
    /// XMM3
    pub const XMM3: u16 = 20;
    /// XMM4
    pub const XMM4: u16 = 21;
    /// XMM5
    pub const XMM5: u16 = 22;
    /// XMM6
    pub const XMM6: u16 = 23;
    /// XMM7
    pub const XMM7: u16 = 24;
    /// XMM8
    pub const XMM8: u16 = 25;
    /// XMM9
    pub const XMM9: u16 = 26;
    /// XMM10
    pub const XMM10: u16 = 27;
    /// XMM11
    pub const XMM11: u16 = 28;
    /// XMM12
    pub const XMM12: u16 = 29;
    /// XMM13
    pub const XMM13: u16 = 30;
    /// XMM14
    pub const XMM14: u16 = 31;
    /// XMM15
    pub const XMM15: u16 = 32;
}

/// DWARF register numbers for i686 (System V i386 cdecl ABI).
///
/// Register numbering follows the i386 ABI DWARF register mapping:
/// - GP registers: 0..7 (EAX, ECX, EDX, EBX, ESP, EBP, ESI, EDI)
/// - Return address (EIP): 8
pub mod i686_regs {
    /// EAX — accumulator, return value register.
    pub const EAX: u16 = 0;
    /// ECX — counter register.
    pub const ECX: u16 = 1;
    /// EDX — data register.
    pub const EDX: u16 = 2;
    /// EBX — callee-saved base register.
    pub const EBX: u16 = 3;
    /// ESP — stack pointer.
    pub const ESP: u16 = 4;
    /// EBP — frame pointer (callee-saved).
    pub const EBP: u16 = 5;
    /// ESI — callee-saved source index.
    pub const ESI: u16 = 6;
    /// EDI — callee-saved destination index.
    pub const EDI: u16 = 7;
    /// Return address register (EIP).
    pub const RA: u16 = 8;
}

/// DWARF register numbers for AArch64 (AAPCS64 ABI).
///
/// Register numbering:
/// - GP registers x0..x30: 0..30
/// - Stack pointer (SP): 31
/// - SIMD/FP registers v0..v31: 64..95
/// - Return address is in x30 (LR — link register)
pub mod aarch64_regs {
    /// x0 — first integer argument / return value.
    pub const X0: u16 = 0;
    /// x1
    pub const X1: u16 = 1;
    /// x2
    pub const X2: u16 = 2;
    /// x3
    pub const X3: u16 = 3;
    /// x4
    pub const X4: u16 = 4;
    /// x5
    pub const X5: u16 = 5;
    /// x6
    pub const X6: u16 = 6;
    /// x7
    pub const X7: u16 = 7;
    /// x8 — indirect result location register.
    pub const X8: u16 = 8;
    /// x9
    pub const X9: u16 = 9;
    /// x10
    pub const X10: u16 = 10;
    /// x11
    pub const X11: u16 = 11;
    /// x12
    pub const X12: u16 = 12;
    /// x13
    pub const X13: u16 = 13;
    /// x14
    pub const X14: u16 = 14;
    /// x15
    pub const X15: u16 = 15;
    /// x16 — IP0 (intra-procedure scratch).
    pub const X16: u16 = 16;
    /// x17 — IP1 (intra-procedure scratch).
    pub const X17: u16 = 17;
    /// x18 — platform register.
    pub const X18: u16 = 18;
    /// x19 — callee-saved.
    pub const X19: u16 = 19;
    /// x20 — callee-saved.
    pub const X20: u16 = 20;
    /// x21 — callee-saved.
    pub const X21: u16 = 21;
    /// x22 — callee-saved.
    pub const X22: u16 = 22;
    /// x23 — callee-saved.
    pub const X23: u16 = 23;
    /// x24 — callee-saved.
    pub const X24: u16 = 24;
    /// x25 — callee-saved.
    pub const X25: u16 = 25;
    /// x26 — callee-saved.
    pub const X26: u16 = 26;
    /// x27 — callee-saved.
    pub const X27: u16 = 27;
    /// x28 — callee-saved.
    pub const X28: u16 = 28;
    /// x29 — frame pointer (FP), callee-saved.
    pub const X29: u16 = 29;
    /// x30 — link register (LR), holds the return address.
    pub const X30: u16 = 30;
    /// SP — stack pointer.
    pub const SP: u16 = 31;
    /// v0 — first SIMD/FP register.
    pub const V0: u16 = 64;
    /// v1
    pub const V1: u16 = 65;
    /// v2
    pub const V2: u16 = 66;
    /// v3
    pub const V3: u16 = 67;
    /// v4
    pub const V4: u16 = 68;
    /// v5
    pub const V5: u16 = 69;
    /// v6
    pub const V6: u16 = 70;
    /// v7
    pub const V7: u16 = 71;
    /// v8 — callee-saved (low 64 bits).
    pub const V8: u16 = 72;
    /// v9
    pub const V9: u16 = 73;
    /// v10
    pub const V10: u16 = 74;
    /// v11
    pub const V11: u16 = 75;
    /// v12
    pub const V12: u16 = 76;
    /// v13
    pub const V13: u16 = 77;
    /// v14
    pub const V14: u16 = 78;
    /// v15
    pub const V15: u16 = 79;
    /// v16
    pub const V16: u16 = 80;
    /// v17
    pub const V17: u16 = 81;
    /// v18
    pub const V18: u16 = 82;
    /// v19
    pub const V19: u16 = 83;
    /// v20
    pub const V20: u16 = 84;
    /// v21
    pub const V21: u16 = 85;
    /// v22
    pub const V22: u16 = 86;
    /// v23
    pub const V23: u16 = 87;
    /// v24
    pub const V24: u16 = 88;
    /// v25
    pub const V25: u16 = 89;
    /// v26
    pub const V26: u16 = 90;
    /// v27
    pub const V27: u16 = 91;
    /// v28
    pub const V28: u16 = 92;
    /// v29
    pub const V29: u16 = 93;
    /// v30
    pub const V30: u16 = 94;
    /// v31
    pub const V31: u16 = 95;
}

/// DWARF register numbers for RISC-V 64 (LP64D ABI).
///
/// Register numbering:
/// - Integer registers x0..x31: 0..31
/// - Floating-point registers f0..f31: 32..63
/// - Return address is in x1 (RA)
pub mod riscv64_regs {
    /// x0 — hardwired zero.
    pub const X0: u16 = 0;
    /// x1 — return address (RA).
    pub const X1: u16 = 1;
    /// x2 — stack pointer (SP).
    pub const X2: u16 = 2;
    /// x3 — global pointer (GP).
    pub const X3: u16 = 3;
    /// x4 — thread pointer (TP).
    pub const X4: u16 = 4;
    /// x5 — temporary (t0).
    pub const X5: u16 = 5;
    /// x6 — temporary (t1).
    pub const X6: u16 = 6;
    /// x7 — temporary (t2).
    pub const X7: u16 = 7;
    /// x8 — saved register / frame pointer (s0/fp).
    pub const X8: u16 = 8;
    /// x9 — saved register (s1).
    pub const X9: u16 = 9;
    /// x10 — argument / return (a0).
    pub const X10: u16 = 10;
    /// x11 — argument / return (a1).
    pub const X11: u16 = 11;
    /// x12 — argument (a2).
    pub const X12: u16 = 12;
    /// x13 — argument (a3).
    pub const X13: u16 = 13;
    /// x14 — argument (a4).
    pub const X14: u16 = 14;
    /// x15 — argument (a5).
    pub const X15: u16 = 15;
    /// x16 — argument (a6).
    pub const X16: u16 = 16;
    /// x17 — argument (a7).
    pub const X17: u16 = 17;
    /// x18 — saved register (s2).
    pub const X18: u16 = 18;
    /// x19 — saved register (s3).
    pub const X19: u16 = 19;
    /// x20 — saved register (s4).
    pub const X20: u16 = 20;
    /// x21 — saved register (s5).
    pub const X21: u16 = 21;
    /// x22 — saved register (s6).
    pub const X22: u16 = 22;
    /// x23 — saved register (s7).
    pub const X23: u16 = 23;
    /// x24 — saved register (s8).
    pub const X24: u16 = 24;
    /// x25 — saved register (s9).
    pub const X25: u16 = 25;
    /// x26 — saved register (s10).
    pub const X26: u16 = 26;
    /// x27 — saved register (s11).
    pub const X27: u16 = 27;
    /// x28 — temporary (t3).
    pub const X28: u16 = 28;
    /// x29 — temporary (t4).
    pub const X29: u16 = 29;
    /// x30 — temporary (t5).
    pub const X30: u16 = 30;
    /// x31 — temporary (t6).
    pub const X31: u16 = 31;
    /// f0 — FP temporary (ft0).
    pub const F0: u16 = 32;
    /// f1 — FP temporary (ft1).
    pub const F1: u16 = 33;
    /// f2 — FP temporary (ft2).
    pub const F2: u16 = 34;
    /// f3 — FP temporary (ft3).
    pub const F3: u16 = 35;
    /// f4 — FP temporary (ft4).
    pub const F4: u16 = 36;
    /// f5 — FP temporary (ft5).
    pub const F5: u16 = 37;
    /// f6 — FP temporary (ft6).
    pub const F6: u16 = 38;
    /// f7 — FP temporary (ft7).
    pub const F7: u16 = 39;
    /// f8 — FP saved (fs0).
    pub const F8: u16 = 40;
    /// f9 — FP saved (fs1).
    pub const F9: u16 = 41;
    /// f10 — FP argument / return (fa0).
    pub const F10: u16 = 42;
    /// f11 — FP argument / return (fa1).
    pub const F11: u16 = 43;
    /// f12 — FP argument (fa2).
    pub const F12: u16 = 44;
    /// f13 — FP argument (fa3).
    pub const F13: u16 = 45;
    /// f14 — FP argument (fa4).
    pub const F14: u16 = 46;
    /// f15 — FP argument (fa5).
    pub const F15: u16 = 47;
    /// f16 — FP argument (fa6).
    pub const F16: u16 = 48;
    /// f17 — FP argument (fa7).
    pub const F17: u16 = 49;
    /// f18 — FP saved (fs2).
    pub const F18: u16 = 50;
    /// f19 — FP saved (fs3).
    pub const F19: u16 = 51;
    /// f20 — FP saved (fs4).
    pub const F20: u16 = 52;
    /// f21 — FP saved (fs5).
    pub const F21: u16 = 53;
    /// f22 — FP saved (fs6).
    pub const F22: u16 = 54;
    /// f23 — FP saved (fs7).
    pub const F23: u16 = 55;
    /// f24 — FP saved (fs8).
    pub const F24: u16 = 56;
    /// f25 — FP saved (fs9).
    pub const F25: u16 = 57;
    /// f26 — FP saved (fs10).
    pub const F26: u16 = 58;
    /// f27 — FP saved (fs11).
    pub const F27: u16 = 59;
    /// f28 — FP temporary (ft8).
    pub const F28: u16 = 60;
    /// f29 — FP temporary (ft9).
    pub const F29: u16 = 61;
    /// f30 — FP temporary (ft10).
    pub const F30: u16 = 62;
    /// f31 — FP temporary (ft11).
    pub const F31: u16 = 63;
}

// ===========================================================================
// CIE Identifier for .debug_frame (32-bit DWARF format)
// ===========================================================================

/// The CIE identifier value for `.debug_frame` in 32-bit DWARF format.
/// This value (0xFFFFFFFF) distinguishes a CIE from an FDE when reading
/// the section. In `.eh_frame` the value would be 0 instead.
const CIE_ID_32: u32 = 0xFFFF_FFFF;

// ===========================================================================
// Core Data Structures
// ===========================================================================

/// Common Information Entry (CIE) — defines the default unwinding rules
/// for an architecture or compilation context.
///
/// A CIE is shared by multiple FDEs. It specifies:
/// - The DWARF version (4)
/// - Alignment factors for code and data
/// - The return address register number
/// - Initial CFI instructions that establish the default register rules
///   at function entry (before any prologue executes)
#[derive(Debug, Clone)]
pub struct CommonInformationEntry {
    /// CIE version — always 4 for DWARF v4.
    pub version: u8,
    /// Augmentation string (empty for basic `.debug_frame`).
    pub augmentation: String,
    /// Target address size in bytes (4 for i686, 8 for 64-bit targets).
    pub address_size: u8,
    /// Segment selector size (always 0 on Linux).
    pub segment_size: u8,
    /// Minimum instruction length; used to factor `advance_loc` deltas.
    /// x86: 1, AArch64: 4, RISC-V: 2.
    pub code_alignment_factor: u64,
    /// Data alignment factor; negative because the stack grows downward.
    /// x86-64: -8, i686: -4, AArch64: -8, RISC-V 64: -8.
    pub data_alignment_factor: i64,
    /// DWARF register number of the return address register.
    pub return_address_register: u64,
    /// Initial CFI instructions defining the CFA and register rules at
    /// function entry.
    pub initial_instructions: Vec<u8>,
}

/// Frame Description Entry (FDE) — describes per-function register
/// save/restore rules for stack unwinding.
///
/// Each FDE references a CIE whose initial rules it extends, and covers
/// a contiguous range of machine code addresses.
#[derive(Debug, Clone)]
pub struct FrameDescriptionEntry {
    /// Offset of the associated CIE within the `.debug_frame` section.
    pub cie_offset: u32,
    /// Start address of the function (or code range).
    pub initial_location: u64,
    /// Size in bytes of the function's address range.
    pub address_range: u64,
    /// CFI instructions describing how the register state changes
    /// throughout the function body.
    pub instructions: Vec<u8>,
}

/// Information about a function's stack frame layout, provided by the
/// code generator to drive FDE construction.
///
/// This is the interface between the codegen backends and the debug
/// frame generator: each backend populates a `FunctionFrameInfo` for
/// every generated function, and the frame module translates it into
/// DWARF CFI instructions.
#[derive(Debug, Clone)]
pub struct FunctionFrameInfo {
    /// Virtual address of the function's first instruction.
    pub start_address: u64,
    /// Total size of the function's machine code in bytes.
    pub size: u64,
    /// Size of the stack frame in bytes (the amount subtracted from SP).
    pub frame_size: u64,
    /// Whether the function uses a frame pointer (RBP/EBP/x29/s0).
    pub uses_frame_pointer: bool,
    /// Callee-saved registers that are saved in the prologue.
    /// Each tuple is (DWARF register number, byte offset from CFA where
    /// the register is saved). Offsets are typically negative (below CFA).
    pub callee_saved_regs: Vec<(u16, i64)>,
    /// Size of the function prologue in bytes (offset from start to end
    /// of prologue).
    pub prologue_size: u64,
    /// Byte offset from function start to the beginning of the epilogue.
    pub epilogue_offset: u64,
}

// ===========================================================================
// CFI Instruction Builder
// ===========================================================================

/// Builder for constructing sequences of DWARF CFI instructions.
///
/// The builder maintains an internal byte buffer and the architecture's
/// alignment factors so that it can automatically factor deltas and
/// choose the most compact encoding for `advance_loc` instructions.
///
/// # Usage
///
/// ```ignore
/// let mut builder = CfiInstructionBuilder::new(1, -8);
/// builder.def_cfa(7, 8);         // CFA = RSP + 8
/// builder.offset(16, 1);         // RA at CFA - 8  (1 * |-8|)
/// builder.advance_loc(1);        // Advance 1 byte
/// builder.def_cfa_offset(16);    // CFA offset now 16
/// let instructions = builder.build();
/// ```
pub struct CfiInstructionBuilder {
    /// Accumulated CFI instruction bytes.
    instructions: Vec<u8>,
    /// Current code offset tracked for informational purposes.
    current_offset: u64,
    /// Minimum instruction length factor for `advance_loc` division.
    code_alignment_factor: u64,
    /// Data alignment factor (negative) for offset factoring.
    data_alignment_factor: i64,
}

impl CfiInstructionBuilder {
    /// Create a new CFI instruction builder.
    ///
    /// # Arguments
    ///
    /// * `code_alignment_factor` — Minimum instruction length for the
    ///   target architecture (1 for x86, 4 for AArch64, 2 for RISC-V).
    /// * `data_alignment_factor` — Signed data alignment factor (e.g. -8
    ///   for x86-64, -4 for i686).
    pub fn new(code_alignment_factor: u64, data_alignment_factor: i64) -> Self {
        Self {
            instructions: Vec::new(),
            current_offset: 0,
            code_alignment_factor,
            data_alignment_factor,
        }
    }

    /// Emit `DW_CFA_def_cfa`: set the Canonical Frame Address to
    /// `register + offset`.
    pub fn def_cfa(&mut self, register: u16, offset: u64) {
        self.instructions.push(DW_CFA_DEF_CFA);
        dwarf::encode_uleb128_to(&mut self.instructions, register as u64);
        dwarf::encode_uleb128_to(&mut self.instructions, offset);
    }

    /// Emit `DW_CFA_def_cfa_register`: change the CFA register while
    /// keeping the current CFA offset unchanged.
    pub fn def_cfa_register(&mut self, register: u16) {
        self.instructions.push(DW_CFA_DEF_CFA_REGISTER);
        dwarf::encode_uleb128_to(&mut self.instructions, register as u64);
    }

    /// Emit `DW_CFA_def_cfa_offset`: change the CFA offset while
    /// keeping the current CFA register unchanged.
    pub fn def_cfa_offset(&mut self, offset: u64) {
        self.instructions.push(DW_CFA_DEF_CFA_OFFSET);
        dwarf::encode_uleb128_to(&mut self.instructions, offset);
    }

    /// Emit an offset instruction recording that `register` is saved at
    /// `CFA - (factored_offset * |data_alignment_factor|)`.
    ///
    /// For registers < 64 the compact `DW_CFA_offset` encoding is used
    /// (register packed into the opcode byte). For registers >= 64 the
    /// extended `DW_CFA_offset_extended` encoding is used.
    pub fn offset(&mut self, register: u16, factored_offset: u64) {
        if register < 64 {
            // Compact encoding: register in low 6 bits of opcode byte.
            self.instructions.push(DW_CFA_OFFSET | (register as u8));
            dwarf::encode_uleb128_to(&mut self.instructions, factored_offset);
        } else {
            // Extended encoding: register as separate ULEB128 operand.
            self.instructions.push(DW_CFA_OFFSET_EXTENDED);
            dwarf::encode_uleb128_to(&mut self.instructions, register as u64);
            dwarf::encode_uleb128_to(&mut self.instructions, factored_offset);
        }
    }

    /// Emit `DW_CFA_same_value`: indicate that `register` has not been
    /// modified and retains its entry value.
    pub fn same_value(&mut self, register: u16) {
        self.instructions.push(DW_CFA_SAME_VALUE);
        dwarf::encode_uleb128_to(&mut self.instructions, register as u64);
    }

    /// Emit an `advance_loc` instruction advancing the location counter by
    /// `delta` bytes. The most compact encoding is chosen automatically:
    ///
    /// 1. If `delta / code_alignment_factor` fits in 6 bits -> `DW_CFA_advance_loc` (1 byte)
    /// 2. If it fits in u8 -> `DW_CFA_advance_loc1` (2 bytes)
    /// 3. If it fits in u16 -> `DW_CFA_advance_loc2` (3 bytes)
    /// 4. Otherwise -> `DW_CFA_advance_loc4` (5 bytes)
    pub fn advance_loc(&mut self, delta: u64) {
        // Guard against division by zero: treat factor of 0 as 1.
        let caf = if self.code_alignment_factor > 0 {
            self.code_alignment_factor
        } else {
            1
        };
        let factored = delta / caf;
        self.current_offset += delta;

        if factored < 0x40 {
            // 1-byte compact encoding: opcode high bits + 6-bit delta.
            self.instructions
                .push(DW_CFA_ADVANCE_LOC | (factored as u8));
        } else if factored <= 0xFF {
            // 2-byte encoding: opcode + 1-byte delta.
            self.instructions.push(DW_CFA_ADVANCE_LOC1);
            dwarf::write_u8(&mut self.instructions, factored as u8);
        } else if factored <= 0xFFFF {
            // 3-byte encoding: opcode + 2-byte LE delta.
            self.instructions.push(DW_CFA_ADVANCE_LOC2);
            dwarf::write_u16_le(&mut self.instructions, factored as u16);
        } else {
            // 5-byte encoding: opcode + 4-byte LE delta.
            self.instructions.push(DW_CFA_ADVANCE_LOC4);
            dwarf::write_u32_le(&mut self.instructions, factored as u32);
        }
    }

    /// Emit `DW_CFA_remember_state`: push the current register rule set
    /// onto an implicit stack.
    pub fn remember_state(&mut self) {
        self.instructions.push(DW_CFA_REMEMBER_STATE);
    }

    /// Emit `DW_CFA_restore_state`: pop the register rule set from the
    /// implicit stack, restoring all register rules to their pushed state.
    pub fn restore_state(&mut self) {
        self.instructions.push(DW_CFA_RESTORE_STATE);
    }

    /// Emit `DW_CFA_nop`: no-operation padding byte.
    pub fn nop(&mut self) {
        self.instructions.push(DW_CFA_NOP);
    }

    /// Return a copy of the accumulated CFI instruction bytes.
    pub fn build(&self) -> Vec<u8> {
        self.instructions.clone()
    }
}

// ===========================================================================
// FrameInfoEmitter — high-level coordinator
// ===========================================================================

/// High-level emitter that generates the complete `.debug_frame` section
/// for a compilation unit.
///
/// Typical workflow:
/// 1. Create an emitter for the target architecture.
/// 2. Call `emit()` with the function frame info list.
/// 3. The emitter builds a CIE, creates an FDE per function, serializes
///    everything, and returns the complete section bytes.
pub struct FrameInfoEmitter {
    /// Target architecture for register numbering and alignment factors.
    arch: Architecture,
    /// Target address size (4 for i686, 8 for 64-bit targets).
    address_size: u8,
}

impl FrameInfoEmitter {
    /// Create a new emitter for the given architecture and address size.
    pub fn new(arch: Architecture, address_size: u8) -> Self {
        Self { arch, address_size }
    }

    /// Generate the complete `.debug_frame` section bytes from a list of
    /// function frame descriptions.
    ///
    /// Returns the serialized section bytes containing one CIE followed by
    /// one FDE per function.
    pub fn emit(&self, functions: &[FunctionFrameInfo]) -> Vec<u8> {
        let cie = build_cie_for_arch(self.arch, self.address_size);
        let daf = cie.data_alignment_factor;
        let fdes: Vec<FrameDescriptionEntry> = functions
            .iter()
            .map(|f| build_function_fde(f, self.arch, daf))
            .collect();
        serialize_debug_frame(&cie, &fdes, self.address_size)
    }
}

// ===========================================================================
// CIE Construction — per-architecture
// ===========================================================================

/// Build a Common Information Entry (CIE) appropriate for the given
/// target architecture.
///
/// The CIE defines:
/// - `code_alignment_factor` — minimum instruction length
/// - `data_alignment_factor` — signed, matches stack slot size
/// - `return_address_register` — DWARF register number of the RA
/// - `initial_instructions` — default CFA definition at function entry
///
/// # Architecture Summary
///
/// | Arch       | code_align | data_align | RA reg    | CFA default |
/// |------------|-----------|-----------|-----------|-------------|
/// | x86-64     | 1         | -8        | 16 (RIP)  | RSP+8       |
/// | i686       | 1         | -4        | 8  (EIP)  | ESP+4       |
/// | AArch64    | 4         | -8        | 30 (LR)   | SP+0        |
/// | RISC-V 64  | 2         | -8        | 1  (RA)   | SP+0        |
pub fn build_cie_for_arch(arch: Architecture, address_size: u8) -> CommonInformationEntry {
    let (caf, daf, ra_reg): (u64, i64, u64) = match arch {
        Architecture::X86_64 => (1, -8, x86_64_regs::RA as u64),
        Architecture::I686 => (1, -4, i686_regs::RA as u64),
        Architecture::Aarch64 => (4, -8, aarch64_regs::X30 as u64),
        Architecture::Riscv64 => (2, -8, riscv64_regs::X1 as u64),
    };

    // Build initial instructions using the CfiInstructionBuilder so that
    // all encoding goes through a single, tested code path.
    let mut builder = CfiInstructionBuilder::new(caf, daf);

    match arch {
        Architecture::X86_64 => {
            // At function entry the CALL instruction has pushed the return
            // address, so CFA = RSP + 8. The return address lives at CFA - 8.
            builder.def_cfa(x86_64_regs::RSP, 8);
            // factored_offset = 1: offset from CFA = 1 * |-8| = 8 bytes below CFA.
            builder.offset(x86_64_regs::RA, 1);
        }
        Architecture::I686 => {
            // CFA = ESP + 4 (return address on stack).
            builder.def_cfa(i686_regs::ESP, 4);
            // Return address at CFA - 4 (1 * |-4|).
            builder.offset(i686_regs::RA, 1);
        }
        Architecture::Aarch64 => {
            // CFA = SP + 0 at function entry (before any stack adjustment).
            builder.def_cfa(aarch64_regs::SP, 0);
            // LR (x30) is still in its register — not yet pushed.
            builder.same_value(aarch64_regs::X30);
        }
        Architecture::Riscv64 => {
            // CFA = SP + 0 at function entry.
            builder.def_cfa(riscv64_regs::X2, 0);
            // RA (x1) is still in its register — not yet pushed.
            builder.same_value(riscv64_regs::X1);
        }
    }

    CommonInformationEntry {
        version: dwarf::DWARF_VERSION as u8,
        augmentation: String::new(),
        address_size,
        segment_size: 0,
        code_alignment_factor: caf,
        data_alignment_factor: daf,
        return_address_register: ra_reg,
        initial_instructions: builder.build(),
    }
}

// ===========================================================================
// FDE Construction
// ===========================================================================

/// Build an FDE for a function from its frame layout information.
///
/// This is the convenience variant that derives the data alignment factor
/// from the target architecture, then delegates to [`build_function_fde`].
pub fn build_fde(func_info: &FunctionFrameInfo, arch: Architecture) -> FrameDescriptionEntry {
    let daf = match arch {
        Architecture::X86_64 => -8i64,
        Architecture::I686 => -4i64,
        Architecture::Aarch64 => -8i64,
        Architecture::Riscv64 => -8i64,
    };
    build_function_fde(func_info, arch, daf)
}

/// Build an FDE from detailed function frame information and the CIE's
/// data alignment factor.
///
/// The generated CFI instruction sequence describes:
/// 1. **Prologue** — CFA adjustments and callee-saved register saves.
/// 2. **Body** — CFA remains stable (tracked via frame pointer or SP).
/// 3. **Epilogue** — register restores and CFA restoration.
///
/// # Frame-Pointer Functions (x86-64 example)
///
/// ```text
/// push %rbp           -> advance_loc, def_cfa_offset(16), offset(RBP, 2)
/// mov  %rsp, %rbp     -> advance_loc, def_cfa_register(RBP)
/// sub  $N, %rsp       -> (no CFI change — CFA tracks RBP)
/// ; body
/// leave               -> advance_loc, def_cfa(RSP, 8), restore(RBP)
/// ret
/// ```
///
/// # SP-Tracking Functions
///
/// ```text
/// sub $N, %rsp        -> advance_loc, def_cfa_offset(8 + N)
/// ; body
/// add $N, %rsp        -> advance_loc, def_cfa_offset(8)
/// ret
/// ```
pub fn build_function_fde(
    frame_info: &FunctionFrameInfo,
    arch: Architecture,
    data_alignment_factor: i64,
) -> FrameDescriptionEntry {
    let caf = match arch {
        Architecture::X86_64 => 1u64,
        Architecture::I686 => 1u64,
        Architecture::Aarch64 => 4u64,
        Architecture::Riscv64 => 2u64,
    };

    let mut builder = CfiInstructionBuilder::new(caf, data_alignment_factor);

    // Determine architecture-specific register numbers for SP and FP.
    let (sp_reg, fp_reg) = match arch {
        Architecture::X86_64 => (x86_64_regs::RSP, x86_64_regs::RBP),
        Architecture::I686 => (i686_regs::ESP, i686_regs::EBP),
        Architecture::Aarch64 => (aarch64_regs::SP, aarch64_regs::X29),
        Architecture::Riscv64 => (riscv64_regs::X2, riscv64_regs::X8),
    };

    // The absolute value of data_alignment_factor, used for offset factoring.
    let daf_abs = data_alignment_factor.unsigned_abs();
    // Guard against zero to avoid division-by-zero if misconfigured.
    let daf_abs = if daf_abs == 0 { 1 } else { daf_abs };

    if frame_info.prologue_size > 0 {
        if frame_info.uses_frame_pointer {
            // === Frame-pointer based function ===
            let ptr_size = frame_info_ptr_size(arch);
            let initial_cfa_offset = match arch {
                Architecture::X86_64 => 8u64, // CIE: CFA = RSP + 8
                Architecture::I686 => 4u64,   // CIE: CFA = ESP + 4
                _ => 0u64,                    // AArch64/RISC-V: CFA = SP + 0
            };

            match arch {
                Architecture::X86_64 | Architecture::I686 => {
                    // push %rbp / push %ebp — both are 1 byte on x86.
                    let push_size = 1u64;
                    builder.advance_loc(push_size);
                    let new_cfa = initial_cfa_offset + ptr_size;
                    builder.def_cfa_offset(new_cfa);
                    // Frame pointer saved at CFA - new_cfa.
                    let fp_factored = new_cfa / daf_abs;
                    builder.offset(fp_reg, fp_factored);

                    // mov %rsp, %rbp (3 bytes on x86-64, 2 bytes on i686).
                    let mov_size = match arch {
                        Architecture::X86_64 => 3u64,
                        Architecture::I686 => 2u64,
                        _ => unreachable!(),
                    };
                    builder.advance_loc(mov_size);
                    builder.def_cfa_register(fp_reg);
                }
                Architecture::Aarch64 => {
                    // stp x29, x30, [sp, #-frame_size]!
                    builder.advance_loc(4); // one A64 instruction
                    if frame_info.frame_size > 0 {
                        builder.def_cfa_offset(frame_info.frame_size);
                    }
                    // x29 (FP) saved at CFA - frame_size.
                    if daf_abs > 0 && frame_info.frame_size > 0 {
                        let fp_fact = frame_info.frame_size / daf_abs;
                        builder.offset(aarch64_regs::X29, fp_fact);
                    }
                    // x30 (LR) saved just above FP (at frame_size - 8).
                    if frame_info.frame_size >= 8 {
                        let lr_fact = (frame_info.frame_size - 8) / daf_abs;
                        builder.offset(aarch64_regs::X30, lr_fact);
                    }
                    // mov x29, sp — CFA now tracks x29.
                    builder.advance_loc(4);
                    builder.def_cfa_register(aarch64_regs::X29);
                }
                Architecture::Riscv64 => {
                    // addi sp, sp, -N — allocate frame (4-byte instruction).
                    builder.advance_loc(4);
                    if frame_info.frame_size > 0 {
                        builder.def_cfa_offset(frame_info.frame_size);
                    }
                    // sd ra, N-8(sp) — save return address.
                    if frame_info.frame_size >= 8 {
                        builder.advance_loc(4);
                        let ra_fact = (frame_info.frame_size - 8) / daf_abs;
                        builder.offset(riscv64_regs::X1, ra_fact);
                    }
                    // sd s0, N-16(sp) — save frame pointer.
                    if frame_info.frame_size >= 16 {
                        builder.advance_loc(4);
                        let fp_fact = (frame_info.frame_size - 16) / daf_abs;
                        builder.offset(riscv64_regs::X8, fp_fact);
                    }
                    // addi s0, sp, N — establish frame pointer.
                    builder.advance_loc(4);
                    builder.def_cfa_register(riscv64_regs::X8);
                }
            }
        } else {
            // === Frame-pointer-less function (SP tracking only) ===
            let initial_cfa_offset = match arch {
                Architecture::X86_64 => 8u64,
                Architecture::I686 => 4u64,
                _ => 0u64,
            };

            if frame_info.frame_size > 0 {
                match arch {
                    Architecture::X86_64 | Architecture::I686 => {
                        // sub $N, %rsp — prologue_size encodes the instruction length.
                        builder.advance_loc(frame_info.prologue_size);
                        builder.def_cfa_offset(initial_cfa_offset + frame_info.frame_size);
                    }
                    Architecture::Aarch64 => {
                        builder.advance_loc(4); // single sub sp instruction
                        builder.def_cfa_offset(frame_info.frame_size);
                    }
                    Architecture::Riscv64 => {
                        builder.advance_loc(4); // addi sp, sp, -N
                        builder.def_cfa_offset(frame_info.frame_size);
                    }
                }
            }
        }

        // Record callee-saved register saves (applicable to both styles).
        // byte_offset is negative (saved below CFA); DWARF uses the
        // factored absolute value.
        for &(reg, byte_offset) in &frame_info.callee_saved_regs {
            let abs_offset = byte_offset.unsigned_abs();
            let factored = abs_offset / daf_abs;
            builder.offset(reg, factored);
        }
    }

    // Generate epilogue CFI if the function has an epilogue marker.
    if frame_info.epilogue_offset > 0 && frame_info.epilogue_offset < frame_info.size {
        // Advance the location counter from the current position to the
        // epilogue start.
        let current_loc = if frame_info.prologue_size > 0 {
            frame_info.prologue_size
        } else {
            0
        };
        let remaining = frame_info.epilogue_offset.saturating_sub(current_loc);
        if remaining > 0 {
            builder.advance_loc(remaining);
        }

        // Restore callee-saved registers in reverse order.
        for &(reg, _) in frame_info.callee_saved_regs.iter().rev() {
            if reg < 64 {
                builder.instructions.push(DW_CFA_RESTORE | (reg as u8));
            } else {
                builder.instructions.push(DW_CFA_RESTORE_EXTENDED);
                dwarf::encode_uleb128_to(&mut builder.instructions, reg as u64);
            }
        }

        // Restore CFA to the initial state established by the CIE.
        let initial_offset = match arch {
            Architecture::X86_64 => 8u64,
            Architecture::I686 => 4u64,
            Architecture::Aarch64 => 0u64,
            Architecture::Riscv64 => 0u64,
        };

        if frame_info.uses_frame_pointer {
            builder.def_cfa(sp_reg, initial_offset);
            // Restore the frame pointer register itself.
            if fp_reg < 64 {
                builder.instructions.push(DW_CFA_RESTORE | (fp_reg as u8));
            } else {
                builder.instructions.push(DW_CFA_RESTORE_EXTENDED);
                dwarf::encode_uleb128_to(&mut builder.instructions, fp_reg as u64);
            }
        } else {
            builder.def_cfa_offset(initial_offset);
        }
    }

    FrameDescriptionEntry {
        cie_offset: 0, // Patched during serialization.
        initial_location: frame_info.start_address,
        address_range: frame_info.size,
        instructions: builder.build(),
    }
}

/// Return pointer/slot size for the given architecture.
fn frame_info_ptr_size(arch: Architecture) -> u64 {
    match arch {
        Architecture::X86_64 | Architecture::Aarch64 | Architecture::Riscv64 => 8,
        Architecture::I686 => 4,
    }
}

// ===========================================================================
// Serialization
// ===========================================================================

/// Serialize a Common Information Entry into its DWARF binary representation.
///
/// Layout (32-bit DWARF format):
/// ```text
/// +---------------------------------------+
/// | length          : u32                |  Total CIE length (excl. this field)
/// | CIE_id          : u32 = 0xFFFFFFFF   |  Identifies this as a CIE
/// | version         : u8  = 4            |  DWARF version
/// | augmentation    : string (NUL-term)  |  Empty for .debug_frame
/// | address_size    : u8                 |  4 or 8
/// | segment_size    : u8                 |  0 (Linux)
/// | code_align_fac  : ULEB128            |
/// | data_align_fac  : SLEB128            |
/// | return_addr_reg : ULEB128            |
/// | initial_instr   : bytes              |
/// | padding         : DW_CFA_nop x N     |  Pad to address_size alignment
/// +---------------------------------------+
/// ```
pub fn serialize_cie(cie: &CommonInformationEntry) -> Vec<u8> {
    // Build the CIE content (everything after the length field).
    let mut content = Vec::new();

    // CIE_id: 0xFFFFFFFF for .debug_frame (32-bit DWARF format).
    dwarf::write_u32_le(&mut content, CIE_ID_32);

    // Version.
    dwarf::write_u8(&mut content, cie.version);

    // Augmentation string (null-terminated).
    content.extend_from_slice(cie.augmentation.as_bytes());
    content.push(0x00);

    // Address size (DWARF v4 addition).
    dwarf::write_u8(&mut content, cie.address_size);

    // Segment size (always 0 on Linux).
    dwarf::write_u8(&mut content, cie.segment_size);

    // Code alignment factor (ULEB128).
    dwarf::encode_uleb128_to(&mut content, cie.code_alignment_factor);

    // Data alignment factor (SLEB128).
    dwarf::encode_sleb128_to(&mut content, cie.data_alignment_factor);

    // Return address register (ULEB128).
    dwarf::encode_uleb128_to(&mut content, cie.return_address_register);

    // Initial instructions.
    content.extend_from_slice(&cie.initial_instructions);

    // Pad to address_size alignment with DW_CFA_nop.
    let align = cie.address_size as usize;
    if align > 0 {
        // Total entry size = 4 (length field) + content.len().
        // We want (4 + content.len()) % align == 0.
        let total = 4 + content.len();
        let remainder = total % align;
        if remainder != 0 {
            let padding = align - remainder;
            for _ in 0..padding {
                content.push(DW_CFA_NOP);
            }
        }
    }

    // Assemble the final CIE: length prefix + content.
    let mut result = Vec::with_capacity(4 + content.len());
    dwarf::write_u32_le(&mut result, content.len() as u32);
    result.extend_from_slice(&content);
    result
}

/// Serialize a Frame Description Entry into its DWARF binary representation.
///
/// Layout (32-bit DWARF format):
/// ```text
/// +----------------------------------------+
/// | length           : u32                |  Total FDE length (excl. this field)
/// | CIE_pointer      : u32               |  Offset of associated CIE
/// | initial_location : addr (4 or 8)     |  Function start address
/// | address_range    : addr (4 or 8)     |  Function code size
/// | instructions     : bytes              |  CFI instructions
/// | padding          : DW_CFA_nop x N     |  Pad to address_size alignment
/// +----------------------------------------+
/// ```
pub fn serialize_fde(fde: &FrameDescriptionEntry, cie_offset: u32, address_size: u8) -> Vec<u8> {
    let mut content = Vec::new();

    // CIE_pointer: offset of the CIE in .debug_frame.
    dwarf::write_u32_le(&mut content, cie_offset);

    // initial_location: start of the function (target-width address).
    dwarf::write_address(&mut content, fde.initial_location, address_size);

    // address_range: function size (target-width address).
    dwarf::write_address(&mut content, fde.address_range, address_size);

    // CFI instructions.
    content.extend_from_slice(&fde.instructions);

    // Pad to address_size alignment with DW_CFA_nop.
    let align = address_size as usize;
    if align > 0 {
        let total = 4 + content.len();
        let remainder = total % align;
        if remainder != 0 {
            let padding = align - remainder;
            for _ in 0..padding {
                content.push(DW_CFA_NOP);
            }
        }
    }

    // Assemble: length prefix + content.
    let mut result = Vec::with_capacity(4 + content.len());
    dwarf::write_u32_le(&mut result, content.len() as u32);
    result.extend_from_slice(&content);
    result
}

/// Serialize a complete `.debug_frame` section containing one CIE followed
/// by all FDEs.
///
/// The CIE is always at offset 0 in the section. Each FDE's `CIE_pointer`
/// field is set to 0 to reference this CIE.
pub fn serialize_debug_frame(
    cie: &CommonInformationEntry,
    fdes: &[FrameDescriptionEntry],
    address_size: u8,
) -> Vec<u8> {
    let cie_bytes = serialize_cie(cie);
    let cie_offset = 0u32; // CIE is always the first entry.
    let mut result = cie_bytes;

    for fde in fdes {
        let fde_bytes = serialize_fde(fde, cie_offset, address_size);
        result.extend_from_slice(&fde_bytes);
    }

    result
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // CIE Construction Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cie_x86_64() {
        let cie = build_cie_for_arch(Architecture::X86_64, 8);
        assert_eq!(cie.code_alignment_factor, 1);
        assert_eq!(cie.data_alignment_factor, -8);
        assert_eq!(cie.return_address_register, 16); // RIP
        assert_eq!(cie.version, dwarf::DWARF_VERSION as u8);
        assert_eq!(cie.address_size, 8);
        assert_eq!(cie.segment_size, 0);
        assert!(cie.augmentation.is_empty());
        assert!(!cie.initial_instructions.is_empty());
    }

    #[test]
    fn test_cie_i686() {
        let cie = build_cie_for_arch(Architecture::I686, 4);
        assert_eq!(cie.code_alignment_factor, 1);
        assert_eq!(cie.data_alignment_factor, -4);
        assert_eq!(cie.return_address_register, 8); // EIP
        assert_eq!(cie.address_size, 4);
    }

    #[test]
    fn test_cie_aarch64() {
        let cie = build_cie_for_arch(Architecture::Aarch64, 8);
        assert_eq!(cie.code_alignment_factor, 4);
        assert_eq!(cie.data_alignment_factor, -8);
        assert_eq!(cie.return_address_register, 30); // LR (x30)
        assert_eq!(cie.address_size, 8);
    }

    #[test]
    fn test_cie_riscv64() {
        let cie = build_cie_for_arch(Architecture::Riscv64, 8);
        assert_eq!(cie.code_alignment_factor, 2);
        assert_eq!(cie.data_alignment_factor, -8);
        assert_eq!(cie.return_address_register, 1); // RA (x1)
        assert_eq!(cie.address_size, 8);
    }

    // -----------------------------------------------------------------------
    // CIE Serialization Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cie_serialization_cie_id() {
        let cie = build_cie_for_arch(Architecture::X86_64, 8);
        let data = serialize_cie(&cie);
        // Bytes 4..8 must be 0xFFFFFFFF (CIE_id for .debug_frame).
        assert!(data.len() >= 8);
        let cie_id = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        assert_eq!(cie_id, 0xFFFF_FFFF);
    }

    #[test]
    fn test_cie_serialization_version() {
        let cie = build_cie_for_arch(Architecture::X86_64, 8);
        let data = serialize_cie(&cie);
        // Byte 8 is the version field, which should be 4.
        assert!(data.len() > 8);
        assert_eq!(data[8], 4);
    }

    #[test]
    fn test_cie_serialization_augmentation_null() {
        let cie = build_cie_for_arch(Architecture::X86_64, 8);
        let data = serialize_cie(&cie);
        // Byte 9 should be 0x00 (empty augmentation string, null-terminated).
        assert!(data.len() > 9);
        assert_eq!(data[9], 0x00);
    }

    #[test]
    fn test_cie_serialization_length_field() {
        let cie = build_cie_for_arch(Architecture::X86_64, 8);
        let data = serialize_cie(&cie);
        let length = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        // Length should equal total size minus the 4-byte length field.
        assert_eq!(length as usize, data.len() - 4);
    }

    #[test]
    fn test_cie_serialization_alignment_padding() {
        // For 8-byte address size, total CIE size should be 8-byte aligned.
        let cie = build_cie_for_arch(Architecture::X86_64, 8);
        let data = serialize_cie(&cie);
        assert_eq!(
            data.len() % 8,
            0,
            "CIE not aligned to 8 bytes (len = {})",
            data.len()
        );

        // For 4-byte address size, total CIE size should be 4-byte aligned.
        let cie4 = build_cie_for_arch(Architecture::I686, 4);
        let data4 = serialize_cie(&cie4);
        assert_eq!(
            data4.len() % 4,
            0,
            "CIE not aligned to 4 bytes (len = {})",
            data4.len()
        );
    }

    #[test]
    fn test_cie_serialization_includes_initial_instructions() {
        let cie = build_cie_for_arch(Architecture::X86_64, 8);
        let instr_len = cie.initial_instructions.len();
        let data = serialize_cie(&cie);
        // The data must be long enough to contain all fields plus instructions.
        // Minimum overhead: 4(length) + 4(CIE_id) + 1(version) + 1(aug NUL) +
        //   1(addr_size) + 1(seg_size) + 1(code_align ULEB) + 1(data_align SLEB) +
        //   1(return_reg ULEB) = 15 bytes.
        assert!(
            data.len() >= 15 + instr_len,
            "CIE data too short to contain initial instructions"
        );
    }

    // -----------------------------------------------------------------------
    // CFI Instruction Encoding Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cfi_def_cfa() {
        let mut builder = CfiInstructionBuilder::new(1, -8);
        builder.def_cfa(7, 8);
        let bytes = builder.build();
        // DW_CFA_def_cfa = 0x0c, reg 7 = 0x07, offset 8 = 0x08.
        assert_eq!(bytes, vec![0x0c, 0x07, 0x08]);
    }

    #[test]
    fn test_cfi_offset_compact() {
        let mut builder = CfiInstructionBuilder::new(1, -8);
        builder.offset(16, 1);
        let bytes = builder.build();
        // DW_CFA_offset | 16 = 0x80 | 0x10 = 0x90, factored_offset 1 = 0x01.
        assert_eq!(bytes, vec![0x90, 0x01]);
    }

    #[test]
    fn test_cfi_offset_extended() {
        let mut builder = CfiInstructionBuilder::new(1, -8);
        builder.offset(70, 1);
        let bytes = builder.build();
        // DW_CFA_offset_extended = 0x05, register 70 = 0x46, factored_offset 1 = 0x01.
        assert_eq!(bytes, vec![0x05, 0x46, 0x01]);
    }

    #[test]
    fn test_cfi_advance_loc_compact() {
        let mut builder = CfiInstructionBuilder::new(1, -8);
        builder.advance_loc(4);
        let bytes = builder.build();
        // DW_CFA_advance_loc | 4 = 0x40 | 0x04 = 0x44.
        assert_eq!(bytes, vec![0x44]);
    }

    #[test]
    fn test_cfi_advance_loc1() {
        let mut builder = CfiInstructionBuilder::new(1, -8);
        builder.advance_loc(100);
        let bytes = builder.build();
        // 100 > 63, so use advance_loc1.
        // DW_CFA_advance_loc1 = 0x02, delta = 100 (0x64).
        assert_eq!(bytes, vec![0x02, 0x64]);
    }

    #[test]
    fn test_cfi_advance_loc2() {
        let mut builder = CfiInstructionBuilder::new(1, -8);
        builder.advance_loc(256);
        let bytes = builder.build();
        // 256 > 255, so use advance_loc2.
        // DW_CFA_advance_loc2 = 0x03, delta = 256 (0x00, 0x01 LE).
        assert_eq!(bytes, vec![0x03, 0x00, 0x01]);
    }

    #[test]
    fn test_cfi_advance_loc4() {
        let mut builder = CfiInstructionBuilder::new(1, -8);
        builder.advance_loc(70000);
        let bytes = builder.build();
        // 70000 > 65535, so use advance_loc4.
        assert_eq!(bytes[0], 0x04);
        let delta = u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]);
        assert_eq!(delta, 70000);
    }

    #[test]
    fn test_cfi_advance_loc_factored() {
        // AArch64: code_alignment_factor = 4, so 16 bytes -> factored delta = 4.
        let mut builder = CfiInstructionBuilder::new(4, -8);
        builder.advance_loc(16);
        let bytes = builder.build();
        // 16 / 4 = 4, fits in 6 bits -> compact encoding.
        assert_eq!(bytes, vec![0x40 | 0x04]);
    }

    #[test]
    fn test_cfi_same_value() {
        let mut builder = CfiInstructionBuilder::new(1, -8);
        builder.same_value(30);
        let bytes = builder.build();
        // DW_CFA_same_value = 0x08, register 30 = 0x1e.
        assert_eq!(bytes, vec![0x08, 0x1e]);
    }

    #[test]
    fn test_cfi_remember_restore_state() {
        let mut builder = CfiInstructionBuilder::new(1, -8);
        builder.remember_state();
        builder.restore_state();
        let bytes = builder.build();
        assert_eq!(bytes, vec![0x0a, 0x0b]);
    }

    #[test]
    fn test_cfi_nop() {
        let mut builder = CfiInstructionBuilder::new(1, -8);
        builder.nop();
        let bytes = builder.build();
        assert_eq!(bytes, vec![0x00]);
    }

    #[test]
    fn test_cfi_def_cfa_register() {
        let mut builder = CfiInstructionBuilder::new(1, -8);
        builder.def_cfa_register(6);
        let bytes = builder.build();
        assert_eq!(bytes, vec![0x0d, 0x06]);
    }

    #[test]
    fn test_cfi_def_cfa_offset() {
        let mut builder = CfiInstructionBuilder::new(1, -8);
        builder.def_cfa_offset(16);
        let bytes = builder.build();
        assert_eq!(bytes, vec![0x0e, 0x10]);
    }

    // -----------------------------------------------------------------------
    // FDE Construction Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fde_with_frame_pointer() {
        let info = FunctionFrameInfo {
            start_address: 0x1000,
            size: 64,
            frame_size: 32,
            uses_frame_pointer: true,
            callee_saved_regs: vec![],
            prologue_size: 4,
            epilogue_offset: 0,
        };
        let fde = build_fde(&info, Architecture::X86_64);
        assert_eq!(fde.initial_location, 0x1000);
        assert_eq!(fde.address_range, 64);
        assert!(!fde.instructions.is_empty());
    }

    #[test]
    fn test_fde_without_frame_pointer() {
        let info = FunctionFrameInfo {
            start_address: 0x2000,
            size: 128,
            frame_size: 48,
            uses_frame_pointer: false,
            callee_saved_regs: vec![],
            prologue_size: 7,
            epilogue_offset: 0,
        };
        let fde = build_fde(&info, Architecture::X86_64);
        assert_eq!(fde.initial_location, 0x2000);
        assert_eq!(fde.address_range, 128);
        assert!(!fde.instructions.is_empty());
    }

    #[test]
    fn test_fde_aarch64() {
        let info = FunctionFrameInfo {
            start_address: 0x3000,
            size: 80,
            frame_size: 32,
            uses_frame_pointer: true,
            callee_saved_regs: vec![],
            prologue_size: 8,
            epilogue_offset: 0,
        };
        let fde = build_fde(&info, Architecture::Aarch64);
        assert_eq!(fde.initial_location, 0x3000);
        assert!(!fde.instructions.is_empty());
    }

    #[test]
    fn test_fde_riscv64() {
        let info = FunctionFrameInfo {
            start_address: 0x4000,
            size: 120,
            frame_size: 48,
            uses_frame_pointer: true,
            callee_saved_regs: vec![],
            prologue_size: 16,
            epilogue_offset: 0,
        };
        let fde = build_fde(&info, Architecture::Riscv64);
        assert_eq!(fde.initial_location, 0x4000);
        assert!(!fde.instructions.is_empty());
    }

    #[test]
    fn test_fde_i686() {
        let info = FunctionFrameInfo {
            start_address: 0x5000,
            size: 40,
            frame_size: 16,
            uses_frame_pointer: true,
            callee_saved_regs: vec![],
            prologue_size: 3,
            epilogue_offset: 0,
        };
        let fde = build_fde(&info, Architecture::I686);
        assert_eq!(fde.initial_location, 0x5000);
        assert!(!fde.instructions.is_empty());
    }

    // -----------------------------------------------------------------------
    // FDE Serialization Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fde_serialization_cie_pointer() {
        let fde = FrameDescriptionEntry {
            cie_offset: 0,
            initial_location: 0x1000,
            address_range: 64,
            instructions: vec![0x44], // advance_loc 4
        };
        let data = serialize_fde(&fde, 0, 8);
        // Bytes 4..8 should be the CIE_pointer (0).
        let cie_ptr = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        assert_eq!(cie_ptr, 0);
    }

    #[test]
    fn test_fde_serialization_initial_location_64() {
        let fde = FrameDescriptionEntry {
            cie_offset: 0,
            initial_location: 0xDEAD_BEEF,
            address_range: 100,
            instructions: vec![],
        };
        let data = serialize_fde(&fde, 0, 8);
        // After 4 (length) + 4 (CIE_ptr) = offset 8, next 8 bytes = initial_location.
        assert!(data.len() >= 16);
        let loc = u64::from_le_bytes([
            data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
        ]);
        assert_eq!(loc, 0xDEAD_BEEF);
    }

    #[test]
    fn test_fde_serialization_initial_location_32() {
        let fde = FrameDescriptionEntry {
            cie_offset: 0,
            initial_location: 0x8000,
            address_range: 50,
            instructions: vec![],
        };
        let data = serialize_fde(&fde, 0, 4);
        // After 4 (length) + 4 (CIE_ptr) = offset 8, next 4 bytes = initial_location.
        assert!(data.len() >= 12);
        let loc = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        assert_eq!(loc, 0x8000);
    }

    #[test]
    fn test_fde_serialization_length_field() {
        let fde = FrameDescriptionEntry {
            cie_offset: 0,
            initial_location: 0x1000,
            address_range: 64,
            instructions: vec![0x44, 0x0c, 0x07, 0x10],
        };
        let data = serialize_fde(&fde, 0, 8);
        let length = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert_eq!(length as usize, data.len() - 4);
    }

    // -----------------------------------------------------------------------
    // Full .debug_frame Serialization Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_debug_frame_cie_plus_fdes() {
        let cie = build_cie_for_arch(Architecture::X86_64, 8);
        let fde1 = FrameDescriptionEntry {
            cie_offset: 0,
            initial_location: 0x1000,
            address_range: 64,
            instructions: vec![0x44],
        };
        let fde2 = FrameDescriptionEntry {
            cie_offset: 0,
            initial_location: 0x2000,
            address_range: 128,
            instructions: vec![0x48],
        };
        let section = serialize_debug_frame(&cie, &[fde1, fde2], 8);

        // First entry: CIE.
        assert!(!section.is_empty());
        let cie_length = u32::from_le_bytes([section[0], section[1], section[2], section[3]]);
        let cie_total = 4 + cie_length as usize;

        // Verify the CIE id.
        let cie_id = u32::from_le_bytes([section[4], section[5], section[6], section[7]]);
        assert_eq!(cie_id, 0xFFFF_FFFF);

        // Second entry: first FDE.
        let fde1_offset = cie_total;
        assert!(section.len() > fde1_offset + 8);
        let fde1_cie_ptr = u32::from_le_bytes([
            section[fde1_offset + 4],
            section[fde1_offset + 5],
            section[fde1_offset + 6],
            section[fde1_offset + 7],
        ]);
        assert_eq!(
            fde1_cie_ptr, 0,
            "FDE1 CIE_pointer should reference CIE at offset 0"
        );

        // Third entry: second FDE.
        let fde1_length = u32::from_le_bytes([
            section[fde1_offset],
            section[fde1_offset + 1],
            section[fde1_offset + 2],
            section[fde1_offset + 3],
        ]);
        let fde2_offset = fde1_offset + 4 + fde1_length as usize;
        assert!(section.len() > fde2_offset + 8);
        let fde2_cie_ptr = u32::from_le_bytes([
            section[fde2_offset + 4],
            section[fde2_offset + 5],
            section[fde2_offset + 6],
            section[fde2_offset + 7],
        ]);
        assert_eq!(
            fde2_cie_ptr, 0,
            "FDE2 CIE_pointer should reference CIE at offset 0"
        );
    }

    #[test]
    fn test_debug_frame_cie_id_is_correct() {
        let cie = build_cie_for_arch(Architecture::I686, 4);
        let section = serialize_debug_frame(&cie, &[], 4);
        // CIE_id at bytes 4..8.
        assert!(section.len() >= 8);
        let cie_id = u32::from_le_bytes([section[4], section[5], section[6], section[7]]);
        assert_eq!(
            cie_id, 0xFFFF_FFFF,
            "CIE_id should be 0xFFFFFFFF for .debug_frame"
        );
    }

    #[test]
    fn test_debug_frame_fde_pointers_all_reference_same_cie() {
        let cie = build_cie_for_arch(Architecture::Aarch64, 8);
        let fdes: Vec<FrameDescriptionEntry> = (0..5)
            .map(|i| FrameDescriptionEntry {
                cie_offset: 0,
                initial_location: 0x1000 * (i + 1),
                address_range: 32,
                instructions: vec![0x41],
            })
            .collect();
        let section = serialize_debug_frame(&cie, &fdes, 8);

        // Walk entries and verify all FDE CIE_pointers are 0.
        let cie_len = u32::from_le_bytes([section[0], section[1], section[2], section[3]]);
        let mut pos = 4 + cie_len as usize;
        let mut fde_count = 0;
        while pos + 8 <= section.len() {
            let fde_len = u32::from_le_bytes([
                section[pos],
                section[pos + 1],
                section[pos + 2],
                section[pos + 3],
            ]);
            if fde_len == 0 {
                break;
            }
            let cie_ptr = u32::from_le_bytes([
                section[pos + 4],
                section[pos + 5],
                section[pos + 6],
                section[pos + 7],
            ]);
            assert_eq!(cie_ptr, 0, "FDE #{} CIE_pointer should be 0", fde_count);
            fde_count += 1;
            pos += 4 + fde_len as usize;
        }
        assert_eq!(fde_count, 5);
    }

    // -----------------------------------------------------------------------
    // Architecture Register Number Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_x86_64_return_address_register() {
        assert_eq!(x86_64_regs::RA, 16);
    }

    #[test]
    fn test_i686_return_address_register() {
        assert_eq!(i686_regs::RA, 8);
    }

    #[test]
    fn test_aarch64_return_address_register() {
        // Return address is in x30 (LR).
        assert_eq!(aarch64_regs::X30, 30);
    }

    #[test]
    fn test_riscv64_return_address_register() {
        // Return address is in x1 (RA).
        assert_eq!(riscv64_regs::X1, 1);
    }

    // -----------------------------------------------------------------------
    // FrameInfoEmitter Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_frame_info_emitter_empty() {
        let emitter = FrameInfoEmitter::new(Architecture::X86_64, 8);
        let data = emitter.emit(&[]);
        // Should contain at least a CIE.
        assert!(!data.is_empty());
        let cie_id = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        assert_eq!(cie_id, 0xFFFF_FFFF);
    }

    #[test]
    fn test_frame_info_emitter_with_functions() {
        let emitter = FrameInfoEmitter::new(Architecture::X86_64, 8);
        let functions = vec![
            FunctionFrameInfo {
                start_address: 0x1000,
                size: 64,
                frame_size: 16,
                uses_frame_pointer: true,
                callee_saved_regs: vec![],
                prologue_size: 4,
                epilogue_offset: 0,
            },
            FunctionFrameInfo {
                start_address: 0x2000,
                size: 128,
                frame_size: 32,
                uses_frame_pointer: false,
                callee_saved_regs: vec![],
                prologue_size: 7,
                epilogue_offset: 0,
            },
        ];
        let data = emitter.emit(&functions);
        // Should contain CIE + 2 FDEs.
        assert!(data.len() > 20, "Section too small for CIE + 2 FDEs");
    }

    #[test]
    fn test_frame_info_emitter_i686() {
        let emitter = FrameInfoEmitter::new(Architecture::I686, 4);
        let data = emitter.emit(&[FunctionFrameInfo {
            start_address: 0x8000,
            size: 32,
            frame_size: 8,
            uses_frame_pointer: true,
            callee_saved_regs: vec![],
            prologue_size: 3,
            epilogue_offset: 0,
        }]);
        assert!(!data.is_empty());
    }

    #[test]
    fn test_frame_info_emitter_aarch64() {
        let emitter = FrameInfoEmitter::new(Architecture::Aarch64, 8);
        let data = emitter.emit(&[FunctionFrameInfo {
            start_address: 0x10000,
            size: 96,
            frame_size: 48,
            uses_frame_pointer: true,
            callee_saved_regs: vec![],
            prologue_size: 8,
            epilogue_offset: 0,
        }]);
        assert!(!data.is_empty());
    }

    #[test]
    fn test_frame_info_emitter_riscv64() {
        let emitter = FrameInfoEmitter::new(Architecture::Riscv64, 8);
        let data = emitter.emit(&[FunctionFrameInfo {
            start_address: 0x20000,
            size: 160,
            frame_size: 64,
            uses_frame_pointer: true,
            callee_saved_regs: vec![],
            prologue_size: 16,
            epilogue_offset: 0,
        }]);
        assert!(!data.is_empty());
    }

    #[test]
    fn test_build_function_fde_with_callee_saved() {
        let info = FunctionFrameInfo {
            start_address: 0x5000,
            size: 200,
            frame_size: 64,
            uses_frame_pointer: true,
            callee_saved_regs: vec![
                (x86_64_regs::RBX, -24), // RBX saved at CFA-24
                (x86_64_regs::R12, -32), // R12 saved at CFA-32
            ],
            prologue_size: 10,
            epilogue_offset: 0,
        };
        let fde = build_function_fde(&info, Architecture::X86_64, -8);
        assert_eq!(fde.initial_location, 0x5000);
        assert_eq!(fde.address_range, 200);
        // Should contain instructions for the callee-saved registers.
        assert!(!fde.instructions.is_empty());
    }

    #[test]
    fn test_build_function_fde_with_epilogue() {
        let info = FunctionFrameInfo {
            start_address: 0x6000,
            size: 100,
            frame_size: 32,
            uses_frame_pointer: true,
            callee_saved_regs: vec![(x86_64_regs::RBX, -24)],
            prologue_size: 5,
            epilogue_offset: 90,
        };
        let fde = build_function_fde(&info, Architecture::X86_64, -8);
        assert_eq!(fde.initial_location, 0x6000);
        assert_eq!(fde.address_range, 100);
        // Should have both prologue and epilogue instructions.
        assert!(!fde.instructions.is_empty());
    }

    #[test]
    fn test_zero_frame_size_function() {
        // A leaf function with no frame allocation.
        let info = FunctionFrameInfo {
            start_address: 0x7000,
            size: 10,
            frame_size: 0,
            uses_frame_pointer: false,
            callee_saved_regs: vec![],
            prologue_size: 0,
            epilogue_offset: 0,
        };
        let fde = build_fde(&info, Architecture::X86_64);
        assert_eq!(fde.initial_location, 0x7000);
        assert_eq!(fde.address_range, 10);
        // No prologue => empty instructions.
        assert!(fde.instructions.is_empty());
    }

    #[test]
    fn test_register_submodule_constants() {
        // Verify a selection of register constants across architectures.
        assert_eq!(x86_64_regs::RAX, 0);
        assert_eq!(x86_64_regs::RSP, 7);
        assert_eq!(x86_64_regs::RBP, 6);
        assert_eq!(x86_64_regs::R15, 15);
        assert_eq!(x86_64_regs::XMM0, 17);
        assert_eq!(x86_64_regs::XMM15, 32);

        assert_eq!(i686_regs::EAX, 0);
        assert_eq!(i686_regs::ESP, 4);
        assert_eq!(i686_regs::EBP, 5);

        assert_eq!(aarch64_regs::X0, 0);
        assert_eq!(aarch64_regs::X29, 29);
        assert_eq!(aarch64_regs::SP, 31);
        assert_eq!(aarch64_regs::V0, 64);
        assert_eq!(aarch64_regs::V31, 95);

        assert_eq!(riscv64_regs::X0, 0);
        assert_eq!(riscv64_regs::X2, 2); // SP
        assert_eq!(riscv64_regs::X8, 8); // s0/fp
        assert_eq!(riscv64_regs::F0, 32);
        assert_eq!(riscv64_regs::F31, 63);
    }
}
