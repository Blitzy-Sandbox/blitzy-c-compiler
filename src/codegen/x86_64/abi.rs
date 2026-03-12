//! System V AMD64 ABI implementation for the x86-64 code generation backend.
//!
//! This module defines the complete calling convention for x86-64 Linux,
//! following the *System V Application Binary Interface — AMD64 Architecture
//! Processor Supplement*. It covers:
//!
//! - **Register assignments**: 16 GPRs (`rax`–`r15`) + 16 XMM registers
//!   (`xmm0`–`xmm15`), with encoding indices matching the x86-64 ISA.
//! - **Argument classification**: The ABI classifies each parameter into one
//!   of five classes (INTEGER, SSE, MEMORY, X87, NO_CLASS) that determine
//!   whether it is passed in an integer register, an SSE register, or on the
//!   stack. Structs are classified per 8-byte "eightbyte" boundaries.
//! - **Register allocation metadata**: Constructs [`RegisterInfo`] consumed
//!   by the shared linear-scan register allocator, with caller-saved registers
//!   prioritised for shorter live ranges.
//! - **Stack frame layout**: Computes frame sizes, local/spill/callee-save
//!   offsets, 16-byte alignment, and red-zone eligibility for leaf functions.
//! - **Prologue / epilogue generation**: Emits [`MachineInstr`] sequences for
//!   function entry (push rbp, establish frame, save callee-saved registers,
//!   allocate locals) and exit (restore, deallocate, ret).
//! - **Return value handling**: Classifies return types as INTEGER (rax/rdx),
//!   SSE (xmm0/xmm1), MEMORY (hidden pointer), or Void.
//!
//! ## Key ABI Rules
//!
//! | Rule                                | Details                                        |
//! |-------------------------------------|------------------------------------------------|
//! | Integer argument registers          | rdi, rsi, rdx, rcx, r8, r9 (in that order)    |
//! | SSE argument registers              | xmm0–xmm7 (in that order)                     |
//! | Integer return registers            | rax, rdx                                       |
//! | SSE return registers                | xmm0, xmm1                                    |
//! | Callee-saved GPRs                   | rbx, r12, r13, r14, r15 (+ rbp as frame ptr)  |
//! | Stack alignment at CALL             | RSP must be 16-byte aligned                    |
//! | Red zone                            | 128 bytes below RSP for leaf functions         |
//! | Struct ≤ 16 bytes                   | Classified per eightbyte (register-passable)   |
//! | Struct > 16 bytes                   | Passed by hidden pointer (MEMORY class)        |
//!
//! ## Zero External Dependencies
//!
//! Only imports from `std` and internal crate modules. No external crates.

use std::collections::HashMap;

use crate::codegen::regalloc::{PhysReg, RegisterInfo};

// RegClass is re-imported for documentation and potential use in
// downstream construction paths that classify registers by class.
#[allow(unused_imports)]
use crate::codegen::regalloc::RegClass;
use crate::codegen::{MachineInstr, MachineOperand};
use crate::ir::types::IrType;

// Value is imported per schema specification for potential use in argument
// mapping contexts where IR values are associated with physical registers
// during ABI-compliant call lowering.
#[allow(unused_imports)]
use crate::ir::instructions::Value;

// ---------------------------------------------------------------------------
// x86-64 Machine Instruction Opcodes (prologue / epilogue only)
// ---------------------------------------------------------------------------
// These constants MUST match the values defined in `super::isel::opcodes`.
// They are duplicated here to avoid a circular or undeclared dependency on
// the `isel` submodule, which is not in this file's `depends_on_files`.

/// `push reg` — opcode used in function prologue to save callee-saved registers.
const OP_PUSH: u32 = 0x0600;
/// `pop reg` — opcode used in function epilogue to restore callee-saved registers.
const OP_POP: u32 = 0x0601;
/// `mov reg, reg` — used for `mov rbp, rsp` in prologue and `mov rsp, rbp` in epilogue.
const OP_MOV_RR: u32 = 0x0300;
/// `sub reg, imm` — used for `sub rsp, <frame_size>` in prologue.
const OP_SUB_RI: u32 = 0x0111;
/// `add reg, imm` — used for `add rsp, <frame_size>` in epilogue.
const OP_ADD_RI: u32 = 0x0101;
/// `ret` — function return instruction.
const OP_RET: u32 = 0x0520;

// =====================================================================
// Section 1: Physical Register Constants
// =====================================================================
// Encoding indices match the x86-64 ISA register numbering used by
// ModR/M and REX prefix generation in the encoder.

/// RAX — accumulator, return value register #1, caller-saved.
pub const RAX: PhysReg = PhysReg(0);
/// RCX — fourth integer argument register, caller-saved.
pub const RCX: PhysReg = PhysReg(1);
/// RDX — third integer argument register, return value register #2, caller-saved.
pub const RDX: PhysReg = PhysReg(2);
/// RBX — callee-saved general-purpose register.
pub const RBX: PhysReg = PhysReg(3);
/// RSP — stack pointer (not allocatable).
pub const RSP: PhysReg = PhysReg(4);
/// RBP — frame pointer (callee-saved, not allocatable).
pub const RBP: PhysReg = PhysReg(5);
/// RSI — second integer argument register, caller-saved.
pub const RSI: PhysReg = PhysReg(6);
/// RDI — first integer argument register, caller-saved.
pub const RDI: PhysReg = PhysReg(7);
/// R8 — fifth integer argument register, caller-saved.
pub const R8: PhysReg = PhysReg(8);
/// R9 — sixth integer argument register, caller-saved.
pub const R9: PhysReg = PhysReg(9);
/// R10 — caller-saved scratch register.
pub const R10: PhysReg = PhysReg(10);
/// R11 — caller-saved scratch register.
pub const R11: PhysReg = PhysReg(11);
/// R12 — callee-saved general-purpose register.
pub const R12: PhysReg = PhysReg(12);
/// R13 — callee-saved general-purpose register.
pub const R13: PhysReg = PhysReg(13);
/// R14 — callee-saved general-purpose register.
pub const R14: PhysReg = PhysReg(14);
/// R15 — callee-saved general-purpose register.
pub const R15: PhysReg = PhysReg(15);

// SSE / XMM registers — offset by 16 in the PhysReg numbering space
// so that `is_xmm_reg` and `xmm_encoding` can distinguish them from GPRs.

/// XMM0 — first SSE argument / return register, caller-saved.
pub const XMM0: PhysReg = PhysReg(16);
/// XMM1 — second SSE argument / return register, caller-saved.
pub const XMM1: PhysReg = PhysReg(17);
/// XMM2 — third SSE argument register, caller-saved.
pub const XMM2: PhysReg = PhysReg(18);
/// XMM3 — fourth SSE argument register, caller-saved.
pub const XMM3: PhysReg = PhysReg(19);
/// XMM4 — fifth SSE argument register, caller-saved.
pub const XMM4: PhysReg = PhysReg(20);
/// XMM5 — sixth SSE argument register, caller-saved.
pub const XMM5: PhysReg = PhysReg(21);
/// XMM6 — seventh SSE argument register, caller-saved.
pub const XMM6: PhysReg = PhysReg(22);
/// XMM7 — eighth SSE argument register, caller-saved.
pub const XMM7: PhysReg = PhysReg(23);
/// XMM8 — caller-saved SSE register (not an argument register).
pub const XMM8: PhysReg = PhysReg(24);
/// XMM9 — caller-saved SSE register.
pub const XMM9: PhysReg = PhysReg(25);
/// XMM10 — caller-saved SSE register.
pub const XMM10: PhysReg = PhysReg(26);
/// XMM11 — caller-saved SSE register.
pub const XMM11: PhysReg = PhysReg(27);
/// XMM12 — caller-saved SSE register.
pub const XMM12: PhysReg = PhysReg(28);
/// XMM13 — caller-saved SSE register.
pub const XMM13: PhysReg = PhysReg(29);
/// XMM14 — caller-saved SSE register.
pub const XMM14: PhysReg = PhysReg(30);
/// XMM15 — caller-saved SSE register.
pub const XMM15: PhysReg = PhysReg(31);

// =====================================================================
// Section 2: Register Classification Arrays
// =====================================================================

/// Integer argument registers in System V AMD64 calling convention order.
///
/// Functions receive their first six INTEGER-class arguments in these
/// registers. If more than six integer arguments are needed, the excess
/// are passed on the stack.
pub const INT_ARG_REGS: [PhysReg; 6] = [RDI, RSI, RDX, RCX, R8, R9];

/// SSE / floating-point argument registers in calling convention order.
///
/// The first eight SSE-class arguments are passed in `xmm0`–`xmm7`.
/// Overflow SSE arguments are passed on the stack.
pub const FLOAT_ARG_REGS: [PhysReg; 8] = [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7];

/// Integer return value registers.
///
/// Scalar integers and pointers are returned in `rax`. Structs that
/// classify as two INTEGER eightbytes are returned in `rax` + `rdx`.
pub const INT_RETURN_REGS: [PhysReg; 2] = [RAX, RDX];

/// SSE return value registers.
///
/// Scalar floats are returned in `xmm0`. Structs with two SSE
/// eightbytes are returned in `xmm0` + `xmm1`.
pub const FLOAT_RETURN_REGS: [PhysReg; 2] = [XMM0, XMM1];

/// Caller-saved (volatile) general-purpose registers.
///
/// These registers are clobbered by function calls and do not need to
/// be preserved across calls. They are preferred for allocation of
/// short-lived values that do not span call sites.
pub const CALLER_SAVED_GPRS: [PhysReg; 9] = [RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11];

/// All XMM registers are caller-saved in the System V AMD64 ABI.
pub const CALLER_SAVED_XMMS: [PhysReg; 16] = [
    XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14,
    XMM15,
];

/// Callee-saved (non-volatile) general-purpose registers.
///
/// Functions must preserve these registers across calls. If the
/// register allocator assigns any of these, the function prologue
/// must save them and the epilogue must restore them.
///
/// Note: `rbp` is also callee-saved but is handled separately as the
/// frame pointer and is excluded from the allocatable set.
pub const CALLEE_SAVED_GPRS: [PhysReg; 5] = [RBX, R12, R13, R14, R15];

// =====================================================================
// Section 3: Register Helper Functions
// =====================================================================

/// Returns `true` if `reg` is an XMM (SSE) register.
///
/// XMM registers occupy PhysReg indices 16–31 in the x86-64 numbering
/// scheme. GPRs occupy indices 0–15.
#[inline]
pub fn is_xmm_reg(reg: PhysReg) -> bool {
    reg.0 >= 16 && reg.0 <= 31
}

/// Returns the 4-bit XMM register encoding index (0–15) for instruction
/// encoding. Subtracts the PhysReg offset of 16.
///
/// # Panics
///
/// Debug-asserts that `reg` is actually an XMM register.
#[inline]
pub fn xmm_encoding(reg: PhysReg) -> u8 {
    debug_assert!(
        is_xmm_reg(reg),
        "xmm_encoding called on non-XMM register {:?}",
        reg
    );
    (reg.0 - 16) as u8
}

// =====================================================================
// Section 4: RegisterInfo for the Shared Register Allocator
// =====================================================================

/// Constructs a [`RegisterInfo`] descriptor for the x86-64 register file.
///
/// The returned descriptor is consumed by the shared linear-scan register
/// allocator. Registers are listed in **allocation priority order**:
/// caller-saved registers appear first (to avoid prologue/epilogue save
/// overhead when the value does not cross a call), followed by
/// callee-saved registers.
///
/// `RSP` and `RBP` are excluded from the allocatable set because they
/// serve as the stack pointer and frame pointer, respectively.
pub fn x86_64_register_info() -> RegisterInfo {
    let mut reg_names: HashMap<PhysReg, &'static str> = HashMap::new();

    // GPR names
    reg_names.insert(RAX, "rax");
    reg_names.insert(RCX, "rcx");
    reg_names.insert(RDX, "rdx");
    reg_names.insert(RBX, "rbx");
    reg_names.insert(RSP, "rsp");
    reg_names.insert(RBP, "rbp");
    reg_names.insert(RSI, "rsi");
    reg_names.insert(RDI, "rdi");
    reg_names.insert(R8, "r8");
    reg_names.insert(R9, "r9");
    reg_names.insert(R10, "r10");
    reg_names.insert(R11, "r11");
    reg_names.insert(R12, "r12");
    reg_names.insert(R13, "r13");
    reg_names.insert(R14, "r14");
    reg_names.insert(R15, "r15");

    // XMM names
    reg_names.insert(XMM0, "xmm0");
    reg_names.insert(XMM1, "xmm1");
    reg_names.insert(XMM2, "xmm2");
    reg_names.insert(XMM3, "xmm3");
    reg_names.insert(XMM4, "xmm4");
    reg_names.insert(XMM5, "xmm5");
    reg_names.insert(XMM6, "xmm6");
    reg_names.insert(XMM7, "xmm7");
    reg_names.insert(XMM8, "xmm8");
    reg_names.insert(XMM9, "xmm9");
    reg_names.insert(XMM10, "xmm10");
    reg_names.insert(XMM11, "xmm11");
    reg_names.insert(XMM12, "xmm12");
    reg_names.insert(XMM13, "xmm13");
    reg_names.insert(XMM14, "xmm14");
    reg_names.insert(XMM15, "xmm15");

    RegisterInfo {
        // Allocatable integer registers in priority order.
        // Caller-saved first (RAX..R11), then callee-saved (RBX, R12-R15).
        // RSP and RBP are excluded (stack/frame pointers).
        int_regs: vec![
            RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11, RBX, R12, R13, R14, R15,
        ],
        // All 16 XMM registers are allocatable (all caller-saved).
        float_regs: vec![
            XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM9, XMM10, XMM11, XMM12, XMM13,
            XMM14, XMM15,
        ],
        callee_saved_int: vec![RBX, R12, R13, R14, R15],
        // No callee-saved XMM registers in System V AMD64 ABI.
        callee_saved_float: vec![],
        reg_names,
    }
}

// =====================================================================
// Section 5: x86-64 Type Size / Alignment Helpers (LP64 model)
// =====================================================================

/// Returns the size in bytes of an IR type under the x86-64 LP64 model.
fn x86_64_type_size(ty: &IrType) -> u64 {
    match ty {
        IrType::Void => 0,
        IrType::I1 | IrType::I8 => 1,
        IrType::I16 => 2,
        IrType::I32 => 4,
        IrType::I64 => 8,
        IrType::F32 => 4,
        IrType::F64 => 8,
        IrType::Pointer(_) => 8,
        IrType::Array { element, count } => x86_64_type_size(element) * (*count as u64),
        IrType::Struct { fields, packed } => {
            if fields.is_empty() {
                return 0;
            }
            if *packed {
                return fields.iter().map(|f| x86_64_type_size(f)).sum();
            }
            let mut offset: u64 = 0;
            let mut max_align: u64 = 1;
            for field in fields {
                let align = x86_64_type_alignment(field);
                offset = (offset + align - 1) & !(align - 1);
                offset += x86_64_type_size(field);
                if align > max_align {
                    max_align = align;
                }
            }
            (offset + max_align - 1) & !(max_align - 1)
        }
        IrType::Function { .. } | IrType::Label => 0,
    }
}

/// Returns the alignment in bytes of an IR type under x86-64.
fn x86_64_type_alignment(ty: &IrType) -> u64 {
    match ty {
        IrType::Void | IrType::I1 | IrType::I8 => 1,
        IrType::I16 => 2,
        IrType::I32 | IrType::F32 => 4,
        IrType::I64 | IrType::F64 => 8,
        IrType::Pointer(_) => 8,
        IrType::Array { element, .. } => x86_64_type_alignment(element),
        IrType::Struct { fields, packed } => {
            if *packed {
                1
            } else {
                fields
                    .iter()
                    .map(|f| x86_64_type_alignment(f))
                    .max()
                    .unwrap_or(1)
            }
        }
        IrType::Function { .. } | IrType::Label => 1,
    }
}

/// Computes byte offsets for each field in a struct on x86-64.
fn x86_64_struct_field_offsets(fields: &[IrType], packed: bool) -> Vec<u64> {
    let mut offsets = Vec::with_capacity(fields.len());
    let mut offset: u64 = 0;
    for field in fields {
        if !packed {
            let align = x86_64_type_alignment(field);
            offset = (offset + align - 1) & !(align - 1);
        }
        offsets.push(offset);
        offset += x86_64_type_size(field);
    }
    offsets
}

// =====================================================================
// Section 6: Argument Classification (ABI Core)
// =====================================================================

/// Classification of a function argument for the System V AMD64 ABI.
///
/// The ABI classifies each parameter into one of five classes which
/// determine how the argument is passed: in an integer register, an
/// SSE register, on the stack, via x87, or not at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgumentClass {
    /// Passed in an integer register (`rdi`, `rsi`, `rdx`, `rcx`, `r8`, `r9`).
    Integer,
    /// Passed in an SSE register (`xmm0`–`xmm7`).
    Sse,
    /// Passed on the stack (struct too large, or register pool exhausted).
    Memory,
    /// x87 floating-point (long double). Rarely used in modern 64-bit code.
    X87,
    /// No class — used for `void` or padding eightbytes.
    NoClass,
}

/// Classification of a function return type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReturnClass {
    /// Returned in integer registers (`rax`, optionally `rdx`).
    Integer { regs: Vec<PhysReg> },
    /// Returned in SSE registers (`xmm0`, optionally `xmm1`).
    Sse { regs: Vec<PhysReg> },
    /// Returned via hidden pointer in `rdi`; callee writes result to it
    /// and returns the pointer in `rax`.
    Memory,
    /// Void return — no return value.
    Void,
}

/// Classifies a single IR type per the System V AMD64 ABI.
///
/// Returns a vector of [`ArgumentClass`] values, one per "eightbyte".
/// Scalar types produce a single-element vector. Structs ≤ 16 bytes
/// produce one or two elements. Structs > 16 bytes are MEMORY.
pub fn classify_type(ty: &IrType) -> Vec<ArgumentClass> {
    match ty {
        IrType::I1 | IrType::I8 | IrType::I16 | IrType::I32 | IrType::I64 => {
            vec![ArgumentClass::Integer]
        }
        IrType::Pointer(_) => vec![ArgumentClass::Integer],
        IrType::F32 | IrType::F64 => vec![ArgumentClass::Sse],
        IrType::Void => vec![ArgumentClass::NoClass],
        IrType::Struct { fields, packed } => {
            let size = x86_64_type_size(ty);
            if size == 0 {
                return vec![ArgumentClass::NoClass];
            }
            let offsets = x86_64_struct_field_offsets(fields, *packed);
            let fields_with_offsets: Vec<(IrType, u64)> =
                fields.iter().cloned().zip(offsets.into_iter()).collect();
            classify_aggregate(size, &fields_with_offsets)
        }
        IrType::Array { element, count } => {
            let size = x86_64_type_size(ty);
            if size == 0 || *count == 0 {
                return vec![ArgumentClass::NoClass];
            }
            if size > 16 {
                return vec![ArgumentClass::Memory];
            }
            let elem_size = x86_64_type_size(element);
            let fields_with_offsets: Vec<(IrType, u64)> = (0..*count)
                .map(|i| (element.as_ref().clone(), i as u64 * elem_size))
                .collect();
            classify_aggregate(size, &fields_with_offsets)
        }
        IrType::Function { .. } => vec![ArgumentClass::Integer],
        IrType::Label => vec![ArgumentClass::NoClass],
    }
}

/// Classifies an aggregate type by its eightbyte composition.
///
/// Per the ABI specification:
/// 1. If `size > 16` bytes → MEMORY.
/// 2. Divide into 8-byte chunks ("eightbytes").
/// 3. Classify each eightbyte by the fields that overlap it.
/// 4. Post-merge: if any eightbyte is MEMORY, the whole argument is MEMORY.
pub fn classify_aggregate(size: u64, fields: &[(IrType, u64)]) -> Vec<ArgumentClass> {
    if size > 16 {
        return vec![ArgumentClass::Memory];
    }
    if size == 0 {
        return vec![ArgumentClass::NoClass];
    }

    let num_eightbytes = if size <= 8 { 1 } else { 2 };
    let mut classes = vec![ArgumentClass::NoClass; num_eightbytes];

    for (field_ty, offset) in fields {
        let field_size = x86_64_type_size(field_ty);
        if field_size == 0 {
            continue;
        }
        let start_eb = (*offset / 8) as usize;
        let end_byte = offset + field_size;
        let end_eb = if end_byte == 0 {
            0
        } else {
            ((end_byte - 1) / 8) as usize
        };
        let field_class = classify_field_type(field_ty);
        for eb in start_eb..=end_eb.min(num_eightbytes - 1) {
            classes[eb] = merge_classes(classes[eb], field_class);
        }
    }

    // Post-merge: any MEMORY → all MEMORY
    if classes.iter().any(|c| *c == ArgumentClass::Memory) {
        return vec![ArgumentClass::Memory];
    }

    // Replace NoClass padding with Integer (per ABI rule)
    for class in &mut classes {
        if *class == ArgumentClass::NoClass {
            *class = ArgumentClass::Integer;
        }
    }

    classes
}

/// Determines the ABI class of a single scalar or pointer field.
fn classify_field_type(ty: &IrType) -> ArgumentClass {
    if ty.is_integer() || ty.is_pointer() {
        ArgumentClass::Integer
    } else if ty.is_float() {
        ArgumentClass::Sse
    } else if ty.is_struct() || ty.is_array() {
        // Recursively classify nested aggregates
        let classes = classify_type(ty);
        if classes.len() == 1 {
            classes[0]
        } else {
            ArgumentClass::Integer
        }
    } else if ty.is_void() {
        ArgumentClass::NoClass
    } else {
        ArgumentClass::Integer
    }
}

/// Merges two eightbyte classes per ABI rules (commutative operation).
///
/// The merge rules are:
/// - Same class → keep that class
/// - NoClass + X → X
/// - MEMORY + anything → MEMORY
/// - X87 + anything → MEMORY
/// - INTEGER + SSE → INTEGER
fn merge_classes(a: ArgumentClass, b: ArgumentClass) -> ArgumentClass {
    if a == b {
        return a;
    }
    if a == ArgumentClass::NoClass {
        return b;
    }
    if b == ArgumentClass::NoClass {
        return a;
    }
    if a == ArgumentClass::Memory || b == ArgumentClass::Memory {
        return ArgumentClass::Memory;
    }
    if a == ArgumentClass::X87 || b == ArgumentClass::X87 {
        return ArgumentClass::Memory;
    }
    // INTEGER + SSE → INTEGER per ABI specification
    ArgumentClass::Integer
}

/// Classifies a return type per the System V AMD64 ABI.
///
/// - Scalars and small aggregates (≤ 16 bytes) are returned in registers.
/// - Large aggregates (> 16 bytes) are returned via hidden pointer (`Memory`).
/// - `void` maps to `Void`.
pub fn classify_return_type(ty: &IrType) -> ReturnClass {
    match ty {
        IrType::Void => ReturnClass::Void,
        IrType::I1 | IrType::I8 | IrType::I16 | IrType::I32 | IrType::I64 => {
            ReturnClass::Integer { regs: vec![RAX] }
        }
        IrType::Pointer(_) => ReturnClass::Integer { regs: vec![RAX] },
        IrType::F32 | IrType::F64 => ReturnClass::Sse { regs: vec![XMM0] },
        IrType::Struct { .. } | IrType::Array { .. } => {
            let size = x86_64_type_size(ty);
            if size == 0 {
                return ReturnClass::Void;
            }
            if size > 16 {
                return ReturnClass::Memory;
            }
            let classes = classify_type(ty);
            classify_return_from_eightbytes(&classes)
        }
        IrType::Function { .. } => ReturnClass::Integer { regs: vec![RAX] },
        IrType::Label => ReturnClass::Void,
    }
}

/// Maps eightbyte classes to return registers.
///
/// For return values that span two eightbytes, each eightbyte is assigned
/// the next available register of the appropriate class:
/// - INTEGER eightbytes use `rax`, then `rdx`.
/// - SSE eightbytes use `xmm0`, then `xmm1`.
fn classify_return_from_eightbytes(classes: &[ArgumentClass]) -> ReturnClass {
    if classes.is_empty() || classes == [ArgumentClass::NoClass] {
        return ReturnClass::Void;
    }
    if classes == [ArgumentClass::Memory] {
        return ReturnClass::Memory;
    }

    let mut int_idx: usize = 0;
    let mut sse_idx: usize = 0;
    let mut int_regs = Vec::new();
    let mut sse_regs = Vec::new();

    for class in classes {
        match class {
            ArgumentClass::Integer => {
                if int_idx < INT_RETURN_REGS.len() {
                    int_regs.push(INT_RETURN_REGS[int_idx]);
                    int_idx += 1;
                } else {
                    return ReturnClass::Memory;
                }
            }
            ArgumentClass::Sse => {
                if sse_idx < FLOAT_RETURN_REGS.len() {
                    sse_regs.push(FLOAT_RETURN_REGS[sse_idx]);
                    sse_idx += 1;
                } else {
                    return ReturnClass::Memory;
                }
            }
            ArgumentClass::Memory => return ReturnClass::Memory,
            _ => {}
        }
    }

    if !int_regs.is_empty() && sse_regs.is_empty() {
        ReturnClass::Integer { regs: int_regs }
    } else if int_regs.is_empty() && !sse_regs.is_empty() {
        ReturnClass::Sse { regs: sse_regs }
    } else if !int_regs.is_empty() && !sse_regs.is_empty() {
        // Mixed integer + SSE eightbytes: report as Integer with all regs
        let mut all = int_regs;
        all.extend(sse_regs);
        ReturnClass::Integer { regs: all }
    } else {
        ReturnClass::Void
    }
}

// =====================================================================
// Section 7: Argument Location / Call Frame Setup
// =====================================================================

/// Describes where a single function argument is placed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArgumentLocation {
    /// The ABI classification of this argument.
    pub class: ArgumentClass,
    /// Physical register assigned, if passed in a register.
    pub register: Option<PhysReg>,
    /// Stack offset from RSP at the call site, if passed on the stack.
    pub stack_offset: Option<i32>,
}

/// Complete layout of all function arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArgumentLayout {
    /// Per-argument location descriptors.
    pub locations: Vec<ArgumentLocation>,
    /// Total stack space (bytes) needed for overflow arguments.
    pub stack_space: u32,
    /// Number of integer argument registers consumed.
    pub int_regs_used: u32,
    /// Number of SSE argument registers consumed.
    pub sse_regs_used: u32,
}

/// Computes the argument locations for a function signature.
///
/// Walks the parameter types in order, classifying each and assigning
/// registers or stack slots per the System V AMD64 ABI rules:
///
/// 1. INTEGER-class arguments are assigned `rdi`, `rsi`, `rdx`, `rcx`,
///    `r8`, `r9` (in that order). Overflow goes to the stack.
/// 2. SSE-class arguments are assigned `xmm0`–`xmm7`. Overflow goes
///    to the stack.
/// 3. MEMORY-class arguments (large structs) are passed by hidden pointer
///    which itself consumes an integer register or stack slot.
/// 4. Multi-eightbyte aggregates must fit entirely in registers or go
///    entirely on the stack (no splitting across register/stack).
pub fn compute_argument_locations(param_types: &[IrType]) -> ArgumentLayout {
    let mut locations = Vec::with_capacity(param_types.len());
    let mut int_idx: usize = 0;
    let mut sse_idx: usize = 0;
    let mut stack_offset: i32 = 0;

    for ty in param_types {
        let classes = classify_type(ty);

        // Void parameter — should not appear but handle gracefully
        if classes.is_empty() || classes == [ArgumentClass::NoClass] {
            locations.push(ArgumentLocation {
                class: ArgumentClass::NoClass,
                register: None,
                stack_offset: None,
            });
            continue;
        }

        // Large struct: passed by hidden pointer (MEMORY class)
        if classes == [ArgumentClass::Memory] || classes.len() > 2 {
            if int_idx < INT_ARG_REGS.len() {
                locations.push(ArgumentLocation {
                    class: ArgumentClass::Memory,
                    register: Some(INT_ARG_REGS[int_idx]),
                    stack_offset: None,
                });
                int_idx += 1;
            } else {
                locations.push(ArgumentLocation {
                    class: ArgumentClass::Memory,
                    register: None,
                    stack_offset: Some(stack_offset),
                });
                stack_offset += 8;
            }
            continue;
        }

        // Multi-eightbyte: check if ALL eightbytes fit in registers.
        // If not, the entire argument goes on the stack (ABI rule).
        let needs_int = classes
            .iter()
            .filter(|c| **c == ArgumentClass::Integer)
            .count();
        let needs_sse = classes.iter().filter(|c| **c == ArgumentClass::Sse).count();
        let have_int = INT_ARG_REGS.len() - int_idx;
        let have_sse = FLOAT_ARG_REGS.len() - sse_idx;

        if needs_int > have_int || needs_sse > have_sse {
            // Not enough registers — pass entire argument on stack
            let size = x86_64_type_size(ty) as i32;
            locations.push(ArgumentLocation {
                class: classes[0],
                register: None,
                stack_offset: Some(stack_offset),
            });
            // Align stack slot to 8 bytes
            stack_offset += (size + 7) & !7;
            continue;
        }

        // Single eightbyte — assign one register
        if classes.len() == 1 {
            match classes[0] {
                ArgumentClass::Integer => {
                    locations.push(ArgumentLocation {
                        class: ArgumentClass::Integer,
                        register: Some(INT_ARG_REGS[int_idx]),
                        stack_offset: None,
                    });
                    int_idx += 1;
                }
                ArgumentClass::Sse => {
                    locations.push(ArgumentLocation {
                        class: ArgumentClass::Sse,
                        register: Some(FLOAT_ARG_REGS[sse_idx]),
                        stack_offset: None,
                    });
                    sse_idx += 1;
                }
                _ => {
                    locations.push(ArgumentLocation {
                        class: classes[0],
                        register: None,
                        stack_offset: Some(stack_offset),
                    });
                    stack_offset += 8;
                }
            }
        } else {
            // Two eightbytes — assign registers for the first eightbyte
            // (we track the "start" location for the caller; the second
            // eightbyte register is consumed but not separately tracked
            // since the struct is logically one argument).
            match classes[0] {
                ArgumentClass::Integer => {
                    locations.push(ArgumentLocation {
                        class: ArgumentClass::Integer,
                        register: Some(INT_ARG_REGS[int_idx]),
                        stack_offset: None,
                    });
                    int_idx += 1;
                }
                ArgumentClass::Sse => {
                    locations.push(ArgumentLocation {
                        class: ArgumentClass::Sse,
                        register: Some(FLOAT_ARG_REGS[sse_idx]),
                        stack_offset: None,
                    });
                    sse_idx += 1;
                }
                _ => {
                    locations.push(ArgumentLocation {
                        class: classes[0],
                        register: None,
                        stack_offset: Some(stack_offset),
                    });
                    stack_offset += 8;
                }
            }
            // Consume register for second eightbyte
            match classes[1] {
                ArgumentClass::Integer => {
                    int_idx += 1;
                }
                ArgumentClass::Sse => {
                    sse_idx += 1;
                }
                _ => {}
            }
        }
    }

    // Align total stack space to 8 bytes
    let total_stack = ((stack_offset as u32) + 7) & !7;

    ArgumentLayout {
        locations,
        stack_space: total_stack,
        int_regs_used: int_idx as u32,
        sse_regs_used: sse_idx as u32,
    }
}

// =====================================================================
// Section 8: Stack Frame Layout
// =====================================================================

/// Describes the stack frame layout for a function.
///
/// # Stack Layout (high to low addresses)
///
/// ```text
/// [return address]       ← old RSP (before CALL)
/// [saved RBP]            ← RBP points here (if frame pointer used)
/// [callee-saved regs]    ← pushed after mov rbp, rsp
/// [local variables]
/// [spill slots]          ← RSP points here (16-byte aligned)
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackFrame {
    /// Total frame size in bytes (the amount subtracted from RSP after
    /// saving callee-saved registers). Aligned to 16 bytes.
    pub frame_size: u32,
    /// Offset of the local variables area from RBP (negative).
    pub locals_offset: i32,
    /// Offset of the spill slots area from RBP (negative).
    pub spills_offset: i32,
    /// Offset of the callee-saved register save area from RBP (negative).
    pub callee_saves_offset: i32,
    /// Callee-saved registers that need saving/restoring.
    pub callee_saved_regs: Vec<PhysReg>,
    /// Whether the function uses RBP as a frame pointer.
    pub uses_frame_pointer: bool,
    /// Whether the red zone optimisation is applicable.
    ///
    /// Leaf functions (no calls) with total frame ≤ 128 bytes can use
    /// the 128-byte red zone below RSP without adjusting RSP, saving
    /// the cost of `sub rsp` / `add rsp` instructions.
    pub use_red_zone: bool,
}

/// Computes the stack frame layout for a function.
///
/// # Arguments
///
/// * `locals_size` — Total bytes needed for local variables.
/// * `spill_slots` — Number of 8-byte spill slots required by regalloc.
/// * `callee_saved` — Callee-saved registers that the function uses.
/// * `has_calls` — `true` if the function contains any `CALL` instructions.
///
/// # Red Zone
///
/// The red zone is a 128-byte area below RSP that leaf functions may
/// use without adjusting RSP. Eligibility requires:
/// - The function is a leaf (no calls)
/// - The total frame (callee saves + locals + spills) is ≤ 128 bytes
/// - No callee-saved registers need saving (to keep the frame trivial)
pub fn compute_stack_frame(
    locals_size: u32,
    spill_slots: u32,
    callee_saved: &[PhysReg],
    has_calls: bool,
) -> StackFrame {
    let spill_bytes = spill_slots * 8;
    let callee_save_bytes = (callee_saved.len() as u32) * 8;

    // Body = locals + spills (allocated via sub rsp after callee saves)
    let body_size = locals_size + spill_bytes;

    // Red zone eligibility:
    // - Leaf function (no calls)
    // - Total frame (callee saves + body) fits in 128 bytes
    // - No callee-saved pushes needed
    // Per the System V AMD64 ABI, the 128-byte area below RSP (the "red zone")
    // can be used by leaf functions without adjusting RSP, using RSP-relative
    // addressing. This eliminates the need for a frame pointer and prologue.
    let total_unaligned = callee_save_bytes + body_size;
    // Red zone requires: leaf function, small frame, no callee saves,
    // AND no locals/spills that would use RBP-relative addressing (since
    // red zone skips the frame pointer setup entirely).
    let use_red_zone =
        !has_calls && total_unaligned <= 128 && callee_saved.is_empty() && body_size == 0;

    // Compute aligned frame size for the sub rsp instruction.
    // After push rbp (8 bytes) the stack is 16-byte aligned (because the
    // CALL that invoked us already pushed the 8-byte return address).
    // Then N callee-saved pushes each subtract 8 bytes.
    //   If N is even: RSP remains 16-byte aligned
    //   If N is odd:  RSP is 8-byte aligned
    // We need (RSP - body_size) to be 16-byte aligned.
    let num_pushes = callee_saved.len() as u32;
    let misalignment = (num_pushes % 2) * 8;
    let adjusted_body = body_size + misalignment;
    let aligned_body = if adjusted_body == 0 {
        0
    } else {
        align_stack_size(adjusted_body) - misalignment
    };

    let frame_size = if use_red_zone { 0 } else { aligned_body };

    // Compute offsets from RBP (negative = below RBP).
    // RBP points at the saved RBP value on the stack.
    let callee_saves_offset = -(callee_save_bytes as i32);
    let locals_offset = callee_saves_offset - (locals_size as i32);
    let spills_offset = locals_offset - (spill_bytes as i32);

    StackFrame {
        frame_size,
        locals_offset,
        spills_offset,
        callee_saves_offset,
        callee_saved_regs: callee_saved.to_vec(),
        uses_frame_pointer: !use_red_zone,
        use_red_zone,
    }
}

// =====================================================================
// Section 9: Prologue / Epilogue Generation
// =====================================================================

/// Generates the function prologue as a sequence of [`MachineInstr`].
///
/// Standard x86-64 prologue:
/// ```text
/// push rbp                  ; save old frame pointer
/// mov  rbp, rsp             ; establish new frame pointer
/// push <callee-saved>       ; save each callee-saved register
/// sub  rsp, <frame_size>    ; allocate locals + spills
/// ```
///
/// If the red zone is used, the prologue is empty (no instructions).
pub fn generate_prologue(frame: &StackFrame) -> Vec<MachineInstr> {
    if frame.use_red_zone {
        return Vec::new();
    }

    let mut instrs = Vec::new();

    // push rbp
    instrs.push(MachineInstr::with_operands(
        OP_PUSH,
        vec![MachineOperand::Register(RBP)],
    ));

    // mov rbp, rsp
    instrs.push(MachineInstr::with_operands(
        OP_MOV_RR,
        vec![MachineOperand::Register(RBP), MachineOperand::Register(RSP)],
    ));

    // Push callee-saved registers (in forward order)
    for &reg in &frame.callee_saved_regs {
        instrs.push(MachineInstr::with_operands(
            OP_PUSH,
            vec![MachineOperand::Register(reg)],
        ));
    }

    // sub rsp, frame_size (allocate locals + spills)
    if frame.frame_size > 0 {
        instrs.push(MachineInstr::with_operands(
            OP_SUB_RI,
            vec![
                MachineOperand::Register(RSP),
                MachineOperand::Immediate(frame.frame_size as i64),
            ],
        ));
    }

    instrs
}

/// Generates the function epilogue as a sequence of [`MachineInstr`].
///
/// Standard x86-64 epilogue (mirror of prologue in reverse):
/// ```text
/// add  rsp, <frame_size>    ; deallocate locals + spills
/// pop  <callee-saved>       ; restore each callee-saved register (reverse)
/// pop  rbp                  ; restore old frame pointer
/// ret
/// ```
///
/// If the red zone was used, the epilogue is just `ret`.
pub fn generate_epilogue(frame: &StackFrame) -> Vec<MachineInstr> {
    let mut instrs = Vec::new();

    if frame.use_red_zone {
        instrs.push(MachineInstr::new(OP_RET));
        return instrs;
    }

    // add rsp, frame_size (deallocate locals + spills)
    if frame.frame_size > 0 {
        instrs.push(MachineInstr::with_operands(
            OP_ADD_RI,
            vec![
                MachineOperand::Register(RSP),
                MachineOperand::Immediate(frame.frame_size as i64),
            ],
        ));
    }

    // Pop callee-saved registers in reverse order
    for &reg in frame.callee_saved_regs.iter().rev() {
        instrs.push(MachineInstr::with_operands(
            OP_POP,
            vec![MachineOperand::Register(reg)],
        ));
    }

    // pop rbp
    instrs.push(MachineInstr::with_operands(
        OP_POP,
        vec![MachineOperand::Register(RBP)],
    ));

    // ret
    instrs.push(MachineInstr::new(OP_RET));

    instrs
}

// =====================================================================
// Section 10: Stack Alignment Helpers
// =====================================================================

/// Rounds `size` up to the next 16-byte boundary.
///
/// The System V AMD64 ABI requires RSP to be 16-byte aligned at every
/// `CALL` instruction. This function ensures frame sizes satisfy that
/// constraint.
#[inline]
pub fn align_stack_size(size: u32) -> u32 {
    (size + 15) & !15
}

/// Computes the stack adjustment needed before a `CALL` instruction
/// to maintain 16-byte alignment.
///
/// The `CALL` instruction pushes 8 bytes (the return address), so RSP
/// must be 16-byte aligned *at* the `CALL`. That means RSP itself must
/// be a multiple of 16 just before executing `CALL`.
///
/// # Arguments
///
/// * `current_stack_offset` — Current RSP offset from the aligned base
///   (typically the frame size + any already-pushed outgoing args).
/// * `args_on_stack` — Total bytes of arguments to be passed on the stack.
///
/// # Returns
///
/// Number of padding bytes to insert (via `sub rsp, N`) before pushing
/// arguments and executing the `CALL`.
pub fn compute_call_stack_adjustment(current_stack_offset: u32, args_on_stack: u32) -> u32 {
    // After pushing args_on_stack bytes AND the 8-byte return address,
    // the callee's RSP must be 16-byte aligned.
    //
    // Total consumed below current RSP = args_on_stack + 8
    // We need (current_stack_offset + args_on_stack + 8) % 16 == 0
    let total = current_stack_offset + args_on_stack + 8;
    let remainder = total % 16;
    if remainder == 0 {
        0
    } else {
        16 - remainder
    }
}

// =====================================================================
// Section 11: Unit Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----- Register constant tests -----

    #[test]
    fn test_register_values() {
        assert_eq!(RAX.0, 0);
        assert_eq!(RCX.0, 1);
        assert_eq!(RDX.0, 2);
        assert_eq!(RBX.0, 3);
        assert_eq!(RSP.0, 4);
        assert_eq!(RBP.0, 5);
        assert_eq!(RSI.0, 6);
        assert_eq!(RDI.0, 7);
        assert_eq!(R8.0, 8);
        assert_eq!(R9.0, 9);
        assert_eq!(R10.0, 10);
        assert_eq!(R11.0, 11);
        assert_eq!(R12.0, 12);
        assert_eq!(R13.0, 13);
        assert_eq!(R14.0, 14);
        assert_eq!(R15.0, 15);
    }

    #[test]
    fn test_xmm_register_values() {
        assert_eq!(XMM0.0, 16);
        assert_eq!(XMM1.0, 17);
        assert_eq!(XMM2.0, 18);
        assert_eq!(XMM3.0, 19);
        assert_eq!(XMM4.0, 20);
        assert_eq!(XMM5.0, 21);
        assert_eq!(XMM6.0, 22);
        assert_eq!(XMM7.0, 23);
        assert_eq!(XMM8.0, 24);
        assert_eq!(XMM9.0, 25);
        assert_eq!(XMM10.0, 26);
        assert_eq!(XMM11.0, 27);
        assert_eq!(XMM12.0, 28);
        assert_eq!(XMM13.0, 29);
        assert_eq!(XMM14.0, 30);
        assert_eq!(XMM15.0, 31);
    }

    #[test]
    fn test_is_xmm_reg() {
        assert!(is_xmm_reg(XMM0));
        assert!(is_xmm_reg(XMM15));
        assert!(!is_xmm_reg(RAX));
        assert!(!is_xmm_reg(R15));
    }

    #[test]
    fn test_xmm_encoding() {
        assert_eq!(xmm_encoding(XMM0), 0);
        assert_eq!(xmm_encoding(XMM1), 1);
        assert_eq!(xmm_encoding(XMM7), 7);
        assert_eq!(xmm_encoding(XMM15), 15);
    }

    // ----- Argument classification tests -----

    #[test]
    fn test_classify_i32() {
        assert_eq!(classify_type(&IrType::I32), vec![ArgumentClass::Integer]);
    }

    #[test]
    fn test_classify_i64() {
        assert_eq!(classify_type(&IrType::I64), vec![ArgumentClass::Integer]);
    }

    #[test]
    fn test_classify_i1() {
        assert_eq!(classify_type(&IrType::I1), vec![ArgumentClass::Integer]);
    }

    #[test]
    fn test_classify_i8() {
        assert_eq!(classify_type(&IrType::I8), vec![ArgumentClass::Integer]);
    }

    #[test]
    fn test_classify_i16() {
        assert_eq!(classify_type(&IrType::I16), vec![ArgumentClass::Integer]);
    }

    #[test]
    fn test_classify_f32() {
        assert_eq!(classify_type(&IrType::F32), vec![ArgumentClass::Sse]);
    }

    #[test]
    fn test_classify_f64() {
        assert_eq!(classify_type(&IrType::F64), vec![ArgumentClass::Sse]);
    }

    #[test]
    fn test_classify_pointer() {
        let ptr = IrType::Pointer(Box::new(IrType::I32));
        assert_eq!(classify_type(&ptr), vec![ArgumentClass::Integer]);
    }

    #[test]
    fn test_classify_void() {
        assert_eq!(classify_type(&IrType::Void), vec![ArgumentClass::NoClass]);
    }

    #[test]
    fn test_classify_small_struct() {
        // struct { i32, i32 } = 8 bytes -> single INTEGER eightbyte
        let s = IrType::Struct {
            fields: vec![IrType::I32, IrType::I32],
            packed: false,
        };
        assert_eq!(classify_type(&s), vec![ArgumentClass::Integer]);
    }

    #[test]
    fn test_classify_two_eightbyte_struct() {
        // struct { i64, i64 } = 16 bytes -> two INTEGER eightbytes
        let s = IrType::Struct {
            fields: vec![IrType::I64, IrType::I64],
            packed: false,
        };
        assert_eq!(
            classify_type(&s),
            vec![ArgumentClass::Integer, ArgumentClass::Integer]
        );
    }

    #[test]
    fn test_classify_large_struct() {
        // struct { i64, i64, i64 } = 24 bytes -> MEMORY
        let s = IrType::Struct {
            fields: vec![IrType::I64, IrType::I64, IrType::I64],
            packed: false,
        };
        assert_eq!(classify_type(&s), vec![ArgumentClass::Memory]);
    }

    #[test]
    fn test_classify_struct_with_float() {
        // struct { f64 } = 8 bytes -> SSE
        let s = IrType::Struct {
            fields: vec![IrType::F64],
            packed: false,
        };
        assert_eq!(classify_type(&s), vec![ArgumentClass::Sse]);
    }

    #[test]
    fn test_classify_struct_int_and_float() {
        // struct { i64, f64 } = 16 bytes -> INTEGER, SSE
        let s = IrType::Struct {
            fields: vec![IrType::I64, IrType::F64],
            packed: false,
        };
        assert_eq!(
            classify_type(&s),
            vec![ArgumentClass::Integer, ArgumentClass::Sse]
        );
    }

    #[test]
    fn test_classify_struct_float_and_int() {
        // struct { f64, i64 } = 16 bytes -> SSE, INTEGER
        let s = IrType::Struct {
            fields: vec![IrType::F64, IrType::I64],
            packed: false,
        };
        assert_eq!(
            classify_type(&s),
            vec![ArgumentClass::Sse, ArgumentClass::Integer]
        );
    }

    #[test]
    fn test_classify_empty_struct() {
        let s = IrType::Struct {
            fields: vec![],
            packed: false,
        };
        assert_eq!(classify_type(&s), vec![ArgumentClass::NoClass]);
    }

    // ----- Argument location tests -----

    #[test]
    fn test_int_args_in_registers() {
        let types = vec![IrType::I64; 6];
        let layout = compute_argument_locations(&types);
        assert_eq!(layout.int_regs_used, 6);
        assert_eq!(layout.sse_regs_used, 0);
        assert_eq!(layout.stack_space, 0);
        assert_eq!(layout.locations[0].register, Some(RDI));
        assert_eq!(layout.locations[1].register, Some(RSI));
        assert_eq!(layout.locations[2].register, Some(RDX));
        assert_eq!(layout.locations[3].register, Some(RCX));
        assert_eq!(layout.locations[4].register, Some(R8));
        assert_eq!(layout.locations[5].register, Some(R9));
    }

    #[test]
    fn test_float_args_in_registers() {
        let types = vec![IrType::F64; 8];
        let layout = compute_argument_locations(&types);
        assert_eq!(layout.sse_regs_used, 8);
        assert_eq!(layout.int_regs_used, 0);
        assert_eq!(layout.stack_space, 0);
        assert_eq!(layout.locations[0].register, Some(XMM0));
        assert_eq!(layout.locations[7].register, Some(XMM7));
    }

    #[test]
    fn test_overflow_to_stack() {
        // 7 integer args: first 6 in registers, 7th on stack
        let types = vec![IrType::I64; 7];
        let layout = compute_argument_locations(&types);
        assert_eq!(layout.int_regs_used, 6);
        assert!(layout.stack_space > 0);
        assert!(layout.locations[6].register.is_none());
        assert!(layout.locations[6].stack_offset.is_some());
    }

    #[test]
    fn test_mixed_int_float() {
        // i64, f64, i64, f64 -> rdi, xmm0, rsi, xmm1
        let types = vec![IrType::I64, IrType::F64, IrType::I64, IrType::F64];
        let layout = compute_argument_locations(&types);
        assert_eq!(layout.int_regs_used, 2);
        assert_eq!(layout.sse_regs_used, 2);
        assert_eq!(layout.locations[0].register, Some(RDI));
        assert_eq!(layout.locations[1].register, Some(XMM0));
        assert_eq!(layout.locations[2].register, Some(RSI));
        assert_eq!(layout.locations[3].register, Some(XMM1));
    }

    #[test]
    fn test_no_args() {
        let layout = compute_argument_locations(&[]);
        assert_eq!(layout.int_regs_used, 0);
        assert_eq!(layout.sse_regs_used, 0);
        assert_eq!(layout.stack_space, 0);
        assert!(layout.locations.is_empty());
    }

    #[test]
    fn test_memory_arg_uses_int_reg() {
        // A large struct (> 16 bytes) is MEMORY class and passed by
        // hidden pointer, consuming an integer register.
        let big = IrType::Struct {
            fields: vec![IrType::I64, IrType::I64, IrType::I64],
            packed: false,
        };
        let layout = compute_argument_locations(&[big]);
        assert_eq!(layout.int_regs_used, 1);
        assert_eq!(layout.locations[0].class, ArgumentClass::Memory);
        assert_eq!(layout.locations[0].register, Some(RDI));
    }

    // ----- Return value tests -----

    #[test]
    fn test_return_int() {
        let rc = classify_return_type(&IrType::I64);
        assert_eq!(rc, ReturnClass::Integer { regs: vec![RAX] });
    }

    #[test]
    fn test_return_i32() {
        let rc = classify_return_type(&IrType::I32);
        assert_eq!(rc, ReturnClass::Integer { regs: vec![RAX] });
    }

    #[test]
    fn test_return_pointer() {
        let rc = classify_return_type(&IrType::Pointer(Box::new(IrType::I8)));
        assert_eq!(rc, ReturnClass::Integer { regs: vec![RAX] });
    }

    #[test]
    fn test_return_float() {
        let rc = classify_return_type(&IrType::F64);
        assert_eq!(rc, ReturnClass::Sse { regs: vec![XMM0] });
    }

    #[test]
    fn test_return_f32() {
        let rc = classify_return_type(&IrType::F32);
        assert_eq!(rc, ReturnClass::Sse { regs: vec![XMM0] });
    }

    #[test]
    fn test_return_void() {
        let rc = classify_return_type(&IrType::Void);
        assert_eq!(rc, ReturnClass::Void);
    }

    #[test]
    fn test_return_large_struct() {
        let s = IrType::Struct {
            fields: vec![IrType::I64, IrType::I64, IrType::I64],
            packed: false,
        };
        assert_eq!(classify_return_type(&s), ReturnClass::Memory);
    }

    #[test]
    fn test_return_small_int_struct() {
        // struct { i64, i64 } -> rax, rdx
        let s = IrType::Struct {
            fields: vec![IrType::I64, IrType::I64],
            packed: false,
        };
        assert_eq!(
            classify_return_type(&s),
            ReturnClass::Integer {
                regs: vec![RAX, RDX]
            }
        );
    }

    #[test]
    fn test_return_small_sse_struct() {
        // struct { f64 } -> xmm0
        let s = IrType::Struct {
            fields: vec![IrType::F64],
            packed: false,
        };
        assert_eq!(
            classify_return_type(&s),
            ReturnClass::Sse { regs: vec![XMM0] }
        );
    }

    // ----- Stack frame tests -----

    #[test]
    fn test_stack_alignment() {
        assert_eq!(align_stack_size(0), 0);
        assert_eq!(align_stack_size(1), 16);
        assert_eq!(align_stack_size(15), 16);
        assert_eq!(align_stack_size(16), 16);
        assert_eq!(align_stack_size(17), 32);
        assert_eq!(align_stack_size(32), 32);
        assert_eq!(align_stack_size(33), 48);
    }

    #[test]
    fn test_compute_stack_frame_basic() {
        let frame = compute_stack_frame(32, 2, &[], true);
        // 32 locals + 16 spill bytes = 48
        assert!(frame.frame_size % 16 == 0);
        assert!(frame.uses_frame_pointer);
        assert!(!frame.use_red_zone);
    }

    #[test]
    fn test_red_zone_optimization() {
        // Red zone applies only to leaf functions with NO body locals
        // (body_size == 0), because our instruction selection always
        // uses RBP-relative addressing for allocas, which requires
        // a prologue (push rbp; mov rbp, rsp).
        let frame = compute_stack_frame(0, 0, &[], false);
        assert!(frame.use_red_zone);
        assert_eq!(frame.frame_size, 0);
        assert!(!frame.uses_frame_pointer);

        // body_size > 0 should disable red zone (needs RBP for allocas)
        let frame2 = compute_stack_frame(16, 0, &[], false);
        assert!(!frame2.use_red_zone);
        assert!(frame2.uses_frame_pointer);
    }

    #[test]
    fn test_no_red_zone_with_calls() {
        let frame = compute_stack_frame(16, 0, &[], true);
        assert!(!frame.use_red_zone);
        assert!(frame.uses_frame_pointer);
    }

    #[test]
    fn test_no_red_zone_with_callee_saved() {
        let frame = compute_stack_frame(16, 0, &[RBX], false);
        assert!(!frame.use_red_zone);
    }

    #[test]
    fn test_no_red_zone_large_frame() {
        // 256 bytes > 128 red zone limit
        let frame = compute_stack_frame(256, 0, &[], false);
        assert!(!frame.use_red_zone);
    }

    #[test]
    fn test_frame_alignment_with_callee_saves() {
        // With callee-saved registers, the frame_size must still produce
        // 16-byte alignment after pushes
        let frame = compute_stack_frame(24, 0, &[RBX, R12], true);
        assert!(frame.frame_size % 16 == 0 || frame.frame_size == 0);
        assert_eq!(frame.callee_saved_regs, vec![RBX, R12]);
    }

    #[test]
    fn test_frame_zero_body() {
        // No locals, no spills, but has calls (can't use red zone)
        let frame = compute_stack_frame(0, 0, &[], true);
        // frame_size may be 0 since body is 0 and alignment is satisfied
        assert!(!frame.use_red_zone);
    }

    // ----- Prologue / epilogue tests -----

    #[test]
    fn test_prologue_with_frame_pointer() {
        let frame = compute_stack_frame(64, 0, &[], true);
        let prologue = generate_prologue(&frame);
        assert!(!prologue.is_empty());
        // First instruction: push rbp
        assert_eq!(prologue[0].opcode, OP_PUSH);
        // Second instruction: mov rbp, rsp
        assert_eq!(prologue[1].opcode, OP_MOV_RR);
    }

    #[test]
    fn test_prologue_red_zone() {
        // Red zone requires body_size == 0 (no local variables) because
        // our isel uses RBP-relative addressing for locals.
        let frame = compute_stack_frame(0, 0, &[], false);
        assert!(frame.use_red_zone);
        let prologue = generate_prologue(&frame);
        assert!(prologue.is_empty());
    }

    #[test]
    fn test_prologue_with_callee_saved() {
        let frame = compute_stack_frame(32, 0, &[RBX, R12], true);
        let prologue = generate_prologue(&frame);
        // push rbp, mov rbp rsp, push rbx, push r12, sub rsp N
        assert!(prologue.len() >= 4);
        assert_eq!(prologue[0].opcode, OP_PUSH); // push rbp
        assert_eq!(prologue[1].opcode, OP_MOV_RR); // mov rbp, rsp
        assert_eq!(prologue[2].opcode, OP_PUSH); // push rbx
        assert_eq!(prologue[3].opcode, OP_PUSH); // push r12
    }

    #[test]
    fn test_epilogue_restores_callee_saved() {
        let frame = compute_stack_frame(32, 0, &[RBX, R12], true);
        let epilogue = generate_epilogue(&frame);
        assert!(!epilogue.is_empty());
        // Last instruction is ret
        let last = epilogue.last().unwrap();
        assert_eq!(last.opcode, OP_RET);
        // Second-to-last is pop rbp
        let second_last = &epilogue[epilogue.len() - 2];
        assert_eq!(second_last.opcode, OP_POP);
    }

    #[test]
    fn test_epilogue_red_zone() {
        // Red zone requires body_size == 0 (no local variables) because
        // our isel uses RBP-relative addressing for locals.
        let frame = compute_stack_frame(0, 0, &[], false);
        assert!(frame.use_red_zone);
        let epilogue = generate_epilogue(&frame);
        // Red zone epilogue: just ret
        assert_eq!(epilogue.len(), 1);
        assert_eq!(epilogue[0].opcode, OP_RET);
    }

    #[test]
    fn test_prologue_epilogue_symmetry() {
        let frame = compute_stack_frame(48, 4, &[RBX, R12, R13], true);
        let prologue = generate_prologue(&frame);
        let epilogue = generate_epilogue(&frame);

        // Count pushes in prologue (push rbp + callee saves)
        let push_count = prologue.iter().filter(|i| i.opcode == OP_PUSH).count();
        // Count pops in epilogue (callee saves + pop rbp)
        let pop_count = epilogue.iter().filter(|i| i.opcode == OP_POP).count();
        assert_eq!(push_count, pop_count);
    }

    // ----- RegisterInfo tests -----

    #[test]
    fn test_x86_64_register_info() {
        let info = x86_64_register_info();
        // 14 allocatable integer registers (no RSP, no RBP)
        assert_eq!(info.int_regs.len(), 14);
        // 16 XMM registers
        assert_eq!(info.float_regs.len(), 16);
        // 5 callee-saved integers
        assert_eq!(info.callee_saved_int.len(), 5);
        // 0 callee-saved floats
        assert_eq!(info.callee_saved_float.len(), 0);
        // 32 register names total (16 GPR + 16 XMM)
        assert_eq!(info.reg_names.len(), 32);
    }

    #[test]
    fn test_caller_saved_first_priority() {
        let info = x86_64_register_info();
        // Caller-saved registers should appear before callee-saved
        let first_callee_saved_pos = info
            .int_regs
            .iter()
            .position(|r| info.callee_saved_int.contains(r))
            .expect("should contain callee-saved registers");
        // All registers before the first callee-saved should be caller-saved
        for i in 0..first_callee_saved_pos {
            assert!(
                !info.callee_saved_int.contains(&info.int_regs[i]),
                "register at position {} should be caller-saved",
                i
            );
        }
        // The first callee-saved should be at index 9
        // (after 9 caller-saved: RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11)
        assert_eq!(first_callee_saved_pos, 9);
    }

    #[test]
    fn test_register_names() {
        let info = x86_64_register_info();
        assert_eq!(info.reg_names[&RAX], "rax");
        assert_eq!(info.reg_names[&RCX], "rcx");
        assert_eq!(info.reg_names[&RSP], "rsp");
        assert_eq!(info.reg_names[&RBP], "rbp");
        assert_eq!(info.reg_names[&R15], "r15");
        assert_eq!(info.reg_names[&XMM0], "xmm0");
        assert_eq!(info.reg_names[&XMM15], "xmm15");
    }

    #[test]
    fn test_rsp_rbp_not_allocatable() {
        let info = x86_64_register_info();
        assert!(!info.int_regs.contains(&RSP));
        assert!(!info.int_regs.contains(&RBP));
    }

    // ----- Call stack adjustment tests -----

    #[test]
    fn test_call_stack_adjustment_aligned() {
        // 0 + 0 + 8 = 8, need 8 padding to reach 16
        let adj = compute_call_stack_adjustment(0, 0);
        assert_eq!(adj, 8);
    }

    #[test]
    fn test_call_stack_adjustment_with_args() {
        // 0 + 16 + 8 = 24, need 8 padding to reach 32
        let adj = compute_call_stack_adjustment(0, 16);
        assert_eq!(adj, 8);
    }

    #[test]
    fn test_call_stack_adjustment_already_correct() {
        // 8 + 0 + 8 = 16, already aligned
        let adj = compute_call_stack_adjustment(8, 0);
        assert_eq!(adj, 0);
    }

    #[test]
    fn test_call_stack_adjustment_with_offset() {
        // 16 + 8 + 8 = 32, already aligned
        let adj = compute_call_stack_adjustment(16, 8);
        assert_eq!(adj, 0);
    }

    // ----- Aggregate classification tests -----

    #[test]
    fn test_classify_aggregate_empty() {
        let result = classify_aggregate(0, &[]);
        assert_eq!(result, vec![ArgumentClass::NoClass]);
    }

    #[test]
    fn test_classify_aggregate_single_int() {
        let fields = vec![(IrType::I32, 0u64)];
        let result = classify_aggregate(4, &fields);
        assert_eq!(result, vec![ArgumentClass::Integer]);
    }

    #[test]
    fn test_classify_aggregate_too_large() {
        let fields = vec![
            (IrType::I64, 0u64),
            (IrType::I64, 8u64),
            (IrType::I64, 16u64),
        ];
        let result = classify_aggregate(24, &fields);
        assert_eq!(result, vec![ArgumentClass::Memory]);
    }

    #[test]
    fn test_classify_aggregate_two_eightbytes() {
        let fields = vec![(IrType::I64, 0u64), (IrType::F64, 8u64)];
        let result = classify_aggregate(16, &fields);
        assert_eq!(result, vec![ArgumentClass::Integer, ArgumentClass::Sse]);
    }

    // ----- Type size helper tests -----

    #[test]
    fn test_x86_64_type_sizes() {
        assert_eq!(x86_64_type_size(&IrType::Void), 0);
        assert_eq!(x86_64_type_size(&IrType::I1), 1);
        assert_eq!(x86_64_type_size(&IrType::I8), 1);
        assert_eq!(x86_64_type_size(&IrType::I16), 2);
        assert_eq!(x86_64_type_size(&IrType::I32), 4);
        assert_eq!(x86_64_type_size(&IrType::I64), 8);
        assert_eq!(x86_64_type_size(&IrType::F32), 4);
        assert_eq!(x86_64_type_size(&IrType::F64), 8);
        assert_eq!(x86_64_type_size(&IrType::Pointer(Box::new(IrType::I32))), 8);
    }

    #[test]
    fn test_struct_size() {
        // struct { i32, i32 } = 4 + 4 = 8
        let s = IrType::Struct {
            fields: vec![IrType::I32, IrType::I32],
            packed: false,
        };
        assert_eq!(x86_64_type_size(&s), 8);

        // struct { i32, i64 } = 4 + 4(pad) + 8 = 16
        let s2 = IrType::Struct {
            fields: vec![IrType::I32, IrType::I64],
            packed: false,
        };
        assert_eq!(x86_64_type_size(&s2), 16);
    }

    #[test]
    fn test_packed_struct_size() {
        // packed struct { i32, i64 } = 4 + 8 = 12 (no padding)
        let s = IrType::Struct {
            fields: vec![IrType::I32, IrType::I64],
            packed: true,
        };
        assert_eq!(x86_64_type_size(&s), 12);
    }

    #[test]
    fn test_array_size() {
        let a = IrType::Array {
            element: Box::new(IrType::I32),
            count: 10,
        };
        assert_eq!(x86_64_type_size(&a), 40);
    }

    #[test]
    fn test_type_alignment() {
        assert_eq!(x86_64_type_alignment(&IrType::I8), 1);
        assert_eq!(x86_64_type_alignment(&IrType::I16), 2);
        assert_eq!(x86_64_type_alignment(&IrType::I32), 4);
        assert_eq!(x86_64_type_alignment(&IrType::I64), 8);
        assert_eq!(x86_64_type_alignment(&IrType::F32), 4);
        assert_eq!(x86_64_type_alignment(&IrType::F64), 8);
        assert_eq!(
            x86_64_type_alignment(&IrType::Pointer(Box::new(IrType::Void))),
            8
        );
    }

    #[test]
    fn test_struct_field_offsets() {
        // struct { i32, i64 } -> offsets: [0, 8] (i64 aligned to 8)
        let fields = vec![IrType::I32, IrType::I64];
        let offsets = x86_64_struct_field_offsets(&fields, false);
        assert_eq!(offsets, vec![0, 8]);
    }

    #[test]
    fn test_packed_struct_field_offsets() {
        // packed struct { i32, i64 } -> offsets: [0, 4] (no alignment padding)
        let fields = vec![IrType::I32, IrType::I64];
        let offsets = x86_64_struct_field_offsets(&fields, true);
        assert_eq!(offsets, vec![0, 4]);
    }

    // ----- Merge classes tests -----

    #[test]
    fn test_merge_classes() {
        assert_eq!(
            merge_classes(ArgumentClass::Integer, ArgumentClass::Integer),
            ArgumentClass::Integer
        );
        assert_eq!(
            merge_classes(ArgumentClass::Sse, ArgumentClass::Sse),
            ArgumentClass::Sse
        );
        assert_eq!(
            merge_classes(ArgumentClass::NoClass, ArgumentClass::Integer),
            ArgumentClass::Integer
        );
        assert_eq!(
            merge_classes(ArgumentClass::Integer, ArgumentClass::NoClass),
            ArgumentClass::Integer
        );
        assert_eq!(
            merge_classes(ArgumentClass::Memory, ArgumentClass::Integer),
            ArgumentClass::Memory
        );
        assert_eq!(
            merge_classes(ArgumentClass::Integer, ArgumentClass::Sse),
            ArgumentClass::Integer
        );
        assert_eq!(
            merge_classes(ArgumentClass::X87, ArgumentClass::Integer),
            ArgumentClass::Memory
        );
    }

    // ----- Classification constants array tests -----

    #[test]
    fn test_int_arg_regs_order() {
        assert_eq!(INT_ARG_REGS, [RDI, RSI, RDX, RCX, R8, R9]);
    }

    #[test]
    fn test_float_arg_regs_order() {
        assert_eq!(
            FLOAT_ARG_REGS,
            [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7]
        );
    }

    #[test]
    fn test_callee_saved_gprs() {
        assert_eq!(CALLEE_SAVED_GPRS, [RBX, R12, R13, R14, R15]);
    }

    #[test]
    fn test_caller_saved_gprs() {
        assert_eq!(
            CALLER_SAVED_GPRS,
            [RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11]
        );
    }

    #[test]
    fn test_int_return_regs() {
        assert_eq!(INT_RETURN_REGS, [RAX, RDX]);
    }

    #[test]
    fn test_float_return_regs() {
        assert_eq!(FLOAT_RETURN_REGS, [XMM0, XMM1]);
    }
}
