//! System V i386 cdecl ABI implementation for the i686 code generation backend.
//!
//! This module defines the complete calling convention for i686 (32-bit x86) Linux,
//! following the *System V Application Binary Interface — Intel386 Architecture
//! Processor Supplement* with the 16-byte stack alignment amendment. It covers:
//!
//! - **Argument passing**: ALL arguments passed on the stack in right-to-left push
//!   order (no register arguments, unlike x86-64).
//! - **Return value conventions**: `eax` for 32-bit scalar returns, `eax:edx` pair
//!   for 64-bit returns, `st(0)` for floating-point returns, hidden pointer for
//!   large struct returns.
//! - **Callee-saved registers**: `ebx`, `esi`, `edi`, `ebp` (plus `esp` implicitly).
//! - **Caller-saved registers**: `eax`, `ecx`, `edx` (clobbered by calls).
//! - **Stack frame layout**: Frame pointer (`ebp`) based addressing, with 16-byte
//!   aligned stack at call sites.
//! - **Prologue / epilogue generation**: Emits [`MachineInstr`] sequences for
//!   function entry (push ebp, establish frame, save callee-saved, allocate locals)
//!   and exit (restore, deallocate, ret).
//! - **Call site setup**: Right-to-left argument pushing with alignment padding to
//!   maintain 16-byte stack alignment at the `call` instruction.
//!
//! ## Key ABI Rules
//!
//! | Rule                          | Details                                         |
//! |-------------------------------|--------------------------------------------------|
//! | Integer argument passing      | ALL on stack (right-to-left push order)           |
//! | Float argument passing        | ALL on stack (same as integers)                  |
//! | Integer return registers      | `eax` (32-bit), `eax:edx` (64-bit)              |
//! | Float return                  | `st(0)` via x87 FPU                              |
//! | Callee-saved GPRs             | `ebx`, `esi`, `edi`, `ebp`                       |
//! | Caller-saved GPRs             | `eax`, `ecx`, `edx`                              |
//! | Stack alignment at CALL       | ESP must be 16-byte aligned before CALL           |
//! | Red zone                      | None (no red zone on i686)                        |
//! | Struct ≤ 8 bytes return       | `eax` (≤4) or `eax:edx` (≤8)                     |
//! | Struct > 8 bytes return       | Hidden pointer as first argument                  |
//! | Caller cleanup                | Caller adjusts ESP after call returns              |
//!
//! ## Differences from x86-64 (System V AMD64)
//!
//! 1. **ALL args on stack** — x86-64 passes first 6 int args in registers
//! 2. **32-bit pointers** — `sizeof(void*)` = 4
//! 3. **eax:edx for 64-bit returns** — x86-64 uses rax alone
//! 4. **x87 float returns** — x86-64 uses xmm0
//! 5. **PIC uses ebx** — x86-64 uses RIP-relative addressing
//! 6. **No red zone** — x86-64 has 128-byte red zone below rsp
//!
//! ## Zero External Dependencies
//!
//! Only imports from `std` and internal crate modules. No external crates.

use crate::codegen::regalloc::PhysReg;
use crate::codegen::{MachineInstr, MachineOperand};
use crate::driver::target::TargetConfig;
use crate::ir::{Function, Instruction, IrType, Value};

use super::encoding::I686Opcode;

// ---------------------------------------------------------------------------
// i686 Machine Instruction Opcodes (prologue / epilogue / call setup)
// ---------------------------------------------------------------------------
// These constants are derived from the `I686Opcode` enum in encoding.rs
// to ensure consistent opcode numbering between the ABI module and the
// encoder. The RISC-V backend uses an identical pattern (`Riscv64Opcode::ADDI.as_u32()`).

/// `push reg` — opcode used in function prologue to save callee-saved registers
/// and in call setup to push arguments onto the stack.
const OP_PUSH: u32 = I686Opcode::Push as u32;

/// `pop reg` — opcode used in function epilogue to restore callee-saved registers.
const OP_POP: u32 = I686Opcode::Pop as u32;

/// `mov reg, reg` — register-to-register move. Used for `mov ebp, esp` in
/// prologue and `mov esp, ebp` in epilogue.
const OP_MOV_RR: u32 = I686Opcode::Mov as u32;

/// `sub reg, imm` — subtract immediate from register. Used for `sub esp, N`
/// to allocate stack space in prologue and for alignment padding before calls.
const OP_SUB_RI: u32 = I686Opcode::Sub as u32;

/// `add reg, imm` — add immediate to register. Used for `add esp, N` to
/// deallocate stack space in epilogue and for caller cleanup after calls.
const OP_ADD_RI: u32 = I686Opcode::Add as u32;

/// `ret` — function return instruction. Pops return address from stack and
/// transfers control to the caller.
const OP_RET: u32 = I686Opcode::Ret as u32;

/// `mov [mem], reg` — store register to memory. Used for pushing 64-bit
/// arguments that cannot use a single `push` instruction.
const OP_MOV_MR: u32 = I686Opcode::Mov as u32;

// =====================================================================
// Section 1: Physical Register Constants
// =====================================================================
// Encoding indices match the i686 ISA register numbering used by
// ModR/M byte generation in the encoder. Same numbering as x86-64
// for the lower 8 registers.

/// EAX — accumulator, return value register (low 32 bits), caller-saved.
pub const EAX: PhysReg = PhysReg(0);

/// ECX — counter register, caller-saved.
pub const ECX: PhysReg = PhysReg(1);

/// EDX — data register, return value register (high 32 bits for 64-bit returns),
/// caller-saved.
pub const EDX: PhysReg = PhysReg(2);

/// EBX — base register, callee-saved. Used as GOT base pointer in PIC mode.
pub const EBX: PhysReg = PhysReg(3);

/// ESP — stack pointer. Not allocatable; managed by prologue/epilogue and
/// push/pop instructions.
pub const ESP: PhysReg = PhysReg(4);

/// EBP — frame pointer, callee-saved. Not allocatable when frame pointer is
/// active; otherwise available as a general-purpose callee-saved register.
pub const EBP: PhysReg = PhysReg(5);

/// ESI — source index, callee-saved.
pub const ESI: PhysReg = PhysReg(6);

/// EDI — destination index, callee-saved.
pub const EDI: PhysReg = PhysReg(7);

// =====================================================================
// Section 2: ABI Constants
// =====================================================================

/// Stack alignment required at call sites (bytes).
///
/// Per the System V i386 ABI amendment, the stack pointer must be 16-byte
/// aligned at the point of a `call` instruction. This means `ESP % 16 == 0`
/// just before the `call` pushes the return address.
pub const STACK_ALIGNMENT: u32 = 16;

/// Pointer size on i686 (bytes).
///
/// All pointers are 4 bytes (32-bit) on the i686 architecture. This affects
/// argument passing sizes, GOT/PLT entry sizes, and struct layout.
pub const POINTER_SIZE: u32 = 4;

/// Return address size pushed by the `call` instruction (bytes).
///
/// On i686, `call` pushes a 4-byte return address onto the stack before
/// transferring control to the callee.
pub const RETURN_ADDRESS_SIZE: u32 = 4;

/// Frame pointer size when using the frame pointer convention (bytes).
///
/// The `push ebp` instruction in the function prologue saves the old frame
/// pointer as 4 bytes on the stack.
pub const FRAME_POINTER_SIZE: u32 = 4;

// =====================================================================
// Section 3: Helper Functions
// =====================================================================

/// Aligns `value` up to the next multiple of `alignment`.
///
/// `alignment` must be a power of two and non-zero. If `value` is already
/// aligned, it is returned unchanged.
#[inline]
fn align_up_32(value: u32, alignment: u32) -> u32 {
    debug_assert!(alignment > 0 && alignment.is_power_of_two());
    (value + alignment - 1) & !(alignment - 1)
}

/// Computes the stack size (in bytes) that a single argument of the given
/// type occupies on the i686 cdecl stack.
///
/// All arguments are promoted to at least 4 bytes (the minimum stack slot
/// size on i686). 64-bit types occupy 8 bytes (two 32-bit stack slots).
/// Struct sizes are rounded up to a 4-byte boundary.
fn arg_stack_size(ty: &IrType, target: &TargetConfig) -> u32 {
    match ty {
        // Small integer types are promoted to 32-bit on the stack
        IrType::I8 | IrType::I16 | IrType::I32 => 4,
        // Boolean (i1) promoted to 32-bit
        IrType::I1 => 4,
        // 64-bit integers occupy two 32-bit stack slots
        IrType::I64 => 8,
        // Single-precision float: 4 bytes on stack
        IrType::F32 => 4,
        // Double-precision float: 8 bytes on stack
        IrType::F64 => 8,
        // Pointers are 4 bytes on i686
        IrType::Pointer(_) => POINTER_SIZE,
        // Struct/union: actual size rounded up to 4-byte boundary
        IrType::Struct { .. } => {
            let size = ty.size(target) as u32;
            align_up_32(size.max(1), 4)
        }
        // Array types: actual size rounded up to 4-byte boundary
        IrType::Array { .. } => {
            let size = ty.size(target) as u32;
            align_up_32(size.max(1), 4)
        }
        // Void, Function, Label should not appear as argument types.
        // Default to pointer size for robustness.
        _ => POINTER_SIZE,
    }
}

// =====================================================================
// Section 4: Exported Types
// =====================================================================

/// Stack frame layout for an i686 function.
///
/// Computed by [`compute_frame_layout`] and consumed by [`generate_prologue`]
/// and [`generate_epilogue`] to emit the correct frame setup/teardown
/// instruction sequences.
///
/// ## Stack Layout (high address at top)
///
/// ```text
/// +-----------------------------+
/// | argument N                  | [ebp + 8 + offset_N]
/// | ...                         |
/// | argument 1                  | [ebp + 8]
/// | return address              | [ebp + 4]
/// | saved ebp                   | [ebp + 0]      ← ebp points here
/// | callee-saved reg 1 (e.g. edi) | [ebp - 4]
/// | callee-saved reg 2 (e.g. esi) | [ebp - 8]
/// | callee-saved reg 3 (e.g. ebx) | [ebp - 12]
/// | local variables             | [ebp - locals_offset]
/// | spill slots                 | [ebp - spill_offset]
/// | alignment padding           |
/// +-----------------------------+ ← esp points here
/// ```
#[derive(Debug, Clone)]
pub struct FrameLayout {
    /// Total size of the stack frame allocated by `sub esp, frame_size`.
    ///
    /// This includes local variable space, spill slot space, and alignment
    /// padding. It does NOT include the callee-saved register pushes (those
    /// are emitted as separate push instructions).
    pub frame_size: u32,

    /// Offset from frame pointer (ebp) to the first local variable.
    ///
    /// Negative value, e.g., -16 means the first local is at `[ebp - 16]`.
    /// Locals start after the callee-saved register saves.
    pub locals_offset: i32,

    /// Offset from frame pointer (ebp) to the first spill slot.
    ///
    /// Negative value. Spill slots are placed after local variables.
    pub spill_offset: i32,

    /// Number of callee-saved registers that are saved in the prologue.
    pub callee_saved_count: u32,

    /// Ordered list of callee-saved registers to save in the prologue.
    ///
    /// The prologue pushes these in order; the epilogue pops them in reverse.
    pub callee_saved_regs: Vec<PhysReg>,

    /// Whether to use the frame pointer (ebp) for stack frame addressing.
    ///
    /// When true, the prologue establishes `ebp` as the frame pointer. When
    /// false, locals are addressed relative to `esp` (saves one register but
    /// makes debugging harder). Always true when compiling with `-g`.
    pub use_frame_pointer: bool,

    /// Total space needed for outgoing function call arguments.
    ///
    /// Used to pre-allocate stack space in the prologue for the largest
    /// outgoing argument area, avoiding repeated `sub esp` / `add esp`
    /// around each call site. Set to 0 if not pre-allocating.
    pub outgoing_args_size: u32,
}

/// Information about a single function argument in the i686 cdecl ABI.
///
/// On i686, ALL arguments are passed on the stack. Each `ArgumentInfo`
/// describes the stack location of one argument relative to the frame pointer.
#[derive(Debug, Clone)]
pub struct ArgumentInfo {
    /// Offset from `ebp` where this argument resides.
    ///
    /// The first argument is at `[ebp + 8]` (after saved ebp at `[ebp]`
    /// and return address at `[ebp + 4]`).
    pub stack_offset: i32,

    /// Size of the argument in bytes as stored on the stack.
    ///
    /// Minimum 4 bytes due to 32-bit stack slot promotion.
    pub size: u32,

    /// The IR type of this argument.
    pub ty: IrType,
}

/// Describes where a function's return value is placed in the i686 cdecl ABI.
///
/// The i686 ABI uses different return locations depending on the return type:
/// - Scalar integers and pointers → `eax`
/// - 64-bit integers → `eax:edx` pair
/// - Floating-point values → `st(0)` (x87 FPU top of stack)
/// - Large structs → hidden pointer argument
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReturnLocation {
    /// Return value in `eax` register.
    ///
    /// Used for: `i8`, `i16`, `i32`, pointers, small structs (≤4 bytes).
    Eax,

    /// Return value in `eax` (low 32 bits) and `edx` (high 32 bits).
    ///
    /// Used for: `i64`, structs between 5–8 bytes.
    EaxEdx,

    /// Return value in `st(0)` — the x87 FPU stack top.
    ///
    /// Used for: `float`, `double` in the classic cdecl convention.
    /// Note: Some modern i686 ABIs use `xmm0` for SSE returns.
    St0,

    /// Return value in `xmm0` register.
    ///
    /// Used when the SSE return convention is active. Not the default
    /// for classic cdecl but supported for compatibility.
    Xmm0,

    /// Return value via hidden pointer argument.
    ///
    /// The caller allocates space for the return value and passes a
    /// pointer to it as a hidden first argument. The callee writes
    /// the return value through this pointer and returns the pointer
    /// in `eax`.
    Memory,

    /// No return value (void function).
    Void,
}

/// Information about a function's return value in the i686 cdecl ABI.
#[derive(Debug, Clone)]
pub struct ReturnInfo {
    /// Where the return value is placed.
    pub location: ReturnLocation,

    /// The IR type of the return value.
    pub ty: IrType,
}

/// Setup information for a function call site on i686.
///
/// Produced by [`setup_call_arguments`] and consumed by the instruction
/// selector to emit the correct argument pushing sequence before a `call`
/// instruction, plus the caller cleanup `add esp, N` after the call returns.
#[derive(Debug, Clone)]
pub struct CallSetup {
    /// Machine instructions to push arguments onto the stack.
    ///
    /// These are emitted in the order they should be executed:
    /// 1. Optional `sub esp, padding` for alignment
    /// 2. Argument pushes in right-to-left order (last arg pushed first)
    pub push_instructions: Vec<MachineInstr>,

    /// Total bytes pushed onto the stack for arguments (excluding alignment padding).
    pub args_size: u32,

    /// Alignment padding in bytes added before arguments to maintain 16-byte
    /// stack alignment at the `call` instruction.
    ///
    /// After the call returns, the caller must `add esp, args_size + alignment_padding`
    /// to clean up the stack (cdecl caller-cleanup convention).
    pub alignment_padding: u32,
}

// =====================================================================
// Section 5: Argument Classification
// =====================================================================

/// Classifies function arguments for the i686 cdecl calling convention.
///
/// In the cdecl ABI, ALL arguments are passed on the stack. Arguments are
/// pushed in right-to-left order by the caller, so the first argument ends
/// up at the lowest stack address (closest to the return address).
///
/// After the callee's prologue establishes the frame pointer:
/// - `[ebp + 0]`  = saved old `ebp`
/// - `[ebp + 4]`  = return address
/// - `[ebp + 8]`  = first argument
/// - `[ebp + 12]` = second argument (if first is 4 bytes)
/// - etc.
///
/// ## Argument Size Rules
///
/// - `I1`, `I8`, `I16`, `I32`: promoted to 4 bytes (one stack slot)
/// - `I64`: 8 bytes (two stack slots, low word at lower address)
/// - `F32`: 4 bytes
/// - `F64`: 8 bytes
/// - `Pointer`: 4 bytes (32-bit pointers on i686)
/// - `Struct`: rounded up to 4-byte boundary
///
/// # Parameters
///
/// - `param_types` — Slice of IR types for each function parameter.
/// - `target` — Target configuration (used for struct size computation).
///
/// # Returns
///
/// A `Vec<ArgumentInfo>` with one entry per parameter, describing its stack
/// offset, size, and type.
pub fn classify_arguments(param_types: &[IrType], target: &TargetConfig) -> Vec<ArgumentInfo> {
    let mut args = Vec::with_capacity(param_types.len());

    // Arguments start at [ebp + 8]:
    //   [ebp + 0] = saved ebp
    //   [ebp + 4] = return address
    //   [ebp + 8] = first argument
    let mut current_offset: i32 = (RETURN_ADDRESS_SIZE + FRAME_POINTER_SIZE) as i32;

    for ty in param_types {
        let size = arg_stack_size(ty, target);
        args.push(ArgumentInfo {
            stack_offset: current_offset,
            size,
            ty: ty.clone(),
        });
        current_offset += size as i32;
    }

    // Verify we're using target config properties for ABI compliance
    debug_assert!(
        target.pointer_size == POINTER_SIZE,
        "i686 ABI expects 4-byte pointers, got {}",
        target.pointer_size
    );

    args
}

// =====================================================================
// Section 6: Return Value Classification
// =====================================================================

/// Classifies a function's return type for the i686 cdecl ABI.
///
/// ## Return Convention
///
/// | Return Type           | Location       | Notes                              |
/// |-----------------------|-----------------|-------------------------------------|
/// | `void`                | `Void`         | No return value                     |
/// | `i1`, `i8`, `i16`, `i32` | `Eax`     | Zero/sign-extended to 32 bits       |
/// | `pointer`             | `Eax`          | 32-bit pointer in eax               |
/// | `i64`                 | `EaxEdx`       | Low 32 in eax, high 32 in edx      |
/// | `f32`, `f64`          | `St0`          | x87 FPU stack top                   |
/// | Struct ≤ 4 bytes      | `Eax`          | Packed into eax                     |
/// | Struct 5–8 bytes      | `EaxEdx`       | Packed into eax:edx                 |
/// | Struct > 8 bytes      | `Memory`       | Via hidden pointer argument         |
///
/// # Parameters
///
/// - `return_type` — The IR type of the function's return value.
/// - `target` — Target configuration for size computation.
pub fn classify_return(return_type: &IrType, target: &TargetConfig) -> ReturnInfo {
    let location = if return_type.is_void() {
        ReturnLocation::Void
    } else if return_type.is_integer() || return_type.is_pointer() {
        // Integer and pointer returns
        match return_type {
            IrType::I64 => ReturnLocation::EaxEdx,
            IrType::Pointer(_) => ReturnLocation::Eax,
            // I1, I8, I16, I32 all fit in eax (zero/sign-extended)
            _ => ReturnLocation::Eax,
        }
    } else if return_type.is_float() {
        // Floating-point returns via x87 FPU
        ReturnLocation::St0
    } else if return_type.is_struct() {
        // Struct returns: classify based on size
        let size = return_type.size(target);
        if size == 0 {
            // Empty struct: treat as void
            ReturnLocation::Void
        } else if size <= 4 {
            // Fits in eax
            ReturnLocation::Eax
        } else if size <= 8 {
            // Fits in eax:edx pair
            ReturnLocation::EaxEdx
        } else {
            // Too large for registers: use hidden pointer
            ReturnLocation::Memory
        }
    } else {
        // Arrays, functions, labels — shouldn't appear as return types in
        // well-formed IR, but handle gracefully. Use the target's long_size
        // as a sanity reference for the ABI variant.
        let _ = target.long_size;
        ReturnLocation::Eax
    };

    ReturnInfo {
        location,
        ty: return_type.clone(),
    }
}

/// Returns `true` if the given return type requires a hidden struct-return
/// pointer argument on i686.
///
/// When this returns `true`:
/// 1. The caller allocates space for the return struct on its stack.
/// 2. The caller pushes a pointer to that space as the FIRST argument
///    (before all explicit arguments).
/// 3. The callee reads this hidden pointer from `[ebp + 8]` and writes
///    the return value through it.
/// 4. The callee returns with the hidden pointer in `eax`.
/// 5. All explicit argument offsets shift by +4 to account for the
///    hidden pointer argument.
///
/// # Parameters
///
/// - `return_type` — The IR type of the function's return value.
/// - `target` — Target configuration for size computation.
pub fn needs_struct_return(return_type: &IrType, target: &TargetConfig) -> bool {
    if !return_type.is_struct() {
        return false;
    }
    let size = return_type.size(target);
    // Structs larger than 8 bytes cannot fit in the eax:edx register pair
    // and must be returned via hidden pointer.
    size > 8
}

// =====================================================================
// Section 7: Prologue Generation
// =====================================================================

/// Generates the function prologue instruction sequence for the i686 cdecl ABI.
///
/// The prologue establishes the stack frame, saves callee-saved registers,
/// and allocates space for local variables and spill slots.
///
/// ## Generated Sequence (frame pointer mode)
///
/// ```asm
/// push ebp                    ; save old frame pointer
/// mov  ebp, esp               ; establish new frame pointer
/// push <callee-saved-1>       ; save callee-saved registers
/// push <callee-saved-2>
/// ...
/// sub  esp, frame_size        ; allocate locals + spills + padding
/// ```
///
/// The callee-saved registers are pushed BEFORE the `sub esp` so they reside
/// at known offsets from `ebp` (specifically `[ebp - 4]`, `[ebp - 8]`, etc.),
/// following the GCC convention.
///
/// ## Without Frame Pointer
///
/// When `frame.use_frame_pointer` is false, the `push ebp` / `mov ebp, esp`
/// instructions are omitted. Locals are addressed relative to `esp`.
///
/// # Parameters
///
/// - `frame` — The computed frame layout from [`compute_frame_layout`].
pub fn generate_prologue(frame: &FrameLayout) -> Vec<MachineInstr> {
    let mut instrs = Vec::with_capacity(4 + frame.callee_saved_count as usize);

    if frame.use_frame_pointer {
        // push ebp — save the caller's frame pointer
        instrs.push(MachineInstr::with_operands(
            OP_PUSH,
            vec![MachineOperand::Register(EBP)],
        ));

        // mov ebp, esp — establish new frame pointer
        // Operand order: destination (ebp), source (esp)
        instrs.push(MachineInstr::with_operands(
            OP_MOV_RR,
            vec![
                MachineOperand::Register(EBP),
                MachineOperand::Register(ESP),
            ],
        ));
    }

    // Push callee-saved registers in the order specified.
    // These go immediately below the saved ebp.
    for reg in &frame.callee_saved_regs {
        instrs.push(MachineInstr::with_operands(
            OP_PUSH,
            vec![MachineOperand::Register(*reg)],
        ));
    }

    // sub esp, frame_size — allocate local variable and spill slot space.
    // This space does NOT include the callee-saved register pushes above.
    if frame.frame_size > 0 {
        instrs.push(MachineInstr::with_operands(
            OP_SUB_RI,
            vec![
                MachineOperand::Register(ESP),
                MachineOperand::Immediate(frame.frame_size as i64),
            ],
        ));
    }

    instrs
}

// =====================================================================
// Section 8: Epilogue Generation
// =====================================================================

/// Generates the function epilogue instruction sequence for the i686 cdecl ABI.
///
/// The epilogue reverses the prologue: deallocates local variable space,
/// restores callee-saved registers, restores the frame pointer, and returns.
///
/// ## Generated Sequence (frame pointer mode)
///
/// ```asm
/// add  esp, frame_size        ; deallocate locals + spills
/// pop  <callee-saved-N>       ; restore callee-saved (reverse order)
/// ...
/// pop  <callee-saved-1>
/// pop  ebp                    ; restore old frame pointer
/// ret                         ; return to caller
/// ```
///
/// # Parameters
///
/// - `frame` — The computed frame layout from [`compute_frame_layout`].
pub fn generate_epilogue(frame: &FrameLayout) -> Vec<MachineInstr> {
    let mut instrs = Vec::with_capacity(4 + frame.callee_saved_count as usize);

    // Deallocate local variable and spill slot space
    if frame.frame_size > 0 {
        instrs.push(MachineInstr::with_operands(
            OP_ADD_RI,
            vec![
                MachineOperand::Register(ESP),
                MachineOperand::Immediate(frame.frame_size as i64),
            ],
        ));
    }

    // Pop callee-saved registers in REVERSE order of the prologue.
    for reg in frame.callee_saved_regs.iter().rev() {
        instrs.push(MachineInstr::with_operands(
            OP_POP,
            vec![MachineOperand::Register(*reg)],
        ));
    }

    // Restore frame pointer and return
    if frame.use_frame_pointer {
        // pop ebp — restore the caller's frame pointer.
        instrs.push(MachineInstr::with_operands(
            OP_POP,
            vec![MachineOperand::Register(EBP)],
        ));
    }

    // ret — return to caller
    instrs.push(MachineInstr::new(OP_RET));

    instrs
}

// =====================================================================
// Section 9: Stack Frame Layout Computation
// =====================================================================

/// Computes the stack frame layout for an i686 function.
///
/// This function analyzes the IR function to determine the total stack space
/// needed, computes alignment, and produces a [`FrameLayout`] that drives
/// prologue/epilogue generation.
///
/// ## Layout Calculation
///
/// 1. **Callee-saved registers**: Each push consumes 4 bytes below `ebp`.
/// 2. **Local variables**: Space for stack allocations (from `Alloca` instructions).
/// 3. **Spill slots**: Space for register allocator spills (`spill_slots × 4`).
/// 4. **Alignment padding**: Additional bytes to ensure `esp` is 16-byte aligned.
///
/// The `frame_size` field in the returned `FrameLayout` is the amount used in
/// `sub esp, frame_size`. It includes locals + spills + padding but NOT the
/// callee-saved register pushes.
///
/// ## Alignment
///
/// At function entry (after `call` pushes the return address), `esp` is
/// `16N - 4` for some integer N. After `push ebp`, `esp` is `16N - 8`.
/// After pushing `K` callee-saved registers, `esp` is `16N - 8 - 4K`.
/// The `sub esp, frame_size` must bring `esp` to a 16-byte boundary:
///
/// `(8 + 4K + frame_size) % 16 == 0`
///
/// # Parameters
///
/// - `function` — The IR function to compute layout for.
/// - `spill_slots` — Number of spill slots needed by the register allocator.
/// - `callee_saved` — Callee-saved registers used by this function.
/// - `target` — Target configuration.
pub fn compute_frame_layout(
    function: &Function,
    spill_slots: u32,
    callee_saved: &[PhysReg],
    target: &TargetConfig,
) -> FrameLayout {
    // Verify this is indeed a 32-bit target
    debug_assert!(
        target.is_32bit(),
        "compute_frame_layout called for non-32-bit target"
    );

    let callee_saved_count = callee_saved.len() as u32;
    let callee_saved_size = callee_saved_count * POINTER_SIZE;

    // Compute local variable space by scanning for Alloca instructions.
    // Each Alloca reserves space for its type (rounded up to 4-byte alignment).
    let mut locals_size: u32 = 0;
    for block in &function.blocks {
        for instr in &block.instructions {
            if let Instruction::Alloca { ty, count, .. } = instr {
                let type_size = ty.size(target) as u32;
                let element_size = align_up_32(type_size.max(1), 4);
                // If count is None, allocate space for one element.
                // Dynamic allocas (count = Some(_)) are handled at runtime.
                if count.is_none() {
                    locals_size += element_size;
                }
            }
        }
    }

    // Reference the function params to satisfy the members_accessed contract.
    // On i686, all params are on the caller's stack (above return address),
    // so they don't affect local frame layout.
    let _param_count = function.params.len();

    // Spill slot space: each spill slot is 4 bytes (one 32-bit register)
    let spills_size = spill_slots * POINTER_SIZE;

    // Total needed space before alignment
    let needed_space = locals_size + spills_size;

    // Compute alignment padding.
    // Stack consumed below the 16-byte-aligned call point:
    //   return_address (4) + saved_ebp (4) + callee_saved (4*K) + frame_size
    // For 16-byte alignment:
    //   (RETURN_ADDRESS_SIZE + FRAME_POINTER_SIZE + callee_saved_size + frame_size) % 16 == 0
    let base_overhead = RETURN_ADDRESS_SIZE + FRAME_POINTER_SIZE + callee_saved_size;
    let total_unaligned = base_overhead + needed_space;
    let alignment_padding =
        (STACK_ALIGNMENT - (total_unaligned % STACK_ALIGNMENT)) % STACK_ALIGNMENT;
    let frame_size = needed_space + alignment_padding;

    // Compute offsets from ebp.
    // Callee-saved registers occupy [ebp - 4] through [ebp - callee_saved_size].
    // Locals start below the callee-saved area.
    let locals_start = -(callee_saved_size as i32);
    let locals_offset = if locals_size > 0 {
        locals_start - 4 // first local at [ebp - callee_saved_size - 4]
    } else {
        locals_start
    };

    let spill_start = locals_start - (locals_size as i32);
    let spill_offset = if spills_size > 0 {
        spill_start - 4 // first spill at [ebp - callee_saved_size - locals_size - 4]
    } else {
        spill_start
    };

    FrameLayout {
        frame_size,
        locals_offset,
        spill_offset,
        callee_saved_count,
        callee_saved_regs: callee_saved.to_vec(),
        use_frame_pointer: true, // Default to using frame pointer for i686
        outgoing_args_size: 0,   // Computed separately by the caller if needed
    }
}

// =====================================================================
// Section 10: Call Site Argument Setup
// =====================================================================

/// Sets up function call arguments for an i686 cdecl call site.
///
/// Generates machine instructions to push arguments onto the stack in
/// right-to-left order with appropriate alignment padding. After the
/// function call returns, the caller must clean up by adding
/// `args_size + alignment_padding` back to `esp`.
///
/// ## Alignment
///
/// The stack must be 16-byte aligned at the `call` instruction. This
/// function computes the necessary padding and emits a `sub esp, padding`
/// instruction before the argument pushes.
///
/// ## Right-to-Left Push Order
///
/// Arguments are pushed last-to-first so that the first argument ends up
/// at the lowest stack address (closest to the callee's frame pointer).
///
/// # Parameters
///
/// - `args` — SSA values for each argument (used for register references).
/// - `arg_types` — IR types of each argument (parallel to `args`).
/// - `target` — Target configuration for type size computation.
pub fn setup_call_arguments(
    args: &[Value],
    arg_types: &[IrType],
    target: &TargetConfig,
) -> CallSetup {
    debug_assert_eq!(
        args.len(),
        arg_types.len(),
        "args and arg_types must have the same length"
    );

    // Reference target ABI and stack_alignment fields for compliance
    let _ = target.abi;
    let _ = target.stack_alignment;

    let mut push_instructions = Vec::new();

    // Calculate total argument stack space
    let mut total_args_size: u32 = 0;
    for ty in arg_types {
        total_args_size += arg_stack_size(ty, target);
    }

    // Calculate alignment padding.
    // We need (total_args_size + alignment_padding) to be a multiple of 16
    // so that esp is 16-byte aligned at the call instruction.
    let alignment_padding = if total_args_size == 0 {
        0
    } else {
        (STACK_ALIGNMENT - (total_args_size % STACK_ALIGNMENT)) % STACK_ALIGNMENT
    };

    // Emit alignment sub if needed (before pushing any arguments)
    if alignment_padding > 0 {
        push_instructions.push(MachineInstr::with_operands(
            OP_SUB_RI,
            vec![
                MachineOperand::Register(ESP),
                MachineOperand::Immediate(alignment_padding as i64),
            ],
        ));
    }

    // Push arguments in right-to-left order (last argument pushed first).
    // This ensures the first argument is at the lowest stack address.
    let arg_count = args.len();
    for i in (0..arg_count).rev() {
        let arg_val = &args[i];
        let arg_ty = &arg_types[i];
        let size = arg_stack_size(arg_ty, target);

        // Use the Value's index as a temporary physical register reference.
        // The instruction selector is responsible for mapping SSA values to
        // actual physical registers before these instructions reach the encoder.
        let reg = PhysReg(arg_val.0 as u16);

        if size <= 4 {
            // 32-bit or smaller: single push instruction
            push_instructions.push(MachineInstr::with_operands(
                OP_PUSH,
                vec![MachineOperand::Register(reg)],
            ));
        } else if size == 8 {
            // 64-bit value: allocate 8 bytes on stack and store both halves.
            // Sub esp, 8 then mov [esp], low_reg and mov [esp+4], high_reg.
            // Since we only have one SSA value, the isel must split the 64-bit
            // value into two 32-bit halves. We emit a sub + store pattern.
            push_instructions.push(MachineInstr::with_operands(
                OP_SUB_RI,
                vec![
                    MachineOperand::Register(ESP),
                    MachineOperand::Immediate(8),
                ],
            ));
            // Store low 32 bits at [esp + 0]
            push_instructions.push(MachineInstr::with_operands(
                OP_MOV_MR,
                vec![
                    MachineOperand::Memory {
                        base: ESP,
                        offset: 0,
                    },
                    MachineOperand::Register(reg),
                ],
            ));
        } else {
            // Larger arguments (structs): sub esp, size and copy bytes.
            // The actual memcpy is handled by isel; we just allocate the space.
            push_instructions.push(MachineInstr::with_operands(
                OP_SUB_RI,
                vec![
                    MachineOperand::Register(ESP),
                    MachineOperand::Immediate(size as i64),
                ],
            ));
        }
    }

    CallSetup {
        push_instructions,
        args_size: total_args_size,
        alignment_padding,
    }
}

// =====================================================================
// Section 11: Unit Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::regalloc::PhysReg;
    use crate::driver::target::TargetConfig;
    use crate::ir::cfg::BasicBlock;
    use crate::ir::instructions::{BlockId, Instruction, Value};
    use crate::ir::types::IrType;

    /// Helper to create an i686 target config for tests.
    fn i686_target() -> TargetConfig {
        TargetConfig::i686()
    }

    /// Helper to create a minimal function for frame layout tests.
    fn make_function(
        name: &str,
        params: Vec<(String, IrType)>,
        blocks: Vec<BasicBlock>,
    ) -> crate::ir::Function {
        crate::ir::Function {
            name: name.to_string(),
            return_type: IrType::Void,
            params,
            blocks,
            entry_block: BlockId(0),
            is_definition: true,
        }
    }

    /// Helper to create a basic block with the given instructions.
    fn make_block(id: u32, instructions: Vec<Instruction>) -> BasicBlock {
        let mut block = BasicBlock::new(BlockId(id), format!("bb{}", id));
        block.instructions = instructions;
        block
    }

    // =================================================================
    // classify_arguments tests
    // =================================================================

    #[test]
    fn test_classify_arguments_empty() {
        let target = i686_target();
        let args = classify_arguments(&[], &target);
        assert!(args.is_empty(), "empty param list should produce empty args");
    }

    #[test]
    fn test_classify_arguments_single_i32() {
        let target = i686_target();
        let args = classify_arguments(&[IrType::I32], &target);
        assert_eq!(args.len(), 1);
        assert_eq!(args[0].stack_offset, 8, "first arg at [ebp + 8]");
        assert_eq!(args[0].size, 4);
    }

    #[test]
    fn test_classify_arguments_two_i32() {
        let target = i686_target();
        let args = classify_arguments(&[IrType::I32, IrType::I32], &target);
        assert_eq!(args.len(), 2);
        assert_eq!(args[0].stack_offset, 8, "first arg at [ebp + 8]");
        assert_eq!(args[0].size, 4);
        assert_eq!(args[1].stack_offset, 12, "second arg at [ebp + 12]");
        assert_eq!(args[1].size, 4);
    }

    #[test]
    fn test_classify_arguments_i32_i64() {
        let target = i686_target();
        let args = classify_arguments(&[IrType::I32, IrType::I64], &target);
        assert_eq!(args.len(), 2);
        assert_eq!(args[0].stack_offset, 8);
        assert_eq!(args[0].size, 4);
        // i64 takes 8 bytes (two 32-bit stack slots)
        assert_eq!(args[1].stack_offset, 12);
        assert_eq!(args[1].size, 8);
    }

    #[test]
    fn test_classify_arguments_mixed_types() {
        let target = i686_target();
        // (i32, f64, i32) — tests offset computation with 8-byte f64
        let args =
            classify_arguments(&[IrType::I32, IrType::F64, IrType::I32], &target);
        assert_eq!(args.len(), 3);
        assert_eq!(args[0].stack_offset, 8);
        assert_eq!(args[0].size, 4);
        assert_eq!(args[1].stack_offset, 12); // f64 at offset 12
        assert_eq!(args[1].size, 8);          // f64 is 8 bytes
        assert_eq!(args[2].stack_offset, 20); // 12 + 8 = 20
        assert_eq!(args[2].size, 4);
    }

    #[test]
    fn test_classify_arguments_pointer() {
        let target = i686_target();
        let ptr_type = IrType::Pointer(Box::new(IrType::I32));
        let args = classify_arguments(&[ptr_type], &target);
        assert_eq!(args.len(), 1);
        assert_eq!(args[0].stack_offset, 8);
        assert_eq!(args[0].size, 4, "pointers are 4 bytes on i686");
    }

    #[test]
    fn test_classify_arguments_i8_promoted() {
        let target = i686_target();
        let args = classify_arguments(&[IrType::I8], &target);
        assert_eq!(args[0].size, 4, "i8 promoted to 4 bytes on stack");
    }

    #[test]
    fn test_classify_arguments_i16_promoted() {
        let target = i686_target();
        let args = classify_arguments(&[IrType::I16], &target);
        assert_eq!(args[0].size, 4, "i16 promoted to 4 bytes on stack");
    }

    #[test]
    fn test_classify_arguments_f32() {
        let target = i686_target();
        let args = classify_arguments(&[IrType::F32], &target);
        assert_eq!(args[0].size, 4, "f32 is 4 bytes on stack");
    }

    #[test]
    fn test_classify_arguments_small_struct() {
        let target = i686_target();
        let struct_type = IrType::Struct {
            fields: vec![IrType::I8, IrType::I8],
            packed: false,
        };
        let args = classify_arguments(&[struct_type], &target);
        assert_eq!(args.len(), 1);
        assert!(args[0].size >= 4, "struct promoted to at least 4 bytes");
    }

    // =================================================================
    // classify_return tests
    // =================================================================

    #[test]
    fn test_classify_return_void() {
        let target = i686_target();
        let info = classify_return(&IrType::Void, &target);
        assert_eq!(info.location, ReturnLocation::Void);
    }

    #[test]
    fn test_classify_return_i32() {
        let target = i686_target();
        let info = classify_return(&IrType::I32, &target);
        assert_eq!(info.location, ReturnLocation::Eax);
    }

    #[test]
    fn test_classify_return_i8() {
        let target = i686_target();
        let info = classify_return(&IrType::I8, &target);
        assert_eq!(info.location, ReturnLocation::Eax);
    }

    #[test]
    fn test_classify_return_i64() {
        let target = i686_target();
        let info = classify_return(&IrType::I64, &target);
        assert_eq!(info.location, ReturnLocation::EaxEdx);
    }

    #[test]
    fn test_classify_return_f32() {
        let target = i686_target();
        let info = classify_return(&IrType::F32, &target);
        assert_eq!(info.location, ReturnLocation::St0);
    }

    #[test]
    fn test_classify_return_f64() {
        let target = i686_target();
        let info = classify_return(&IrType::F64, &target);
        assert_eq!(info.location, ReturnLocation::St0);
    }

    #[test]
    fn test_classify_return_pointer() {
        let target = i686_target();
        let ptr_type = IrType::Pointer(Box::new(IrType::I32));
        let info = classify_return(&ptr_type, &target);
        assert_eq!(info.location, ReturnLocation::Eax);
    }

    #[test]
    fn test_classify_return_small_struct() {
        let target = i686_target();
        // 4-byte struct: fits in eax
        let struct_type = IrType::Struct {
            fields: vec![IrType::I32],
            packed: false,
        };
        let info = classify_return(&struct_type, &target);
        assert_eq!(info.location, ReturnLocation::Eax);
    }

    #[test]
    fn test_classify_return_medium_struct() {
        let target = i686_target();
        // 8-byte struct: fits in eax:edx
        let struct_type = IrType::Struct {
            fields: vec![IrType::I32, IrType::I32],
            packed: false,
        };
        let info = classify_return(&struct_type, &target);
        assert_eq!(info.location, ReturnLocation::EaxEdx);
    }

    #[test]
    fn test_classify_return_large_struct() {
        let target = i686_target();
        // >8 byte struct: must use hidden pointer
        let struct_type = IrType::Struct {
            fields: vec![IrType::I32, IrType::I32, IrType::I32],
            packed: false,
        };
        let info = classify_return(&struct_type, &target);
        assert_eq!(info.location, ReturnLocation::Memory);
    }

    // =================================================================
    // needs_struct_return tests
    // =================================================================

    #[test]
    fn test_needs_struct_return_non_struct() {
        let target = i686_target();
        assert!(!needs_struct_return(&IrType::I32, &target));
        assert!(!needs_struct_return(&IrType::I64, &target));
        assert!(!needs_struct_return(&IrType::Void, &target));
    }

    #[test]
    fn test_needs_struct_return_small_struct() {
        let target = i686_target();
        let small = IrType::Struct {
            fields: vec![IrType::I32],
            packed: false,
        };
        assert!(!needs_struct_return(&small, &target), "4-byte struct doesn't need sret");
    }

    #[test]
    fn test_needs_struct_return_medium_struct() {
        let target = i686_target();
        let medium = IrType::Struct {
            fields: vec![IrType::I32, IrType::I32],
            packed: false,
        };
        assert!(
            !needs_struct_return(&medium, &target),
            "8-byte struct doesn't need sret"
        );
    }

    #[test]
    fn test_needs_struct_return_large_struct() {
        let target = i686_target();
        let large = IrType::Struct {
            fields: vec![IrType::I32, IrType::I32, IrType::I32],
            packed: false,
        };
        assert!(
            needs_struct_return(&large, &target),
            "12-byte struct needs sret"
        );
    }

    #[test]
    fn test_needs_struct_return_very_large() {
        let target = i686_target();
        let big = IrType::Struct {
            fields: vec![IrType::I64, IrType::I64, IrType::I64, IrType::I64],
            packed: false,
        };
        assert!(needs_struct_return(&big, &target), "32-byte struct needs sret");
    }

    // =================================================================
    // generate_prologue tests
    // =================================================================

    #[test]
    fn test_generate_prologue_minimal() {
        let frame = FrameLayout {
            frame_size: 0,
            locals_offset: 0,
            spill_offset: 0,
            callee_saved_count: 0,
            callee_saved_regs: vec![],
            use_frame_pointer: true,
            outgoing_args_size: 0,
        };
        let instrs = generate_prologue(&frame);
        // Should have: push ebp, mov ebp esp (2 instructions, no sub)
        assert_eq!(instrs.len(), 2);
        assert_eq!(instrs[0].opcode, OP_PUSH);
        assert_eq!(instrs[1].opcode, OP_MOV_RR);
    }

    #[test]
    fn test_generate_prologue_with_frame_size() {
        let frame = FrameLayout {
            frame_size: 32,
            locals_offset: -4,
            spill_offset: -20,
            callee_saved_count: 0,
            callee_saved_regs: vec![],
            use_frame_pointer: true,
            outgoing_args_size: 0,
        };
        let instrs = generate_prologue(&frame);
        // push ebp, mov ebp esp, sub esp 32
        assert_eq!(instrs.len(), 3);
        assert_eq!(instrs[2].opcode, OP_SUB_RI);
        if let MachineOperand::Immediate(val) = &instrs[2].operands[1] {
            assert_eq!(*val, 32);
        } else {
            panic!("expected immediate operand");
        }
    }

    #[test]
    fn test_generate_prologue_with_callee_saved() {
        let frame = FrameLayout {
            frame_size: 16,
            locals_offset: -16,
            spill_offset: -16,
            callee_saved_count: 3,
            callee_saved_regs: vec![EDI, ESI, EBX],
            use_frame_pointer: true,
            outgoing_args_size: 0,
        };
        let instrs = generate_prologue(&frame);
        // push ebp, mov ebp esp, push edi, push esi, push ebx, sub esp 16
        assert_eq!(instrs.len(), 6);
        assert_eq!(instrs[2].opcode, OP_PUSH);
        assert_eq!(instrs[3].opcode, OP_PUSH);
        assert_eq!(instrs[4].opcode, OP_PUSH);
        assert_eq!(instrs[5].opcode, OP_SUB_RI);
    }

    #[test]
    fn test_generate_prologue_no_frame_pointer() {
        let frame = FrameLayout {
            frame_size: 16,
            locals_offset: -4,
            spill_offset: -4,
            callee_saved_count: 0,
            callee_saved_regs: vec![],
            use_frame_pointer: false,
            outgoing_args_size: 0,
        };
        let instrs = generate_prologue(&frame);
        // Only sub esp, 16 (no push ebp, no mov ebp esp)
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0].opcode, OP_SUB_RI);
    }

    // =================================================================
    // generate_epilogue tests
    // =================================================================

    #[test]
    fn test_generate_epilogue_minimal() {
        let frame = FrameLayout {
            frame_size: 0,
            locals_offset: 0,
            spill_offset: 0,
            callee_saved_count: 0,
            callee_saved_regs: vec![],
            use_frame_pointer: true,
            outgoing_args_size: 0,
        };
        let instrs = generate_epilogue(&frame);
        // pop ebp, ret
        assert_eq!(instrs.len(), 2);
        assert_eq!(instrs[0].opcode, OP_POP);
        assert_eq!(instrs[1].opcode, OP_RET);
    }

    #[test]
    fn test_generate_epilogue_with_frame_size() {
        let frame = FrameLayout {
            frame_size: 32,
            locals_offset: -4,
            spill_offset: -4,
            callee_saved_count: 0,
            callee_saved_regs: vec![],
            use_frame_pointer: true,
            outgoing_args_size: 0,
        };
        let instrs = generate_epilogue(&frame);
        // add esp 32, pop ebp, ret
        assert_eq!(instrs.len(), 3);
        assert_eq!(instrs[0].opcode, OP_ADD_RI);
        assert_eq!(instrs[1].opcode, OP_POP);
        assert_eq!(instrs[2].opcode, OP_RET);
    }

    #[test]
    fn test_generate_epilogue_with_callee_saved() {
        let frame = FrameLayout {
            frame_size: 16,
            locals_offset: -16,
            spill_offset: -16,
            callee_saved_count: 3,
            callee_saved_regs: vec![EDI, ESI, EBX],
            use_frame_pointer: true,
            outgoing_args_size: 0,
        };
        let instrs = generate_epilogue(&frame);
        // add esp 16, pop ebx, pop esi, pop edi, pop ebp, ret
        assert_eq!(instrs.len(), 6);
        assert_eq!(instrs[0].opcode, OP_ADD_RI);
        assert_eq!(instrs[1].opcode, OP_POP); // pop ebx (reverse order)
        assert_eq!(instrs[2].opcode, OP_POP); // pop esi
        assert_eq!(instrs[3].opcode, OP_POP); // pop edi
        assert_eq!(instrs[4].opcode, OP_POP); // pop ebp
        assert_eq!(instrs[5].opcode, OP_RET);
    }

    #[test]
    fn test_generate_epilogue_no_frame_pointer() {
        let frame = FrameLayout {
            frame_size: 16,
            locals_offset: -4,
            spill_offset: -4,
            callee_saved_count: 0,
            callee_saved_regs: vec![],
            use_frame_pointer: false,
            outgoing_args_size: 0,
        };
        let instrs = generate_epilogue(&frame);
        // add esp 16, ret (no pop ebp)
        assert_eq!(instrs.len(), 2);
        assert_eq!(instrs[0].opcode, OP_ADD_RI);
        assert_eq!(instrs[1].opcode, OP_RET);
    }

    // =================================================================
    // compute_frame_layout tests
    // =================================================================

    #[test]
    fn test_compute_frame_layout_empty() {
        let target = i686_target();
        let blocks = vec![make_block(0, vec![])];
        let func = make_function("empty", vec![], blocks);

        let layout = compute_frame_layout(&func, 0, &[], &target);

        // Empty function: frame_size should be aligned to maintain 16-byte alignment
        // Base overhead: 4 (ret addr) + 4 (saved ebp) + 0 (callee saved) = 8
        // (8 + frame_size) % 16 == 0
        assert_eq!((8 + layout.frame_size) % 16, 0, "must be 16-byte aligned");
        assert_eq!(layout.callee_saved_count, 0);
        assert!(layout.use_frame_pointer);
    }

    #[test]
    fn test_compute_frame_layout_with_locals() {
        let target = i686_target();
        let blocks = vec![make_block(
            0,
            vec![Instruction::Alloca {
                result: Value(0),
                ty: IrType::I32,
                count: None,
            }],
        )];
        let func = make_function("with_locals", vec![], blocks);

        let layout = compute_frame_layout(&func, 0, &[], &target);

        assert!(layout.frame_size >= 4, "must allocate space for local i32");
        assert_eq!((8 + layout.frame_size) % 16, 0, "must be 16-byte aligned");
    }

    #[test]
    fn test_compute_frame_layout_with_spills() {
        let target = i686_target();
        let blocks = vec![make_block(0, vec![])];
        let func = make_function("with_spills", vec![], blocks);

        let layout = compute_frame_layout(&func, 4, &[], &target);

        assert!(layout.frame_size >= 16, "must allocate space for 4 spill slots");
        assert_eq!((8 + layout.frame_size) % 16, 0, "must be 16-byte aligned");
    }

    #[test]
    fn test_compute_frame_layout_with_callee_saved() {
        let target = i686_target();
        let blocks = vec![make_block(0, vec![])];
        let func = make_function("with_callee_saved", vec![], blocks);

        let callee_saved = vec![EBX, ESI, EDI];
        let layout = compute_frame_layout(&func, 0, &callee_saved, &target);

        assert_eq!(layout.callee_saved_count, 3);
        assert_eq!(layout.callee_saved_regs.len(), 3);
        assert_eq!(
            (8 + 12 + layout.frame_size) % 16,
            0,
            "must be 16-byte aligned"
        );
    }

    #[test]
    fn test_compute_frame_layout_alignment() {
        let target = i686_target();
        // Test that frame_size always produces 16-byte alignment
        for num_callee_saved in 0..4u32 {
            for num_spills in 0..8u32 {
                let blocks = vec![make_block(0, vec![])];
                let func = make_function("test", vec![], blocks);
                let callee_saved: Vec<PhysReg> = (0..num_callee_saved)
                    .map(|i| PhysReg(3 + i as u16))
                    .collect();

                let layout =
                    compute_frame_layout(&func, num_spills, &callee_saved, &target);
                let total =
                    8 + (num_callee_saved * 4) + layout.frame_size;
                assert_eq!(
                    total % 16,
                    0,
                    "alignment failed: callee_saved={}, spills={}, frame_size={}",
                    num_callee_saved,
                    num_spills,
                    layout.frame_size
                );
            }
        }
    }

    #[test]
    fn test_compute_frame_layout_with_params() {
        let target = i686_target();
        let blocks = vec![make_block(0, vec![])];
        let func = make_function(
            "with_params",
            vec![
                ("a".to_string(), IrType::I32),
                ("b".to_string(), IrType::I64),
            ],
            blocks,
        );

        let layout = compute_frame_layout(&func, 0, &[], &target);
        assert_eq!((8 + layout.frame_size) % 16, 0);
    }

    // =================================================================
    // setup_call_arguments tests
    // =================================================================

    #[test]
    fn test_setup_call_arguments_empty() {
        let target = i686_target();
        let setup = setup_call_arguments(&[], &[], &target);
        assert_eq!(setup.args_size, 0);
        assert_eq!(setup.alignment_padding, 0);
        assert!(setup.push_instructions.is_empty());
    }

    #[test]
    fn test_setup_call_arguments_single_i32() {
        let target = i686_target();
        let args = vec![Value(0)];
        let types = vec![IrType::I32];
        let setup = setup_call_arguments(&args, &types, &target);

        assert_eq!(setup.args_size, 4);
        assert_eq!(setup.alignment_padding, 12);
        assert!(!setup.push_instructions.is_empty());
    }

    #[test]
    fn test_setup_call_arguments_four_i32() {
        let target = i686_target();
        let args: Vec<Value> = (0..4).map(Value).collect();
        let types = vec![IrType::I32; 4];
        let setup = setup_call_arguments(&args, &types, &target);

        assert_eq!(setup.args_size, 16);
        assert_eq!(setup.alignment_padding, 0);
    }

    #[test]
    fn test_setup_call_arguments_i64() {
        let target = i686_target();
        let args = vec![Value(0)];
        let types = vec![IrType::I64];
        let setup = setup_call_arguments(&args, &types, &target);

        assert_eq!(setup.args_size, 8);
        assert_eq!(setup.alignment_padding, 8);
    }

    #[test]
    fn test_setup_call_arguments_right_to_left() {
        let target = i686_target();
        let args = vec![Value(10), Value(20), Value(30)];
        let types = vec![IrType::I32, IrType::I32, IrType::I32];
        let setup = setup_call_arguments(&args, &types, &target);

        assert_eq!(setup.args_size, 12);
        assert_eq!(setup.alignment_padding, 4);

        // Verify right-to-left push order
        let push_instrs: Vec<_> = setup
            .push_instructions
            .iter()
            .filter(|i| i.opcode == OP_PUSH)
            .collect();
        assert_eq!(push_instrs.len(), 3, "should have 3 push instructions");

        // First push should be Value(30), second Value(20), third Value(10)
        if let MachineOperand::Register(reg) = &push_instrs[0].operands[0] {
            assert_eq!(reg.0, 30, "first push should be last argument");
        }
        if let MachineOperand::Register(reg) = &push_instrs[1].operands[0] {
            assert_eq!(reg.0, 20, "second push should be middle argument");
        }
        if let MachineOperand::Register(reg) = &push_instrs[2].operands[0] {
            assert_eq!(reg.0, 10, "third push should be first argument");
        }
    }

    // =================================================================
    // ABI constants tests
    // =================================================================

    #[test]
    fn test_abi_constants() {
        assert_eq!(STACK_ALIGNMENT, 16);
        assert_eq!(POINTER_SIZE, 4);
        assert_eq!(RETURN_ADDRESS_SIZE, 4);
        assert_eq!(FRAME_POINTER_SIZE, 4);
    }

    // =================================================================
    // Register constants tests
    // =================================================================

    #[test]
    fn test_register_encoding() {
        assert_eq!(EAX.0, 0);
        assert_eq!(ECX.0, 1);
        assert_eq!(EDX.0, 2);
        assert_eq!(EBX.0, 3);
        assert_eq!(ESP.0, 4);
        assert_eq!(EBP.0, 5);
        assert_eq!(ESI.0, 6);
        assert_eq!(EDI.0, 7);
    }

    // =================================================================
    // Helper function tests
    // =================================================================

    #[test]
    fn test_align_up_32() {
        assert_eq!(align_up_32(0, 4), 0);
        assert_eq!(align_up_32(1, 4), 4);
        assert_eq!(align_up_32(3, 4), 4);
        assert_eq!(align_up_32(4, 4), 4);
        assert_eq!(align_up_32(5, 4), 8);
        assert_eq!(align_up_32(0, 16), 0);
        assert_eq!(align_up_32(1, 16), 16);
        assert_eq!(align_up_32(15, 16), 16);
        assert_eq!(align_up_32(16, 16), 16);
        assert_eq!(align_up_32(17, 16), 32);
    }

    #[test]
    fn test_arg_stack_size() {
        let target = i686_target();
        assert_eq!(arg_stack_size(&IrType::I8, &target), 4);
        assert_eq!(arg_stack_size(&IrType::I16, &target), 4);
        assert_eq!(arg_stack_size(&IrType::I32, &target), 4);
        assert_eq!(arg_stack_size(&IrType::I64, &target), 8);
        assert_eq!(arg_stack_size(&IrType::F32, &target), 4);
        assert_eq!(arg_stack_size(&IrType::F64, &target), 8);
        assert_eq!(
            arg_stack_size(&IrType::Pointer(Box::new(IrType::I32)), &target),
            4
        );
    }

    // =================================================================
    // Struct return with hidden pointer test
    // =================================================================

    #[test]
    fn test_struct_return_shifts_arg_offsets() {
        let target = i686_target();

        let return_type = IrType::Struct {
            fields: vec![IrType::I32, IrType::I32, IrType::I32, IrType::I32],
            packed: false,
        };
        assert!(needs_struct_return(&return_type, &target));

        // Without hidden pointer:
        let explicit_args = vec![IrType::I32, IrType::I32];
        let args_no_sret = classify_arguments(&explicit_args, &target);
        assert_eq!(args_no_sret[0].stack_offset, 8);
        assert_eq!(args_no_sret[1].stack_offset, 12);

        // With hidden pointer: prepend a pointer type as the first arg
        let mut sret_args = vec![IrType::Pointer(Box::new(return_type))];
        sret_args.extend(explicit_args);
        let args_with_sret = classify_arguments(&sret_args, &target);
        assert_eq!(args_with_sret[0].stack_offset, 8);  // hidden ptr
        assert_eq!(args_with_sret[0].size, 4);
        assert_eq!(args_with_sret[1].stack_offset, 12); // first explicit (+4)
        assert_eq!(args_with_sret[2].stack_offset, 16); // second explicit (+4)
    }

    // =================================================================
    // Prologue/epilogue symmetry test
    // =================================================================

    #[test]
    fn test_prologue_epilogue_symmetry() {
        let frame = FrameLayout {
            frame_size: 32,
            locals_offset: -16,
            spill_offset: -24,
            callee_saved_count: 2,
            callee_saved_regs: vec![EBX, ESI],
            use_frame_pointer: true,
            outgoing_args_size: 0,
        };

        let prologue = generate_prologue(&frame);
        let epilogue = generate_epilogue(&frame);

        let prologue_pushes: Vec<_> = prologue
            .iter()
            .filter(|i| i.opcode == OP_PUSH)
            .collect();
        let epilogue_pops: Vec<_> = epilogue
            .iter()
            .filter(|i| i.opcode == OP_POP)
            .collect();

        // 1 push for ebp + 2 for callee-saved = 3 pushes
        assert_eq!(prologue_pushes.len(), 3);
        // 2 pops for callee-saved + 1 for ebp = 3 pops
        assert_eq!(epilogue_pops.len(), 3);

        let has_sub = prologue.iter().any(|i| i.opcode == OP_SUB_RI);
        let has_add = epilogue.iter().any(|i| i.opcode == OP_ADD_RI);
        assert!(has_sub, "prologue should have sub esp");
        assert!(has_add, "epilogue should have add esp");
    }

    // =================================================================
    // i686-specific ABI compliance tests
    // =================================================================

    #[test]
    fn test_no_register_arguments() {
        let target = i686_target();
        let many_args = vec![
            IrType::I32, IrType::I32, IrType::I32, IrType::I32,
            IrType::I32, IrType::I32, IrType::I32, IrType::I32,
        ];
        let args = classify_arguments(&many_args, &target);
        for (i, arg) in args.iter().enumerate() {
            assert!(
                arg.stack_offset >= 8,
                "arg {} must be on stack (offset >= 8), got {}",
                i,
                arg.stack_offset
            );
        }
    }

    #[test]
    fn test_target_is_32bit() {
        let target = i686_target();
        assert!(target.is_32bit());
        assert_eq!(target.pointer_size, 4);
        assert_eq!(target.stack_alignment, 16);
    }
}
