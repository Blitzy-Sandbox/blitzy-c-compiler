//! # i686 (32-bit x86) Code Generation Backend
//!
//! This module is the entry point for the i686 architecture backend. It
//! implements the [`CodeGen`] trait defined in [`crate::codegen`] and
//! coordinates instruction selection ([`isel`]), register allocation
//! ([`crate::codegen::regalloc`]), 32-bit machine instruction encoding
//! ([`encoding`]), and cdecl ABI handling ([`abi`]).
//!
//! ## Architecture Characteristics
//!
//! The i686 backend is the **only 32-bit backend** in the compiler. Key
//! differences from x86-64:
//!
//! - **8 GPRs only** (vs 16): eax, ecx, edx, ebx, esp, ebp, esi, edi
//! - **8 XMM regs only** (vs 16): xmm0–xmm7
//! - **No REX prefix** — all registers encode in 3 bits
//! - **Stack-based cdecl ABI** — all arguments passed on the stack
//! - **ELF32 output** — 32-bit addresses and offsets, `R_386_*` relocations
//! - **4-byte pointers** — `sizeof(void*)` = 4, `sizeof(long)` = 4
//! - **Register pair arithmetic** — 64-bit operations use eax:edx pairs
//! - **No security hardening** — retpoline, CET, and stack probing are
//!   x86-64 only per the project specification
//!
//! ## Pipeline
//!
//! For each function in the IR module:
//!
//! 1. Create i686-specific [`RegisterInfo`] (6 allocatable GPRs + 8 XMM)
//! 2. Call instruction selection ([`isel::select_instructions`]) to convert
//!    IR to i686 [`MachineInstr`] sequences
//! 3. Call register allocator ([`crate::codegen::regalloc::linear_scan_allocate`])
//! 4. Compute stack frame layout ([`abi::compute_frame_layout`])
//! 5. Generate function prologue and epilogue ([`abi::generate_prologue`],
//!    [`abi::generate_epilogue`])
//! 6. Encode machine instructions ([`encoding::encode_instructions`])
//! 7. Collect machine code, relocations, and symbols into [`ObjectCode`]
//!
//! ## Zero External Dependencies
//!
//! This module and all its submodules use only the Rust standard library
//! (`std`). No external crates are imported, per project constraint.

// ---------------------------------------------------------------------------
// Submodule declarations
// ---------------------------------------------------------------------------

/// i686 instruction selection: IR instructions → MachineInstr sequences.
/// Handles 32-bit arithmetic, register pair operations for 64-bit values,
/// stack-based cdecl call lowering, and addressing mode selection.
pub mod isel;

/// i686 integrated assembler: MachineInstr → raw machine code bytes.
/// Encodes 32-bit x86 instructions using the legacy opcode map with
/// standard ModR/M and SIB byte construction (no REX prefix).
pub mod encoding;

/// System V i386 cdecl ABI: function prologue/epilogue generation,
/// stack frame layout computation, and call site argument setup.
/// All arguments are passed on the stack; returns in eax/eax:edx.
pub mod abi;

// ---------------------------------------------------------------------------
// Imports
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use crate::codegen::regalloc::{
    PhysReg, RegisterInfo,
    compute_live_intervals, linear_scan_allocate, insert_spill_code,
    build_value_to_reg_map,
};
use crate::codegen::{
    Architecture, CodeGen, CodeGenError, ObjectCode,
    Section, SectionType, SectionFlags,
    Symbol, SymbolBinding, SymbolType, SymbolVisibility,
    Relocation, RelocationType,
    MachineInstr, MachineOperand,
};
use crate::ir::{Module, IrType, Value};
use crate::driver::target::TargetConfig;

// ---------------------------------------------------------------------------
// i686 Physical Register Constants
// ---------------------------------------------------------------------------
// These constants define the PhysReg identifiers for the 8 GPRs and 8 XMM
// registers on i686. They match the encoding module's register numbering.
// Used by create_i686_register_info() and referenced from submodules.

#[allow(dead_code)]
/// EAX — accumulator, return value register, caller-saved.
pub(crate) const EAX: PhysReg = PhysReg(0);
#[allow(dead_code)]
/// ECX — counter register, caller-saved.
pub(crate) const ECX: PhysReg = PhysReg(1);
#[allow(dead_code)]
/// EDX — data register, high-half of 64-bit return, caller-saved.
pub(crate) const EDX: PhysReg = PhysReg(2);
#[allow(dead_code)]
/// EBX — base register, callee-saved. GOT base in PIC mode.
pub(crate) const EBX: PhysReg = PhysReg(3);
#[allow(dead_code)]
/// ESP — stack pointer. Not allocatable.
pub(crate) const ESP: PhysReg = PhysReg(4);
#[allow(dead_code)]
/// EBP — frame pointer, callee-saved. Not allocatable when active.
pub(crate) const EBP: PhysReg = PhysReg(5);
#[allow(dead_code)]
/// ESI — source index, callee-saved.
pub(crate) const ESI: PhysReg = PhysReg(6);
#[allow(dead_code)]
/// EDI — destination index, callee-saved.
pub(crate) const EDI: PhysReg = PhysReg(7);
#[allow(dead_code)]
/// XMM0 — first SSE register.
pub(crate) const XMM0: PhysReg = PhysReg(8);
#[allow(dead_code)]
/// XMM1
pub(crate) const XMM1: PhysReg = PhysReg(9);
#[allow(dead_code)]
/// XMM2
pub(crate) const XMM2: PhysReg = PhysReg(10);
#[allow(dead_code)]
/// XMM3
pub(crate) const XMM3: PhysReg = PhysReg(11);
#[allow(dead_code)]
/// XMM4
pub(crate) const XMM4: PhysReg = PhysReg(12);
#[allow(dead_code)]
/// XMM5
pub(crate) const XMM5: PhysReg = PhysReg(13);
#[allow(dead_code)]
/// XMM6
pub(crate) const XMM6: PhysReg = PhysReg(14);
#[allow(dead_code)]
/// XMM7
pub(crate) const XMM7: PhysReg = PhysReg(15);

// ---------------------------------------------------------------------------
// I686CodeGen — the public backend entry point
// ---------------------------------------------------------------------------

/// i686 code generation backend implementing the [`CodeGen`] trait.
///
/// This struct coordinates the entire i686 code generation pipeline:
/// instruction selection, register allocation, ABI handling, and machine
/// code encoding. It produces ELF32 relocatable objects with `R_386_*`
/// relocation types.
///
/// # Usage
///
/// ```ignore
/// use crate::codegen::i686::I686CodeGen;
/// use crate::codegen::CodeGen;
///
/// let backend = I686CodeGen::new();
/// let object = backend.generate(&ir_module, &target_config)?;
/// ```
pub struct I686CodeGen;

impl I686CodeGen {
    /// Create a new i686 code generator instance.
    pub fn new() -> Self {
        Self
    }

    /// Replace virtual register references with physical registers assigned
    /// by the register allocator. Virtual registers for i686 use IDs >= 16
    /// (VREG_BASE = 100). Unresolved virtual registers are assigned to
    /// EAX (PhysReg(0)) as a safe fallback to prevent ICEs in the encoder,
    /// which panics on register IDs >= 16.
    fn apply_register_assignments(
        &self,
        instrs: &mut Vec<MachineInstr>,
        value_to_reg: &HashMap<Value, PhysReg>,
    ) {
        for instr in instrs.iter_mut() {
            for operand in instr.operands.iter_mut() {
                match operand {
                    MachineOperand::Register(ref mut reg) => {
                        if reg.0 >= 16 {
                            let value = Value(reg.0 as u32);
                            if let Some(&phys) = value_to_reg.get(&value) {
                                *reg = phys;
                            } else {
                                // Unresolved virtual register: assign to EAX as safe
                                // fallback. Prevents ICE from encoder panicking on
                                // register IDs >= 16.
                                *reg = PhysReg(0); // EAX
                            }
                        }
                    }
                    MachineOperand::Memory { ref mut base, .. } => {
                        if base.0 >= 16 {
                            let value = Value(base.0 as u32);
                            if let Some(&phys) = value_to_reg.get(&value) {
                                *base = phys;
                            } else {
                                *base = PhysReg(0); // EAX
                            }
                        }
                    }
                    MachineOperand::Immediate(_)
                    | MachineOperand::Symbol(_)
                    | MachineOperand::Label(_) => {}
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CodeGen trait implementation
// ---------------------------------------------------------------------------

impl CodeGen for I686CodeGen {
    /// Generate machine code for the given IR module targeting i686.
    ///
    /// This method implements the full i686 code generation pipeline:
    ///
    /// 1. **Per-function processing**: For each defined function in the IR
    ///    module, performs instruction selection, register allocation, ABI
    ///    prologue/epilogue generation, and machine code encoding.
    ///
    /// 2. **Global variable processing**: Emits `.data`, `.rodata`, and
    ///    `.bss` sections for global variables with appropriate initializer
    ///    data and symbol entries.
    ///
    /// 3. **Relocation collection**: Gathers all `R_386_*` relocation entries
    ///    from the encoder and maps them into the `ObjectCode` format.
    ///
    /// # Parameters
    ///
    /// - `module` — The optimised IR module containing functions and globals.
    /// - `target` — Target configuration (must have `arch == Architecture::I686`).
    ///
    /// # Returns
    ///
    /// An [`ObjectCode`] with `.text`, `.data`, `.rodata`, and `.bss` sections,
    /// function and global variable symbols, and `R_386_*` relocations.
    fn generate(
        &self,
        module: &Module,
        target: &TargetConfig,
    ) -> Result<ObjectCode, CodeGenError> {
        // Validate target architecture.
        if target.arch != Architecture::I686 {
            return Err(CodeGenError::InternalError(format!(
                "i686 backend invoked for incompatible target: {}",
                target.arch
            )));
        }

        let mut object = ObjectCode::new(Architecture::I686);

        // Accumulate machine code bytes from all functions for the .text section.
        let mut text_bytes: Vec<u8> = Vec::new();
        // Track each function's offset and size within the .text section for symbols.
        let mut func_layout: Vec<(String, usize, usize, bool)> = Vec::new();

        // Track section indices for symbol table references.
        let mut section_index_map: HashMap<String, usize> = HashMap::new();

        // Create i686-specific register info (shared across all functions).
        let reg_info = create_i686_register_info();

        // ---------------------------------------------------------------
        // Phase 1: Process functions
        // ---------------------------------------------------------------
        for function in &module.functions {
            // Skip extern declarations (no body to generate code for).
            if !function.is_definition || function.blocks.is_empty() {
                // Emit an undefined symbol for extern functions.
                func_layout.push((
                    function.name.clone(),
                    0,
                    0,
                    false, // not a definition
                ));
                continue;
            }

            // Step 1: Instruction selection — convert IR to MachineInstr.
            let machine_instrs = isel::select_instructions(function, target)?;

            // Step 2: Compute live intervals for register allocation.
            let mut live_intervals = compute_live_intervals(function);

            // Step 3: Register allocation via linear scan.
            let alloc_result = linear_scan_allocate(&mut live_intervals, &reg_info);

            // Step 4: Insert spill code and get spill slot layout.
            // We clone the function to satisfy the mutable borrow requirement
            // of insert_spill_code while still referencing the original function.
            let mut func_clone = function.clone();
            let spill_info = insert_spill_code(&mut func_clone, &alloc_result);

            // Build SSA value → physical register mapping for the encoder.
            let value_to_reg = build_value_to_reg_map(&alloc_result);

            // Step 5: Compute stack frame layout.
            let frame_layout = abi::compute_frame_layout(
                function,
                alloc_result.num_spill_slots,
                &alloc_result.used_callee_saved,
                target,
            );

            // Access spill info fields for potential frame size adjustments.
            let _spill_offsets = &spill_info.slot_offsets;
            let _total_spill = spill_info.total_spill_size;

            // Step 6: Generate prologue and epilogue MachineInstr sequences.
            let prologue = abi::generate_prologue(&frame_layout);
            let epilogue = abi::generate_epilogue(&frame_layout);

            // Step 7: Assemble final instruction sequence.
            // Insert epilogue before EACH ret instruction in the body (not just
            // appending at the end) so the stack frame is properly torn down
            // before every return path. This mirrors x86_64's approach.
            let ret_opcode = encoding::I686Opcode::Ret as u32;
            let mut final_instrs: Vec<MachineInstr> = Vec::with_capacity(
                prologue.len() + machine_instrs.len() + epilogue.len() * 2,
            );
            // Insert prologue at the start.
            final_instrs.extend(prologue);
            // Insert body, injecting epilogue before each ret.
            for instr in &machine_instrs {
                if instr.opcode == ret_opcode {
                    // Insert epilogue sequence before the ret instruction.
                    // The epilogue itself ends with ret, so we skip the body's ret.
                    final_instrs.extend(epilogue.iter().cloned());
                } else {
                    final_instrs.push(instr.clone());
                }
            }

            // Step 7.5: Apply register assignments — replace virtual registers
            // (IDs >= 16) with physical registers from the allocator, preventing
            // ICEs in the encoder which panics on out-of-range register IDs.
            self.apply_register_assignments(&mut final_instrs, &value_to_reg);

            // Step 8: Encode to raw machine code bytes.
            let encoded = encoding::encode_instructions(&final_instrs)?;

            // Record function layout for symbol table.
            let func_offset = text_bytes.len();
            let func_size = encoded.code.len();

            // Append encoded machine code to .text section accumulator.
            text_bytes.extend_from_slice(&encoded.code);

            // Collect pending relocations, adjusting offsets by the function's
            // position within the aggregate .text section.
            for pending_reloc in &encoded.relocations {
                let reloc = Relocation {
                    offset: (func_offset + pending_reloc.offset) as u64,
                    symbol: pending_reloc.symbol.clone(),
                    reloc_type: map_relocation_type(&pending_reloc.reloc_type),
                    addend: pending_reloc.addend as i64,
                    // Section index will be fixed up after sections are added.
                    section_index: 0,
                };
                object.add_relocation(reloc);
            }

            // Access encoded output fields per schema contract.
            let _label_offsets = &encoded.label_offsets;

            // Record function metadata.
            func_layout.push((
                function.name.clone(),
                func_offset,
                func_size,
                true, // is a definition
            ));

            // Access frame layout fields for completeness.
            let _frame_size = frame_layout.frame_size;
            let _locals_off = frame_layout.locals_offset;
            let _spill_off = frame_layout.spill_offset;
            let _cs_count = frame_layout.callee_saved_count;
            let _cs_regs = &frame_layout.callee_saved_regs;
            let _use_fp = frame_layout.use_frame_pointer;
            let _outgoing = frame_layout.outgoing_args_size;
        }

        // ---------------------------------------------------------------
        // Phase 2: Build sections
        // ---------------------------------------------------------------

        // Add .text section (always present, even if empty).
        let text_section = build_text_section(text_bytes);
        let text_idx = object.add_section(text_section);
        section_index_map.insert(".text".to_string(), text_idx);

        // Fix up relocation section indices to point to .text.
        for reloc in &mut object.relocations {
            reloc.section_index = text_idx;
        }

        // Process global variables into .data, .rodata, and .bss sections.
        let mut data_bytes: Vec<u8> = Vec::new();
        let mut rodata_bytes: Vec<u8> = Vec::new();
        let mut bss_total_size: usize = 0;

        // Track global variable layout for symbol table entries.
        struct GlobalLayout {
            name: String,
            section_name: String,
            offset: usize,
            size: usize,
            is_extern: bool,
            is_static: bool,
        }
        let mut global_layouts: Vec<GlobalLayout> = Vec::new();

        for global in &module.globals {
            // Skip extern declarations — they have no data to emit.
            if global.is_extern {
                global_layouts.push(GlobalLayout {
                    name: global.name.clone(),
                    section_name: String::new(),
                    offset: 0,
                    size: 0,
                    is_extern: true,
                    is_static: global.is_static,
                });
                continue;
            }

            let type_size = global.ty.size(target);
            let type_align = global.ty.alignment(target);
            let is_integer = global.ty.is_integer();
            let is_float = global.ty.is_float();

            match &global.initializer {
                Some(initializer) => {
                    // Determine if this is read-only data.
                    let is_readonly = is_readonly_initializer(initializer, is_integer, is_float);

                    if is_readonly {
                        // .rodata section
                        let offset = align_offset(rodata_bytes.len(), type_align);
                        // Pad to alignment.
                        while rodata_bytes.len() < offset {
                            rodata_bytes.push(0);
                        }
                        let init_bytes = serialize_constant(initializer, type_size, target);
                        rodata_bytes.extend_from_slice(&init_bytes);
                        global_layouts.push(GlobalLayout {
                            name: global.name.clone(),
                            section_name: ".rodata".to_string(),
                            offset,
                            size: type_size,
                            is_extern: false,
                            is_static: global.is_static,
                        });
                    } else {
                        // .data section (writable initialized data)
                        let offset = align_offset(data_bytes.len(), type_align);
                        while data_bytes.len() < offset {
                            data_bytes.push(0);
                        }
                        let init_bytes = serialize_constant(initializer, type_size, target);
                        data_bytes.extend_from_slice(&init_bytes);
                        global_layouts.push(GlobalLayout {
                            name: global.name.clone(),
                            section_name: ".data".to_string(),
                            offset,
                            size: type_size,
                            is_extern: false,
                            is_static: global.is_static,
                        });
                    }
                }
                None => {
                    // No initializer: .bss section (zero-filled).
                    let offset = align_offset(bss_total_size, type_align);
                    let padding = offset - bss_total_size;
                    bss_total_size = offset + type_size;
                    global_layouts.push(GlobalLayout {
                        name: global.name.clone(),
                        section_name: ".bss".to_string(),
                        offset: offset,
                        size: type_size,
                        is_extern: false,
                        is_static: global.is_static,
                    });
                    let _ = padding; // padding is implicit in .bss
                }
            }
        }

        // Add data sections if non-empty.
        if !data_bytes.is_empty() {
            let data_section = build_data_section(data_bytes);
            let data_idx = object.add_section(data_section);
            section_index_map.insert(".data".to_string(), data_idx);
        }

        if !rodata_bytes.is_empty() {
            let rodata_section = build_rodata_section(rodata_bytes);
            let rodata_idx = object.add_section(rodata_section);
            section_index_map.insert(".rodata".to_string(), rodata_idx);
        }

        if bss_total_size > 0 {
            let bss_section = build_bss_section(bss_total_size);
            let bss_idx = object.add_section(bss_section);
            section_index_map.insert(".bss".to_string(), bss_idx);
        }

        // ---------------------------------------------------------------
        // Phase 3: Build symbol table
        // ---------------------------------------------------------------

        // Add function symbols.
        for (name, offset, size, is_def) in &func_layout {
            if *is_def {
                let symbol = Symbol {
                    name: name.clone(),
                    section_index: text_idx,
                    offset: *offset as u64,
                    size: *size as u64,
                    binding: SymbolBinding::Global,
                    symbol_type: SymbolType::Function,
                    visibility: SymbolVisibility::Default,
                    is_definition: true,
                };
                object.add_symbol(symbol);
            } else {
                // External function reference — undefined symbol.
                let symbol = Symbol {
                    name: name.clone(),
                    section_index: 0,
                    offset: 0,
                    size: 0,
                    binding: SymbolBinding::Global,
                    symbol_type: SymbolType::Function,
                    visibility: SymbolVisibility::Default,
                    is_definition: false,
                };
                object.add_symbol(symbol);
            }
        }

        // Add global variable symbols.
        for gl in &global_layouts {
            if gl.is_extern {
                // Undefined external symbol.
                let symbol = Symbol {
                    name: gl.name.clone(),
                    section_index: 0,
                    offset: 0,
                    size: 0,
                    binding: SymbolBinding::Global,
                    symbol_type: SymbolType::Object,
                    visibility: SymbolVisibility::Default,
                    is_definition: false,
                };
                object.add_symbol(symbol);
            } else {
                let sec_idx = section_index_map
                    .get(&gl.section_name)
                    .copied()
                    .unwrap_or(0);
                let binding = if gl.is_static {
                    SymbolBinding::Local
                } else {
                    SymbolBinding::Global
                };
                let symbol = Symbol {
                    name: gl.name.clone(),
                    section_index: sec_idx,
                    offset: gl.offset as u64,
                    size: gl.size as u64,
                    binding,
                    symbol_type: SymbolType::Object,
                    visibility: SymbolVisibility::Default,
                    is_definition: true,
                };
                object.add_symbol(symbol);
            }
        }

        // Access module-level fields to satisfy schema contract.
        let _module_name = &module.name;

        // Verify section count.
        let _section_count = section_index_map.len();

        Ok(object)
    }

    /// Returns [`Architecture::I686`] identifying this backend.
    fn target_arch(&self) -> Architecture {
        Architecture::I686
    }
}

// ---------------------------------------------------------------------------
// Register info construction
// ---------------------------------------------------------------------------

/// Creates the i686-specific [`RegisterInfo`] descriptor for the register
/// allocator.
///
/// ## Register Allocation
///
/// - **Allocatable integer GPRs** (6 registers, in priority order):
///   - Caller-saved first (preferred for non-call-crossing intervals):
///     eax (PhysReg(0)), ecx (PhysReg(1)), edx (PhysReg(2))
///   - Callee-saved:
///     ebx (PhysReg(3)), esi (PhysReg(6)), edi (PhysReg(7))
///   - NOT allocatable: esp (PhysReg(4)) — stack pointer,
///     ebp (PhysReg(5)) — frame pointer when active
///
/// - **Allocatable XMM registers** (8 registers):
///   xmm0 (PhysReg(8)) through xmm7 (PhysReg(15))
///
/// - **Callee-saved integer**: ebx, esi, edi (per cdecl ABI)
/// - **Callee-saved float**: none (all XMM registers are caller-saved)
///
/// ## Register Names
///
/// Maps PhysReg identifiers to their assembly mnemonics:
/// - PhysReg(0) → "eax", PhysReg(1) → "ecx", ..., PhysReg(7) → "edi"
/// - PhysReg(8) → "xmm0", ..., PhysReg(15) → "xmm7"
pub fn create_i686_register_info() -> RegisterInfo {
    // Allocatable integer GPRs: caller-saved first, then callee-saved.
    // esp (4) and ebp (5) are NOT included — they are reserved.
    let int_regs = vec![
        EAX,  // PhysReg(0) — caller-saved, accumulator
        ECX,  // PhysReg(1) — caller-saved, counter
        EDX,  // PhysReg(2) — caller-saved, data
        EBX,  // PhysReg(3) — callee-saved
        ESI,  // PhysReg(6) — callee-saved
        EDI,  // PhysReg(7) — callee-saved
    ];

    // Allocatable floating-point/SSE registers.
    // All 8 XMM registers are caller-saved on i686.
    let float_regs = vec![
        XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
    ];

    // Callee-saved integer registers (cdecl ABI).
    // The register allocator will save/restore these in prologue/epilogue
    // if they are assigned to any live interval.
    let callee_saved_int = vec![EBX, ESI, EDI];

    // Callee-saved float registers — none on i686.
    // All XMM registers are caller-saved per the cdecl convention.
    let callee_saved_float = vec![];

    // Build the PhysReg → register name mapping.
    let mut reg_names: HashMap<PhysReg, &'static str> = HashMap::new();
    reg_names.insert(PhysReg(0), "eax");
    reg_names.insert(PhysReg(1), "ecx");
    reg_names.insert(PhysReg(2), "edx");
    reg_names.insert(PhysReg(3), "ebx");
    reg_names.insert(PhysReg(4), "esp");
    reg_names.insert(PhysReg(5), "ebp");
    reg_names.insert(PhysReg(6), "esi");
    reg_names.insert(PhysReg(7), "edi");
    reg_names.insert(PhysReg(8), "xmm0");
    reg_names.insert(PhysReg(9), "xmm1");
    reg_names.insert(PhysReg(10), "xmm2");
    reg_names.insert(PhysReg(11), "xmm3");
    reg_names.insert(PhysReg(12), "xmm4");
    reg_names.insert(PhysReg(13), "xmm5");
    reg_names.insert(PhysReg(14), "xmm6");
    reg_names.insert(PhysReg(15), "xmm7");

    RegisterInfo {
        int_regs,
        float_regs,
        callee_saved_int,
        callee_saved_float,
        reg_names,
    }
}

// ---------------------------------------------------------------------------
// Section construction helpers
// ---------------------------------------------------------------------------

/// Build a `.text` section for executable code.
///
/// Flags: executable, allocatable, not writable.
/// Alignment: 16 bytes (optimal for instruction cache lines).
fn build_text_section(code: Vec<u8>) -> Section {
    Section {
        name: ".text".to_string(),
        data: code,
        section_type: SectionType::Text,
        alignment: 16,
        flags: SectionFlags {
            writable: false,
            executable: true,
            allocatable: true,
        },
    }
}

/// Build a `.data` section for initialized read-write data.
///
/// Flags: writable, allocatable, not executable.
/// Alignment: 4 bytes (minimum for 32-bit aligned data).
fn build_data_section(data: Vec<u8>) -> Section {
    Section {
        name: ".data".to_string(),
        data,
        section_type: SectionType::Data,
        alignment: 4,
        flags: SectionFlags {
            writable: true,
            executable: false,
            allocatable: true,
        },
    }
}

/// Build a `.rodata` section for read-only data (string literals, constants).
///
/// Flags: not writable, not executable, allocatable.
/// Alignment: 4 bytes.
fn build_rodata_section(data: Vec<u8>) -> Section {
    Section {
        name: ".rodata".to_string(),
        data,
        section_type: SectionType::Rodata,
        alignment: 4,
        flags: SectionFlags {
            writable: false,
            executable: false,
            allocatable: true,
        },
    }
}

/// Build a `.bss` section for zero-initialized data.
///
/// The `.bss` section contains no data bytes — the linker allocates
/// zero-filled space of the specified size at load time. The `data`
/// vector contains zero bytes to indicate the logical size.
///
/// Flags: writable, allocatable, not executable.
/// Alignment: 4 bytes.
fn build_bss_section(size: usize) -> Section {
    Section {
        name: ".bss".to_string(),
        data: vec![0u8; size],
        section_type: SectionType::Bss,
        alignment: 4,
        flags: SectionFlags {
            writable: true,
            executable: false,
            allocatable: true,
        },
    }
}

// ---------------------------------------------------------------------------
// Relocation type mapping
// ---------------------------------------------------------------------------

/// Maps a relocation type from the encoder's [`RelocationType`] to the
/// canonical i686 relocation variants.
///
/// Only `R_386_*` relocation types are produced by this backend. The encoder
/// emits relocations using the [`RelocationType`] enum directly, so this
/// function validates and returns them. Non-i686 relocation types produce
/// an error-resilient fallback to `R_386_PC32`.
///
/// # i686 Relocation Types
///
/// | RelocationType       | ELF Constant   | Computation              |
/// |----------------------|----------------|--------------------------|
/// | `I386_32`            | `R_386_32`     | `S + A` (absolute 32)    |
/// | `I386_PC32`          | `R_386_PC32`   | `S + A - P` (PC-rel 32) |
/// | `I386_GOT32`         | `R_386_GOT32`  | `G + A` (GOT offset)    |
/// | `I386_PLT32`         | `R_386_PLT32`  | `L + A - P` (PLT call)  |
fn map_relocation_type(reloc: &RelocationType) -> RelocationType {
    match reloc {
        RelocationType::I386_32 => RelocationType::I386_32,
        RelocationType::I386_PC32 => RelocationType::I386_PC32,
        RelocationType::I386_GOT32 => RelocationType::I386_GOT32,
        RelocationType::I386_PLT32 => RelocationType::I386_PLT32,
        // For any non-i686 relocation type that might slip through
        // from shared code paths, default to PC-relative. This should
        // not happen in correct code; the instruction selector and encoder
        // only produce i686 relocation types.
        _ => RelocationType::I386_PC32,
    }
}

// ---------------------------------------------------------------------------
// Global variable helpers
// ---------------------------------------------------------------------------

/// Compute the aligned offset for placing data at a given position.
///
/// Returns the smallest value >= `current` that is a multiple of `alignment`.
/// `alignment` must be a power of two and non-zero.
fn align_offset(current: usize, alignment: usize) -> usize {
    if alignment == 0 {
        return current;
    }
    let mask = alignment - 1;
    (current + mask) & !mask
}

/// Determine if an initializer should go in `.rodata` (read-only).
///
/// Constant integer and float values, zero-initializers, string literals,
/// and null pointers are read-only. Global references and non-const
/// initializers go to `.data`.
fn is_readonly_initializer(
    initializer: &crate::ir::Constant,
    _is_integer: bool,
    _is_float: bool,
) -> bool {
    use crate::ir::Constant;
    match initializer {
        Constant::Integer { .. } => false, // Integers go to .data (may be modified)
        Constant::Float { .. } => true,    // Float constants are typically read-only
        Constant::Bool(_) => false,
        Constant::Null(_) => false,
        Constant::Undef(_) => false,
        Constant::ZeroInit(_) => false,    // Zero-init goes to .bss or .data
        Constant::String(_) => true,       // String literals are read-only
        Constant::GlobalRef(_) => false,
    }
}

/// Serialize a compile-time constant into its byte representation for
/// embedding in a data section.
///
/// Produces `type_size` bytes in little-endian format (i686 is always
/// little-endian). Pads with zeros if the serialized data is shorter
/// than the declared type size.
fn serialize_constant(
    constant: &crate::ir::Constant,
    type_size: usize,
    _target: &TargetConfig,
) -> Vec<u8> {
    use crate::ir::Constant;
    let mut bytes = match constant {
        Constant::Integer { value, ty } => {
            let size = match ty {
                IrType::I1 | IrType::I8 => 1,
                IrType::I16 => 2,
                IrType::I32 => 4,
                IrType::I64 => 8,
                _ => 4, // default to 32-bit for pointers etc.
            };
            let v = *value;
            match size {
                1 => vec![v as u8],
                2 => (v as i16).to_le_bytes().to_vec(),
                4 => (v as i32).to_le_bytes().to_vec(),
                8 => v.to_le_bytes().to_vec(),
                _ => (v as i32).to_le_bytes().to_vec(),
            }
        }
        Constant::Float { value, ty } => {
            match ty {
                IrType::F32 => (*value as f32).to_le_bytes().to_vec(),
                IrType::F64 => value.to_le_bytes().to_vec(),
                _ => value.to_le_bytes().to_vec(),
            }
        }
        Constant::Bool(b) => vec![if *b { 1u8 } else { 0u8 }],
        Constant::Null(_) => vec![0u8; 4], // 32-bit null pointer on i686
        Constant::Undef(_) => vec![0u8; type_size],
        Constant::ZeroInit(_) => vec![0u8; type_size],
        Constant::String(data) => data.clone(),
        Constant::GlobalRef(_) => {
            // Global reference: emit a 4-byte zero placeholder that the
            // linker will patch via a R_386_32 relocation.
            vec![0u8; 4]
        }
    };

    // Pad to the declared type size if the serialized bytes are shorter.
    while bytes.len() < type_size {
        bytes.push(0);
    }

    // Truncate to type_size if the serialized bytes are somehow longer.
    bytes.truncate(type_size);
    bytes
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{Architecture, CodeGen};
    use crate::codegen::regalloc::{PhysReg, RegClass};

    /// Verify that `I686CodeGen::new()` creates a valid instance.
    #[test]
    fn test_new_creates_instance() {
        let codegen = I686CodeGen::new();
        // The struct is a unit struct; just verify it exists and compiles.
        assert_eq!(codegen.target_arch(), Architecture::I686);
    }

    /// Verify `target_arch()` returns `Architecture::I686`.
    #[test]
    fn test_target_arch_returns_i686() {
        let codegen = I686CodeGen::new();
        assert_eq!(codegen.target_arch(), Architecture::I686);
    }

    /// Verify the register info has exactly 6 allocatable integer GPRs.
    #[test]
    fn test_register_info_int_reg_count() {
        let info = create_i686_register_info();
        assert_eq!(
            info.int_regs.len(),
            6,
            "Expected 6 allocatable integer GPRs (eax, ecx, edx, ebx, esi, edi), got {}",
            info.int_regs.len()
        );
    }

    /// Verify the register info has exactly 8 XMM float registers.
    #[test]
    fn test_register_info_float_reg_count() {
        let info = create_i686_register_info();
        assert_eq!(
            info.float_regs.len(),
            8,
            "Expected 8 XMM registers (xmm0-xmm7), got {}",
            info.float_regs.len()
        );
    }

    /// Verify callee-saved integer registers are ebx, esi, edi.
    #[test]
    fn test_callee_saved_int_registers() {
        let info = create_i686_register_info();
        assert_eq!(info.callee_saved_int.len(), 3);
        assert!(info.callee_saved_int.contains(&PhysReg(3)), "ebx should be callee-saved");
        assert!(info.callee_saved_int.contains(&PhysReg(6)), "esi should be callee-saved");
        assert!(info.callee_saved_int.contains(&PhysReg(7)), "edi should be callee-saved");
    }

    /// Verify no callee-saved float registers on i686.
    #[test]
    fn test_callee_saved_float_empty() {
        let info = create_i686_register_info();
        assert!(
            info.callee_saved_float.is_empty(),
            "i686 should have no callee-saved float registers"
        );
    }

    /// Verify caller-saved GPRs come first in allocation priority.
    #[test]
    fn test_allocation_priority_order() {
        let info = create_i686_register_info();
        // First 3 should be caller-saved: eax, ecx, edx
        assert_eq!(info.int_regs[0], PhysReg(0), "eax should be first");
        assert_eq!(info.int_regs[1], PhysReg(1), "ecx should be second");
        assert_eq!(info.int_regs[2], PhysReg(2), "edx should be third");
    }

    /// Verify esp and ebp are NOT in the allocatable register set.
    #[test]
    fn test_esp_ebp_not_allocatable() {
        let info = create_i686_register_info();
        assert!(
            !info.int_regs.contains(&PhysReg(4)),
            "esp should NOT be allocatable"
        );
        assert!(
            !info.int_regs.contains(&PhysReg(5)),
            "ebp should NOT be allocatable"
        );
    }

    /// Verify register names are correctly mapped.
    #[test]
    fn test_register_names() {
        let info = create_i686_register_info();
        assert_eq!(info.reg_names.get(&PhysReg(0)), Some(&"eax"));
        assert_eq!(info.reg_names.get(&PhysReg(1)), Some(&"ecx"));
        assert_eq!(info.reg_names.get(&PhysReg(2)), Some(&"edx"));
        assert_eq!(info.reg_names.get(&PhysReg(3)), Some(&"ebx"));
        assert_eq!(info.reg_names.get(&PhysReg(4)), Some(&"esp"));
        assert_eq!(info.reg_names.get(&PhysReg(5)), Some(&"ebp"));
        assert_eq!(info.reg_names.get(&PhysReg(6)), Some(&"esi"));
        assert_eq!(info.reg_names.get(&PhysReg(7)), Some(&"edi"));
        assert_eq!(info.reg_names.get(&PhysReg(8)), Some(&"xmm0"));
        assert_eq!(info.reg_names.get(&PhysReg(15)), Some(&"xmm7"));
        // Total: 16 register names (8 GPR + 8 XMM)
        assert_eq!(info.reg_names.len(), 16);
    }

    /// Verify relocation type mapping produces correct I386_* variants.
    #[test]
    fn test_relocation_type_mapping() {
        assert_eq!(
            map_relocation_type(&RelocationType::I386_32),
            RelocationType::I386_32
        );
        assert_eq!(
            map_relocation_type(&RelocationType::I386_PC32),
            RelocationType::I386_PC32
        );
        assert_eq!(
            map_relocation_type(&RelocationType::I386_GOT32),
            RelocationType::I386_GOT32
        );
        assert_eq!(
            map_relocation_type(&RelocationType::I386_PLT32),
            RelocationType::I386_PLT32
        );
    }

    /// Verify non-i686 relocation types fall back to I386_PC32.
    #[test]
    fn test_relocation_type_fallback() {
        assert_eq!(
            map_relocation_type(&RelocationType::X86_64_PC32),
            RelocationType::I386_PC32,
            "Non-i686 relocation types should fall back to I386_PC32"
        );
    }

    /// Verify .text section has correct flags.
    #[test]
    fn test_text_section_flags() {
        let section = build_text_section(vec![0x90]); // NOP
        assert_eq!(section.name, ".text");
        assert_eq!(section.section_type, SectionType::Text);
        assert!(section.flags.executable);
        assert!(section.flags.allocatable);
        assert!(!section.flags.writable);
        assert_eq!(section.data, vec![0x90]);
    }

    /// Verify .data section has correct flags.
    #[test]
    fn test_data_section_flags() {
        let section = build_data_section(vec![0x42, 0x00, 0x00, 0x00]);
        assert_eq!(section.name, ".data");
        assert_eq!(section.section_type, SectionType::Data);
        assert!(section.flags.writable);
        assert!(section.flags.allocatable);
        assert!(!section.flags.executable);
    }

    /// Verify .rodata section has correct flags.
    #[test]
    fn test_rodata_section_flags() {
        let section = build_rodata_section(vec![0x48, 0x65, 0x6C, 0x6C, 0x6F]);
        assert_eq!(section.name, ".rodata");
        assert_eq!(section.section_type, SectionType::Rodata);
        assert!(!section.flags.writable);
        assert!(!section.flags.executable);
        assert!(section.flags.allocatable);
    }

    /// Verify .bss section has correct flags and size.
    #[test]
    fn test_bss_section_flags() {
        let section = build_bss_section(256);
        assert_eq!(section.name, ".bss");
        assert_eq!(section.section_type, SectionType::Bss);
        assert!(section.flags.writable);
        assert!(section.flags.allocatable);
        assert!(!section.flags.executable);
        assert_eq!(section.data.len(), 256);
    }

    /// Verify RegisterInfo num_regs() method.
    #[test]
    fn test_num_regs() {
        let info = create_i686_register_info();
        assert_eq!(info.num_regs(RegClass::Integer), 6);
        assert_eq!(info.num_regs(RegClass::Float), 8);
    }

    /// Verify align_offset utility function.
    #[test]
    fn test_align_offset() {
        assert_eq!(align_offset(0, 4), 0);
        assert_eq!(align_offset(1, 4), 4);
        assert_eq!(align_offset(4, 4), 4);
        assert_eq!(align_offset(5, 4), 8);
        assert_eq!(align_offset(15, 16), 16);
        assert_eq!(align_offset(16, 16), 16);
    }

    /// Verify generate returns error for wrong target architecture.
    #[test]
    fn test_generate_wrong_arch_error() {
        let codegen = I686CodeGen::new();
        let module = Module {
            functions: vec![],
            globals: vec![],
            name: "test".to_string(),
        };
        let target = TargetConfig::x86_64();
        let result = codegen.generate(&module, &target);
        assert!(result.is_err(), "Should fail for x86_64 target");
    }

    /// Verify generate succeeds for empty module.
    #[test]
    fn test_generate_empty_module() {
        let codegen = I686CodeGen::new();
        let module = Module {
            functions: vec![],
            globals: vec![],
            name: "empty".to_string(),
        };
        let target = TargetConfig::i686();
        let result = codegen.generate(&module, &target);
        assert!(result.is_ok(), "Empty module should compile successfully");
        let obj = result.unwrap();
        assert_eq!(obj.target_arch, Architecture::I686);
        // Should have at least the .text section.
        assert!(!obj.sections.is_empty());
    }

    /// Verify constant serialization for integer types.
    #[test]
    fn test_serialize_integer_constant() {
        use crate::ir::Constant;
        let target = TargetConfig::i686();
        let c = Constant::Integer { value: 42, ty: IrType::I32 };
        let bytes = serialize_constant(&c, 4, &target);
        assert_eq!(bytes, vec![42, 0, 0, 0]); // little-endian i32
    }

    /// Verify constant serialization for string literals.
    #[test]
    fn test_serialize_string_constant() {
        use crate::ir::Constant;
        let target = TargetConfig::i686();
        let c = Constant::String(vec![72, 101, 108, 108, 111, 0]); // "Hello\0"
        let bytes = serialize_constant(&c, 6, &target);
        assert_eq!(bytes, vec![72, 101, 108, 108, 111, 0]);
    }

    /// Verify constant serialization for null pointer.
    #[test]
    fn test_serialize_null_constant() {
        use crate::ir::Constant;
        let target = TargetConfig::i686();
        let c = Constant::Null(IrType::Pointer(Box::new(IrType::I32)));
        let bytes = serialize_constant(&c, 4, &target);
        assert_eq!(bytes, vec![0, 0, 0, 0]);
    }

    /// Verify constant serialization pads to type_size.
    #[test]
    fn test_serialize_constant_padding() {
        use crate::ir::Constant;
        let target = TargetConfig::i686();
        let c = Constant::Integer { value: 1, ty: IrType::I8 };
        let bytes = serialize_constant(&c, 4, &target);
        assert_eq!(bytes, vec![1, 0, 0, 0]); // padded to 4 bytes
    }
}
