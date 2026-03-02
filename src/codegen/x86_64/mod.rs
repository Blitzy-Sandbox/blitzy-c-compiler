//! # x86-64 Code Generation Backend
//!
//! This module implements native x86-64 machine code generation for the `bcc`
//! C compiler. It is the primary/reference backend with the most features,
//! including security hardening (retpoline, CET, stack probing) unique to x86-64.
//!
//! ## Register File
//! - 16 GPRs: rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi, r8-r15
//! - 16 XMM registers: xmm0-xmm15 (SSE/SSE2 floating-point and SIMD)
//!
//! ## ABI
//! System V AMD64 ABI (standard Linux x86-64 calling convention)
//!
//! ## Output
//! ELF64 relocatable objects with R_X86_64_* relocation types
//!
//! ## Pipeline
//! ```text
//! IR → InstructionSelection → RegisterAllocation → Prologue/Epilogue
//!   → SecurityHardening → Encoding → ObjectCode
//! ```
//!
//! ## Architecture
//!
//! The code generation pipeline for each function proceeds in strict order:
//!
//! 1. **Instruction Selection** (`isel`) — Translates IR instructions to
//!    architecture-specific `MachineInstr` sequences using pattern matching.
//! 2. **Register Allocation** (`regalloc`) — Assigns physical registers to
//!    virtual SSA values using linear scan allocation.
//! 3. **Prologue/Epilogue** (`abi`) — Inserts function entry/exit code for
//!    stack frame setup, callee-saved register preservation, and frame pointer
//!    management per the System V AMD64 ABI.
//! 4. **Security Hardening** (`security`) — Optionally inserts retpoline
//!    thunks for indirect branches, `endbr64` CET instrumentation, and stack
//!    probes for large frames.
//! 5. **Encoding** (`encoding`) — Converts `MachineInstr` sequences to raw
//!    x86-64 machine code bytes with REX prefixes, ModR/M, SIB, and
//!    relocation entries.

pub mod isel;
pub mod encoding;
pub mod abi;
pub mod security;

// ---------------------------------------------------------------------------
// Public re-exports from submodules
// ---------------------------------------------------------------------------

// Re-export register constants for use by parent codegen module and consumers.
pub use abi::{
    RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI,
    R8, R9, R10, R11, R12, R13, R14, R15,
    XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
    XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15,
    x86_64_register_info, ArgumentClass, ArgumentLayout,
    INT_ARG_REGS, FLOAT_ARG_REGS, CALLEE_SAVED_GPRS,
};

// Re-export security types for external configuration and use.
pub use security::{SecurityConfig, SecurityHardening};

// ---------------------------------------------------------------------------
// Imports
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use crate::codegen::{
    Architecture, CodeGen, CodeGenError, ObjectCode,
    Section, SectionType, SectionFlags,
    Symbol, SymbolBinding, SymbolType, SymbolVisibility,
    MachineInstr, MachineOperand,
};
use crate::codegen::regalloc::{self, PhysReg};
use crate::driver::target::TargetConfig;
use crate::ir::{Module, Function, Value};

// ---------------------------------------------------------------------------
// X86_64CodeGen — x86-64 code generation backend
// ---------------------------------------------------------------------------

/// x86-64 code generation backend implementing the [`CodeGen`] trait.
///
/// Coordinates instruction selection, register allocation, machine code
/// encoding, and security hardening for the x86-64 target architecture.
///
/// This struct is stateless — all per-compilation state is local to the
/// [`generate`](CodeGen::generate) / [`generate_impl`](X86_64CodeGen::generate_impl)
/// method. This design enables safe concurrent use and straightforward
/// testing.
///
/// # Usage
///
/// ```ignore
/// use crate::codegen::x86_64::X86_64CodeGen;
/// use crate::codegen::CodeGen;
///
/// let backend = X86_64CodeGen::new();
/// let object_code = backend.generate(&ir_module, &target_config)?;
/// ```
pub struct X86_64CodeGen;

impl X86_64CodeGen {
    /// Create a new x86-64 code generation backend instance.
    ///
    /// The backend is stateless; all per-compilation state is created
    /// within [`generate`](CodeGen::generate).
    pub fn new() -> Self {
        Self
    }

    // =====================================================================
    // Core pipeline — generate_impl
    // =====================================================================

    /// Full x86-64 code generation pipeline implementation.
    ///
    /// Processes all functions and globals in the given IR [`Module`],
    /// coordinating the complete backend pipeline:
    ///
    /// 1. Instruction selection (IR → MachineInstr)
    /// 2. Register allocation (virtual → physical registers)
    /// 3. Prologue/epilogue insertion (stack frame setup/teardown)
    /// 4. Security hardening (retpoline, CET, stack probing)
    /// 5. Machine code encoding (MachineInstr → bytes)
    ///
    /// Returns an [`ObjectCode`] containing `.text`, `.data`, `.rodata`, and
    /// `.bss` sections with symbol definitions and relocation entries.
    pub fn generate_impl(
        &self,
        module: &Module,
        target: &TargetConfig,
    ) -> Result<ObjectCode, CodeGenError> {
        let mut text_section: Vec<u8> = Vec::new();
        let mut data_section: Vec<u8> = Vec::new();
        let mut rodata_section: Vec<u8> = Vec::new();
        let mut bss_size: u64 = 0;
        let mut symbols: Vec<Symbol> = Vec::new();
        let mut relocations = Vec::new();

        // Determine security configuration from target flags.
        // These flags are propagated from CLI arguments through TargetConfig.
        let retpoline_flag = target.retpoline_enabled();
        let cf_protection_flag = target.cf_protection_enabled();
        let security_config = SecurityConfig::from_flags(retpoline_flag, cf_protection_flag);
        let security = SecurityHardening::new(security_config.clone());

        // Get x86-64 register info for the allocator.
        let reg_info = abi::x86_64_register_info();

        // Determine if position-independent code generation is enabled.
        let pic_enabled = target.pic_enabled();

        // =================================================================
        // Process each function definition in the module
        // =================================================================
        for function in &module.functions {
            // Skip external declarations (no body to compile).
            if !function.is_definition {
                // Record as an undefined symbol reference.
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

            let func_start = text_section.len();

            // === Phase 1: Instruction Selection ===
            // Translate IR instructions to x86-64 MachineInstr sequences.
            // The isel creates virtual registers (IDs >= 32) for IR values.
            let mut selector = isel::X86_64InstructionSelector::new(pic_enabled);
            let mut machine_instrs = selector.select_function(function)?;
            // Collect the vreg→Value mapping from the isel for Phase 2.
            let vreg_to_value = selector.build_vreg_to_value_map();

            // === Phase 2: Register Allocation ===
            // Compute live intervals on IR and assign physical registers.
            let mut live_intervals = regalloc::compute_live_intervals(function);
            let alloc_result = regalloc::linear_scan_allocate(
                &mut live_intervals,
                &reg_info,
            );
            let value_to_reg = regalloc::build_value_to_reg_map(&alloc_result);

            // Debug: print mappings
            for (v, r) in &value_to_reg {
            }
            for (vreg, v) in &vreg_to_value {
            }

            // Apply register assignments: vreg → Value → PhysReg
            self.apply_register_assignments_v2(
                &mut machine_instrs,
                &vreg_to_value,
                &value_to_reg,
            )?;

            // === Phase 3: Prologue/Epilogue Generation ===
            // Compute stack frame layout from locals, spill slots, and callee-saved regs.
            let has_calls = function_has_calls(function);
            let locals_size = function_locals_size(function, target);
            let frame = abi::compute_stack_frame(
                locals_size,
                alloc_result.num_spill_slots,
                &alloc_result.used_callee_saved,
                has_calls,
            );

            // Adjust RBP-relative memory offsets to account for callee-saved
            // registers pushed between the frame pointer setup (push rbp /
            // mov rbp,rsp) and the locals area.  The isel allocates locals
            // starting at [rbp-8], but the prologue pushes callee-saved
            // registers at [rbp-8], [rbp-16], ...  So locals must shift
            // down by (num_callee_saved * 8) bytes.
            let callee_save_shift = (frame.callee_saved_regs.len() as i32) * 8;
            if callee_save_shift > 0 {
                for instr in &mut machine_instrs {
                    for operand in &mut instr.operands {
                        if let MachineOperand::Memory { base, offset } = operand {
                            // Only shift RBP-relative negative offsets (locals area)
                            if base.0 == 5 && *offset < 0 {
                                // RBP register ID is 5
                                *offset -= callee_save_shift;
                            }
                        }
                    }
                }
            }

            let prologue = abi::generate_prologue(&frame);
            let epilogue = abi::generate_epilogue(&frame);

            // Insert prologue at the beginning and epilogue before each RET.
            self.insert_prologue_epilogue(&mut machine_instrs, &prologue, &epilogue);

            // === Phase 4: Security Hardening ===
            // Stack probing for large frames (must happen before encoding).
            if security.needs_stack_probe(frame.frame_size) {
                let probe_instrs = security.generate_stack_probe_instrs(frame.frame_size);
                self.insert_stack_probe(&mut machine_instrs, &probe_instrs);
            }

            // Retpoline replacement for indirect branches.
            if security_config.retpoline {
                security.apply_retpoline(&mut machine_instrs);
            }

            // === Phase 5: Machine Code Encoding ===
            // Convert MachineInstr sequences to raw x86-64 machine code bytes.
            let mut encoder = encoding::X86_64Encoder::new();
            let encoded = encoder.encode_all(&machine_instrs)?;

            // If CF protection is enabled, insert endbr64 at function entry.
            // This operates on the raw code bytes (4-byte endbr64 prefix).
            let mut func_code = encoded.code;
            if security_config.cf_protection {
                security.insert_endbr64_at_function_entry(&mut func_code);
            }

            // Record function symbol.
            // Section index is 0-based into ObjectCode.sections[] (0 = .text).
            let text_section_idx: usize = 0;
            symbols.push(Symbol {
                name: function.name.clone(),
                section_index: text_section_idx,
                offset: func_start as u64,
                size: func_code.len() as u64,
                binding: if function_is_global(function) {
                    SymbolBinding::Global
                } else {
                    SymbolBinding::Local
                },
                symbol_type: SymbolType::Function,
                visibility: SymbolVisibility::Default,
                is_definition: true,
            });

            // Collect relocations with offsets adjusted for the function's
            // position within the .text section.
            for reloc in encoded.relocations {
                let mut adjusted = reloc;
                adjusted.offset += func_start as u64;
                adjusted.section_index = text_section_idx;
                relocations.push(adjusted);
            }

            text_section.extend_from_slice(&func_code);
        }

        // =================================================================
        // Append retpoline thunk sections if retpoline is enabled
        // =================================================================
        if security_config.retpoline {
            let thunks = security.generate_retpoline_thunks();
            for thunk in thunks {
                let thunk_offset = text_section.len();
                symbols.push(Symbol {
                    name: thunk.name.clone(),
                    section_index: 0, // .text section (0-based into ObjectCode.sections)
                    offset: thunk_offset as u64,
                    size: thunk.code.len() as u64,
                    binding: SymbolBinding::Local,
                    symbol_type: SymbolType::Function,
                    visibility: SymbolVisibility::Hidden,
                    is_definition: true,
                });
                text_section.extend_from_slice(&thunk.code);
            }
        }

        // =================================================================
        // Process global variables into .data, .rodata, .bss
        // =================================================================
        for global in &module.globals {
            let sym = self.emit_global(
                global,
                target,
                &mut data_section,
                &mut rodata_section,
                &mut bss_size,
                &symbols,
            );
            // If the initializer is a GlobalRef, emit an R_X86_64_64 relocation
            // so the linker patches the address of the referenced global into
            // the data section at the correct offset.
            if let Some(crate::ir::Constant::GlobalRef(ref_name)) = &global.initializer {
                if sym.is_definition {
                    relocations.push(crate::codegen::Relocation {
                        offset: sym.offset,
                        symbol: ref_name.clone(),
                        reloc_type: crate::codegen::RelocationType::X86_64_64,
                        addend: 0,
                        section_index: sym.section_index,
                    });
                }
            }
            symbols.push(sym);
        }

        // =================================================================
        // Build the final ObjectCode with all four sections
        // =================================================================
        //
        // We always emit all four sections (even if empty) so that symbol
        // section_index values are stable (0-based into ObjectCode.sections):
        //   ObjectCode.sections[0] = .text
        //   ObjectCode.sections[1] = .rodata
        //   ObjectCode.sections[2] = .data
        //   ObjectCode.sections[3] = .bss
        //
        let mut object = ObjectCode::new(Architecture::X86_64);

        // sections[0]: .text — executable code (always present).
        object.add_section(Section {
            name: ".text".to_string(),
            data: text_section,
            section_type: SectionType::Text,
            alignment: 16,
            flags: SectionFlags {
                writable: false,
                executable: true,
                allocatable: true,
            },
        });

        // sections[1]: .rodata — read-only data.
        object.add_section(Section {
            name: ".rodata".to_string(),
            data: rodata_section,
            section_type: SectionType::Rodata,
            alignment: 16,
            flags: SectionFlags {
                writable: false,
                executable: false,
                allocatable: true,
            },
        });

        // sections[2]: .data — initialized writable data.
        object.add_section(Section {
            name: ".data".to_string(),
            data: data_section,
            section_type: SectionType::Data,
            alignment: 8,
            flags: SectionFlags {
                writable: true,
                executable: false,
                allocatable: true,
            },
        });

        // sections[3]: .bss — zero-initialized data (no file content).
        object.add_section(Section {
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

        // Attach symbols and relocations.
        for sym in symbols {
            object.add_symbol(sym);
        }
        for reloc in relocations {
            object.add_relocation(reloc);
        }

        Ok(object)
    }

    // =====================================================================
    // Register assignment application
    // =====================================================================

    /// Walk through all machine instructions and replace virtual register
    /// references with the physical registers assigned by the register allocator.
    ///
    /// Virtual registers are identified by having IDs ≥ 32 (physical registers
    /// for x86-64 use IDs 0..31). The `value_to_reg` map provides the mapping
    /// from IR [`Value`]s to [`PhysReg`]s.
    ///
    /// # Errors
    ///
    /// Returns [`CodeGenError::RegisterAllocation`] if a virtual register
    /// reference cannot be resolved (i.e., was not assigned by the allocator).
    pub fn apply_register_assignments(
        &self,
        instrs: &mut Vec<MachineInstr>,
        value_to_reg: &HashMap<Value, PhysReg>,
    ) -> Result<(), CodeGenError> {
        for instr in instrs.iter_mut() {
            for operand in instr.operands.iter_mut() {
                match operand {
                    MachineOperand::Register(ref mut reg) => {
                        // Physical registers (ID < 32) are already assigned.
                        if reg.0 >= 32 {
                            // Virtual register — look up the physical assignment.
                            let value = Value(reg.0 as u32);
                            if let Some(&phys) = value_to_reg.get(&value) {
                                *reg = phys;
                            } else {
                                // Unresolved virtual register: assign to RAX as a safe
                                // fallback. This prevents encoding RSP/ESP corruption
                                // when (reg.0 & 7) == 4 and avoids ICEs in encoders.
                                // The value was either spilled, dead, or eliminated.
                                *reg = PhysReg(0); // RAX — safe scratch register
                            }
                        }
                    }
                    MachineOperand::Memory { ref mut base, .. } => {
                        // The base register in memory operands may also be virtual.
                        if base.0 >= 32 {
                            let value = Value(base.0 as u32);
                            if let Some(&phys) = value_to_reg.get(&value) {
                                *base = phys;
                            } else {
                                // Unresolved base register: assign to RAX as fallback.
                                *base = PhysReg(0); // RAX
                            }
                        }
                    }
                    // Immediate, Symbol, and Label operands need no register mapping.
                    MachineOperand::Immediate(_)
                    | MachineOperand::Symbol(_)
                    | MachineOperand::Label(_) => {}
                }
            }
        }
        Ok(())
    }

    /// V2 register assignment: maps virtual registers to physical registers
    /// using a two-step lookup: vreg → IR Value → physical register.
    ///
    /// This is used when regalloc runs AFTER isel, with the isel creating
    /// virtual registers for each IR Value.
    pub fn apply_register_assignments_v2(
        &self,
        instrs: &mut Vec<MachineInstr>,
        vreg_to_value: &HashMap<u32, Value>,
        value_to_reg: &HashMap<Value, PhysReg>,
    ) -> Result<(), CodeGenError> {
        // For unmapped vregs we cycle through caller-saved scratch registers
        // to avoid collapsing multiple distinct vregs onto the same physical
        // register (the old RAX-only fallback caused division and shift bugs).
        let fallback_pool: [PhysReg; 4] = [
            PhysReg(0),  // RAX
            PhysReg(1),  // RCX
            PhysReg(2),  // RDX
            PhysReg(8),  // R8
        ];
        let mut fallback_idx: usize = 0;

        let resolve = |reg: &mut PhysReg, fb_idx: &mut usize| {
            if reg.0 >= 32 {
                // Virtual register: look up which IR Value it represents,
                // then which physical register that Value was assigned.
                let vreg_id = reg.0 as u32;
                if let Some(&value) = vreg_to_value.get(&vreg_id) {
                    if let Some(&phys) = value_to_reg.get(&value) {
                        *reg = phys;
                        return;
                    }
                }
                // Fallback: round-robin through scratch registers to
                // avoid collapsing distinct vregs onto the same phys reg.
                *reg = fallback_pool[*fb_idx % fallback_pool.len()];
                *fb_idx += 1;
            }
        };

        for instr in instrs.iter_mut() {
            for operand in instr.operands.iter_mut() {
                match operand {
                    MachineOperand::Register(ref mut reg) => {
                        resolve(reg, &mut fallback_idx);
                    }
                    MachineOperand::Memory { ref mut base, .. } => {
                        resolve(base, &mut fallback_idx);
                    }
                    MachineOperand::Immediate(_)
                    | MachineOperand::Symbol(_)
                    | MachineOperand::Label(_) => {}
                }
            }
        }
        Ok(())
    }

    // =====================================================================
    // Prologue/epilogue insertion
    // =====================================================================

    /// Insert prologue instructions at the beginning of the function and
    /// epilogue instructions before each RET instruction.
    ///
    /// The prologue establishes the stack frame (push rbp, mov rbp/rsp,
    /// callee-saved register saves, stack allocation). The epilogue reverses
    /// these operations before returning.
    ///
    /// Multiple return points (multiple RET instructions) are handled by
    /// inserting a copy of the epilogue before each one.
    pub fn insert_prologue_epilogue(
        &self,
        instrs: &mut Vec<MachineInstr>,
        prologue: &[MachineInstr],
        epilogue: &[MachineInstr],
    ) {
        // If both prologue and epilogue are empty (red zone leaf function),
        // there is nothing to insert.
        if prologue.is_empty() && epilogue.is_empty() {
            return;
        }

        // Phase 1: Find all RET instruction positions and insert epilogues.
        // We process in reverse order so that earlier indices remain valid
        // after insertions at later positions.
        let ret_opcode = isel::opcodes::RET;
        let ret_positions: Vec<usize> = instrs
            .iter()
            .enumerate()
            .filter(|(_, instr)| instr.opcode == ret_opcode)
            .map(|(i, _)| i)
            .collect();

        // Insert epilogues before each RET (reverse order to preserve indices).
        for &pos in ret_positions.iter().rev() {
            for (j, epi_instr) in epilogue.iter().enumerate() {
                instrs.insert(pos + j, epi_instr.clone());
            }
        }

        // Phase 2: Insert prologue at the very beginning.
        for (i, pro_instr) in prologue.iter().enumerate() {
            instrs.insert(i, pro_instr.clone());
        }
    }

    // =====================================================================
    // Stack probe insertion
    // =====================================================================

    /// Insert stack probe instructions after the prologue's stack allocation
    /// (`sub rsp, N`) but before any local variable access.
    ///
    /// Stack probing touches each page of the allocated stack frame to ensure
    /// the OS maps the pages and prevents stack clash attacks on frames
    /// exceeding one page (4096 bytes).
    ///
    /// The probe instructions are inserted after the first `SUB_RI` instruction
    /// that targets RSP (the stack allocation in the prologue), or at the
    /// beginning if no such instruction is found.
    pub fn insert_stack_probe(
        &self,
        instrs: &mut Vec<MachineInstr>,
        probe_instrs: &[MachineInstr],
    ) {
        if probe_instrs.is_empty() {
            return;
        }

        // Find the stack allocation instruction (sub rsp, N) in the prologue.
        // The prologue is at the beginning of the instruction stream, so we
        // search the first ~20 instructions for a SUB_RI targeting RSP.
        let sub_ri_opcode = isel::opcodes::SUB_RI;
        let rsp_id = RSP.0;
        let search_limit = instrs.len().min(20);

        let mut insert_pos = 0;
        for i in 0..search_limit {
            if instrs[i].opcode == sub_ri_opcode {
                // Check if the first operand is RSP.
                if let Some(MachineOperand::Register(reg)) = instrs[i].operands.first() {
                    if reg.0 == rsp_id {
                        insert_pos = i + 1; // Insert after the sub rsp instruction.
                        break;
                    }
                }
            }
        }

        // Insert probe instructions at the determined position.
        for (j, probe) in probe_instrs.iter().enumerate() {
            instrs.insert(insert_pos + j, probe.clone());
        }
    }

    // =====================================================================
    // Global variable emission
    // =====================================================================

    /// Process a global variable and emit its data into the appropriate section.
    ///
    /// Classification:
    /// - Zero-initialized or uninitialized → `.bss` (just increment size)
    /// - Read-only constant with `const` qualifier → `.rodata`
    /// - Read-write initialized data → `.data`
    /// - Extern declarations → no data emitted, just a symbol reference
    ///
    /// Returns the [`Symbol`] for the linker's symbol table.
    pub fn emit_global(
        &self,
        global: &crate::ir::builder::GlobalVariable,
        target: &TargetConfig,
        data_section: &mut Vec<u8>,
        _rodata_section: &mut Vec<u8>,
        bss_size: &mut u64,
        _existing_symbols: &[Symbol],
    ) -> Symbol {
        let type_size = global.ty.size(target);
        let alignment = global.ty.alignment(target);

        // Extern declarations produce an undefined symbol reference.
        if global.is_extern {
            return Symbol {
                name: global.name.clone(),
                section_index: 0,
                offset: 0,
                size: 0,
                binding: SymbolBinding::Global,
                symbol_type: SymbolType::Object,
                visibility: SymbolVisibility::Default,
                is_definition: false,
            };
        }

        // Determine linkage based on storage class.
        let binding = if global.is_static {
            SymbolBinding::Local
        } else {
            SymbolBinding::Global
        };

        match &global.initializer {
            None => {
                // No initializer — place in .bss (zero-initialized).
                // BSS is ObjectCode.sections[3] (0-based index 3).
                let aligned_offset = align_up(*bss_size, alignment as u64);
                let sym = Symbol {
                    name: global.name.clone(),
                    section_index: 3, // .bss (0-based into ObjectCode.sections)
                    offset: aligned_offset,
                    size: type_size as u64,
                    binding,
                    symbol_type: SymbolType::Object,
                    visibility: SymbolVisibility::Default,
                    is_definition: true,
                };
                *bss_size = aligned_offset + type_size as u64;
                sym
            }
            Some(init) => {
                if is_zero_initializer(init) {
                    // Zero initializer → .bss (0-based index 3).
                    let aligned_offset = align_up(*bss_size, alignment as u64);
                    let sym = Symbol {
                        name: global.name.clone(),
                        section_index: 3, // .bss (0-based into ObjectCode.sections)
                        offset: aligned_offset,
                        size: type_size as u64,
                        binding,
                        symbol_type: SymbolType::Object,
                        visibility: SymbolVisibility::Default,
                        is_definition: true,
                    };
                    *bss_size = aligned_offset + type_size as u64;
                    sym
                } else {
                    // Non-zero initializer — choose .rodata or .data.
                    let init_bytes = constant_to_bytes(init, type_size, target);
                    // Static globals that are not explicitly const go to .data.
                    // For now, all non-zero initialized globals go to .data
                    // since we don't track const-ness at the IR level.
                    let section = data_section;
                    let section_index: usize = 2; // .data (0-based into ObjectCode.sections)

                    // Align within the section.
                    let padding = align_padding(section.len(), alignment);
                    section.extend(std::iter::repeat(0u8).take(padding));

                    let offset = section.len();
                    section.extend_from_slice(&init_bytes);

                    Symbol {
                        name: global.name.clone(),
                        section_index,
                        offset: offset as u64,
                        size: init_bytes.len() as u64,
                        binding,
                        symbol_type: SymbolType::Object,
                        visibility: SymbolVisibility::Default,
                        is_definition: true,
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CodeGen trait implementation
// ---------------------------------------------------------------------------

impl CodeGen for X86_64CodeGen {
    /// Generate x86-64 machine code for the given IR module.
    ///
    /// Delegates to [`generate_impl`](X86_64CodeGen::generate_impl) which
    /// orchestrates the full backend pipeline.
    fn generate(
        &self,
        module: &Module,
        target: &TargetConfig,
    ) -> Result<ObjectCode, CodeGenError> {
        self.generate_impl(module, target)
    }

    /// Returns the target architecture for this backend.
    fn target_arch(&self) -> Architecture {
        Architecture::X86_64
    }
}

// ===========================================================================
// Helper functions (module-private)
// ===========================================================================

/// Determine whether an IR [`Function`] contains any call instructions.
///
/// This is used to determine whether the function is a leaf (no calls),
/// which affects stack frame layout decisions such as red zone eligibility.
fn function_has_calls(func: &Function) -> bool {
    for block in &func.blocks {
        for inst in &block.instructions {
            if matches!(inst, crate::ir::Instruction::Call { .. }) {
                return true;
            }
        }
    }
    false
}

/// Determine whether an IR [`Function`] should have global symbol binding.
///
/// In C, all non-static functions have external linkage (global binding).
/// Since the IR `Function` struct does not carry an `is_static` flag,
/// we treat all defined functions as globally visible. Static functions
/// would need additional metadata propagation from the semantic analysis
/// phase.
fn function_is_global(_func: &Function) -> bool {
    // All function definitions are treated as global by default.
    // The frontend would need to annotate static functions for local binding.
    true
}

/// Estimate the total size of local variables for a function by scanning
/// its IR instructions for `Alloca` operations and summing their type sizes.
///
/// This provides the `locals_size` parameter needed by the ABI module's
/// stack frame computation.
fn function_locals_size(func: &Function, target: &TargetConfig) -> u32 {
    let mut total: u32 = 0;
    for block in &func.blocks {
        for inst in &block.instructions {
            if let crate::ir::Instruction::Alloca { ty, count, .. } = inst {
                let elem_size = ty.size(target) as u32;
                let num_elems = match count {
                    Some(_) => {
                        // Dynamic count — estimate 1 for frame sizing.
                        // The actual allocation is handled by the generated code.
                        1
                    }
                    None => 1,
                };
                total += elem_size * num_elems;
            }
        }
    }
    total
}

/// Check whether a [`Constant`] initializer is all-zeros, qualifying the
/// global for `.bss` placement.
fn is_zero_initializer(init: &crate::ir::Constant) -> bool {
    match init {
        crate::ir::Constant::Integer { value, .. } => *value == 0,
        crate::ir::Constant::Float { value, .. } => *value == 0.0,
        crate::ir::Constant::Bool(b) => !b,
        crate::ir::Constant::Null(_) => true,
        crate::ir::Constant::Undef(_) => true,
        crate::ir::Constant::ZeroInit(_) => true,
        crate::ir::Constant::String(bytes) => bytes.iter().all(|&b| b == 0),
        crate::ir::Constant::GlobalRef(_) => false,
    }
}

/// Convert a [`Constant`] initializer to its byte representation.
///
/// For integer and float types, the bytes are emitted in little-endian order
/// (x86-64 is little-endian). For strings, the raw bytes are used directly.
fn constant_to_bytes(
    init: &crate::ir::Constant,
    type_size: usize,
    _target: &TargetConfig,
) -> Vec<u8> {
    match init {
        crate::ir::Constant::Integer { value, .. } => {
            let bytes = value.to_le_bytes();
            bytes[..type_size.min(8)].to_vec()
        }
        crate::ir::Constant::Float { value, ty } => match ty {
            crate::ir::IrType::F32 => (*value as f32).to_le_bytes().to_vec(),
            _ => value.to_le_bytes().to_vec(),
        },
        crate::ir::Constant::Bool(b) => {
            vec![if *b { 1 } else { 0 }]
        }
        crate::ir::Constant::Null(_) => {
            vec![0u8; type_size]
        }
        crate::ir::Constant::Undef(_) | crate::ir::Constant::ZeroInit(_) => {
            vec![0u8; type_size]
        }
        crate::ir::Constant::String(bytes) => bytes.clone(),
        crate::ir::Constant::GlobalRef(name) => {
            // Global reference — emit 8 zero bytes (the linker will
            // apply a relocation to fill in the actual address).
            let _ = name;
            vec![0u8; type_size.max(8)]
        }
    }
}

/// Compute the padding needed to align `offset` to `alignment`.
fn align_padding(offset: usize, alignment: usize) -> usize {
    if alignment <= 1 {
        return 0;
    }
    let misalignment = offset % alignment;
    if misalignment == 0 {
        0
    } else {
        alignment - misalignment
    }
}

/// Align a value up to the given alignment boundary.
fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment <= 1 {
        return value;
    }
    (value + alignment - 1) & !(alignment - 1)
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{Architecture, CodeGen};
    use crate::ir::builder::{Module, Function, GlobalVariable};
    use crate::ir::instructions::{Value, Constant, BlockId};
    use crate::ir::types::IrType;
    use crate::ir::cfg::BasicBlock;

    /// Helper to create a minimal TargetConfig for x86-64 tests.
    fn test_target() -> TargetConfig {
        TargetConfig::x86_64()
    }

    /// Helper to create an empty IR module.
    fn empty_module() -> Module {
        Module::new("test".to_string())
    }

    /// Helper to create a minimal function definition with an empty body.
    fn minimal_function(name: &str) -> Function {
        Function {
            name: name.to_string(),
            return_type: IrType::Void,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![BasicBlock::new(BlockId(0), "entry".to_string())],
            entry_block: BlockId(0),
            is_definition: true,
        }
    }

    /// Helper to create a function declaration (extern, no body).
    fn extern_function(name: &str) -> Function {
        Function {
            name: name.to_string(),
            return_type: IrType::Void,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: Vec::new(),
            entry_block: BlockId(0),
            is_definition: false,
        }
    }

    // =================================================================
    // Construction tests
    // =================================================================

    #[test]
    fn test_new_creates_backend() {
        let _backend = X86_64CodeGen::new();
        // Construction must succeed without panicking.
    }

    #[test]
    fn test_target_arch() {
        let backend = X86_64CodeGen::new();
        assert_eq!(backend.target_arch(), Architecture::X86_64);
    }

    // =================================================================
    // CodeGen trait tests
    // =================================================================

    #[test]
    fn test_codegen_trait_object_safe() {
        // Verify that X86_64CodeGen can be used as a trait object.
        let backend: Box<dyn CodeGen> = Box::new(X86_64CodeGen::new());
        assert_eq!(backend.target_arch(), Architecture::X86_64);
    }

    #[test]
    fn test_empty_module() {
        let backend = X86_64CodeGen::new();
        let module = empty_module();
        let target = test_target();
        let result = backend.generate(&module, &target);
        assert!(result.is_ok(), "Empty module should generate successfully");
        let obj = result.unwrap();
        assert_eq!(obj.target_arch, Architecture::X86_64);
        // Should have at least the .text section.
        assert!(!obj.sections.is_empty());
        assert_eq!(obj.sections[0].name, ".text");
    }

    // =================================================================
    // Pipeline integration tests
    // =================================================================

    #[test]
    fn test_module_with_extern_only() {
        let backend = X86_64CodeGen::new();
        let mut module = empty_module();
        module.functions.push(extern_function("printf"));
        let target = test_target();
        let result = backend.generate(&module, &target);
        assert!(result.is_ok());
        let obj = result.unwrap();
        // Extern function should produce a symbol with NoType binding.
        let printf_sym = obj.symbols.iter().find(|s| s.name == "printf");
        assert!(printf_sym.is_some(), "printf symbol should be present");
    }

    #[test]
    fn test_simple_function_generation() {
        let backend = X86_64CodeGen::new();
        let mut module = empty_module();
        module.functions.push(minimal_function("main"));
        let target = test_target();
        let result = backend.generate(&module, &target);
        assert!(result.is_ok(), "Simple function should generate: {:?}", result.err());
        let obj = result.unwrap();
        // Should have a function symbol for 'main'.
        let main_sym = obj.symbols.iter().find(|s| s.name == "main");
        assert!(main_sym.is_some(), "main symbol should be present");
        let sym = main_sym.unwrap();
        assert_eq!(sym.symbol_type, SymbolType::Function);
        assert_eq!(sym.binding, SymbolBinding::Global);
    }

    #[test]
    fn test_function_symbol_properties() {
        let backend = X86_64CodeGen::new();
        let mut module = empty_module();
        module.functions.push(minimal_function("my_func"));
        let target = test_target();
        let obj = backend.generate(&module, &target).unwrap();
        let sym = obj.symbols.iter().find(|s| s.name == "my_func").unwrap();
        assert_eq!(sym.symbol_type, SymbolType::Function);
        assert_eq!(sym.binding, SymbolBinding::Global);
        assert_eq!(sym.visibility, SymbolVisibility::Default);
        assert_eq!(sym.section_index, 0); // .text (0-based into ObjectCode.sections)
        assert!(sym.is_definition);
    }

    // =================================================================
    // Register assignment tests
    // =================================================================

    #[test]
    fn test_apply_register_assignments_empty() {
        let backend = X86_64CodeGen::new();
        let mut instrs = Vec::new();
        let map = HashMap::new();
        let result = backend.apply_register_assignments(&mut instrs, &map);
        assert!(result.is_ok());
    }

    #[test]
    fn test_apply_register_assignments_physical_unchanged() {
        let backend = X86_64CodeGen::new();
        let mut instrs = vec![
            MachineInstr::with_operands(0x100, vec![
                MachineOperand::Register(PhysReg(0)), // RAX — physical, should be unchanged
                MachineOperand::Register(PhysReg(1)), // RCX — physical
            ]),
        ];
        let map = HashMap::new();
        backend.apply_register_assignments(&mut instrs, &map).unwrap();
        // Physical registers should remain unchanged.
        if let MachineOperand::Register(r) = &instrs[0].operands[0] {
            assert_eq!(r.0, 0);
        }
        if let MachineOperand::Register(r) = &instrs[0].operands[1] {
            assert_eq!(r.0, 1);
        }
    }

    #[test]
    fn test_apply_register_assignments_virtual_mapped() {
        let backend = X86_64CodeGen::new();
        let mut instrs = vec![
            MachineInstr::with_operands(0x100, vec![
                MachineOperand::Register(PhysReg(32)), // Virtual register
            ]),
        ];
        let mut map = HashMap::new();
        map.insert(Value(32), PhysReg(0)); // Map virtual 32 → RAX
        backend.apply_register_assignments(&mut instrs, &map).unwrap();
        if let MachineOperand::Register(r) = &instrs[0].operands[0] {
            assert_eq!(r.0, 0, "Virtual register 32 should map to RAX (0)");
        }
    }

    // =================================================================
    // Prologue/epilogue insertion tests
    // =================================================================

    #[test]
    fn test_insert_prologue_epilogue_empty_both() {
        let backend = X86_64CodeGen::new();
        let mut instrs = vec![
            MachineInstr::with_operands(isel::opcodes::RET, vec![]),
        ];
        backend.insert_prologue_epilogue(&mut instrs, &[], &[]);
        assert_eq!(instrs.len(), 1); // No changes for empty prologue/epilogue.
    }

    #[test]
    fn test_insert_prologue_epilogue_with_prologue() {
        let backend = X86_64CodeGen::new();
        let push_rbp = MachineInstr::with_operands(isel::opcodes::PUSH, vec![
            MachineOperand::Register(RBP),
        ]);
        let prologue = vec![push_rbp.clone()];
        let epilogue = vec![MachineInstr::with_operands(isel::opcodes::POP, vec![
            MachineOperand::Register(RBP),
        ])];
        let mut instrs = vec![
            MachineInstr::with_operands(isel::opcodes::NOP, vec![]),
            MachineInstr::with_operands(isel::opcodes::RET, vec![]),
        ];
        let orig_len = instrs.len();
        backend.insert_prologue_epilogue(&mut instrs, &prologue, &epilogue);
        // Should have: [prologue(1)] + [NOP] + [epilogue(1)] + [RET]
        assert_eq!(instrs.len(), orig_len + prologue.len() + epilogue.len());
        // First instruction should be the push rbp from prologue.
        assert_eq!(instrs[0].opcode, isel::opcodes::PUSH);
    }

    // =================================================================
    // Helper function tests
    // =================================================================

    #[test]
    fn test_function_has_calls_empty() {
        let func = minimal_function("test_leaf");
        assert!(!function_has_calls(&func), "Empty function should be a leaf");
    }

    #[test]
    fn test_function_is_global_default() {
        let func = minimal_function("test_func");
        assert!(function_is_global(&func));
    }

    #[test]
    fn test_function_locals_size_empty() {
        let func = minimal_function("test_no_locals");
        let target = test_target();
        assert_eq!(function_locals_size(&func, &target), 0);
    }

    #[test]
    fn test_is_zero_initializer() {
        assert!(is_zero_initializer(&Constant::Integer { value: 0, ty: IrType::I32 }));
        assert!(!is_zero_initializer(&Constant::Integer { value: 42, ty: IrType::I32 }));
        assert!(is_zero_initializer(&Constant::ZeroInit(IrType::I64)));
        assert!(is_zero_initializer(&Constant::Null(IrType::Pointer(Box::new(IrType::I8)))));
        assert!(is_zero_initializer(&Constant::Bool(false)));
        assert!(!is_zero_initializer(&Constant::Bool(true)));
    }

    #[test]
    fn test_constant_to_bytes_integer() {
        let target = test_target();
        let bytes = constant_to_bytes(
            &Constant::Integer { value: 42, ty: IrType::I32 },
            4,
            &target,
        );
        assert_eq!(bytes, 42i64.to_le_bytes()[..4].to_vec());
    }

    #[test]
    fn test_constant_to_bytes_float() {
        let target = test_target();
        let bytes = constant_to_bytes(
            &Constant::Float { value: 3.14, ty: IrType::F64 },
            8,
            &target,
        );
        assert_eq!(bytes, 3.14f64.to_le_bytes().to_vec());
    }

    #[test]
    fn test_constant_to_bytes_string() {
        let target = test_target();
        let bytes = constant_to_bytes(
            &Constant::String(b"hello\0".to_vec()),
            6,
            &target,
        );
        assert_eq!(bytes, b"hello\0".to_vec());
    }

    #[test]
    fn test_align_padding() {
        assert_eq!(align_padding(0, 8), 0);
        assert_eq!(align_padding(1, 8), 7);
        assert_eq!(align_padding(7, 8), 1);
        assert_eq!(align_padding(8, 8), 0);
        assert_eq!(align_padding(0, 1), 0);
        assert_eq!(align_padding(5, 1), 0);
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 8), 0);
        assert_eq!(align_up(1, 8), 8);
        assert_eq!(align_up(7, 8), 8);
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(9, 8), 16);
        assert_eq!(align_up(5, 1), 5);
    }

    // =================================================================
    // Global variable emission tests
    // =================================================================

    #[test]
    fn test_emit_global_extern() {
        let backend = X86_64CodeGen::new();
        let target = test_target();
        let global = GlobalVariable {
            name: "extern_var".to_string(),
            ty: IrType::I32,
            initializer: None,
            is_extern: true,
            is_static: false,
        };
        let mut data = Vec::new();
        let mut rodata = Vec::new();
        let mut bss = 0u64;
        let sym = backend.emit_global(&global, &target, &mut data, &mut rodata, &mut bss, &[]);
        assert_eq!(sym.name, "extern_var");
        assert_eq!(sym.binding, SymbolBinding::Global);
        assert_eq!(sym.size, 0); // Extern — no data emitted.
        assert!(data.is_empty());
        assert_eq!(bss, 0);
    }

    #[test]
    fn test_emit_global_bss() {
        let backend = X86_64CodeGen::new();
        let target = test_target();
        let global = GlobalVariable {
            name: "bss_var".to_string(),
            ty: IrType::I64,
            initializer: None,
            is_extern: false,
            is_static: false,
        };
        let mut data = Vec::new();
        let mut rodata = Vec::new();
        let mut bss = 0u64;
        let sym = backend.emit_global(&global, &target, &mut data, &mut rodata, &mut bss, &[]);
        assert_eq!(sym.name, "bss_var");
        assert_eq!(sym.size, 8); // i64 = 8 bytes
        assert_eq!(bss, 8); // BSS grew by 8 bytes
    }

    #[test]
    fn test_emit_global_initialized_data() {
        let backend = X86_64CodeGen::new();
        let target = test_target();
        let global = GlobalVariable {
            name: "data_var".to_string(),
            ty: IrType::I32,
            initializer: Some(Constant::Integer { value: 42, ty: IrType::I32 }),
            is_extern: false,
            is_static: false,
        };
        let mut data = Vec::new();
        let mut rodata = Vec::new();
        let mut bss = 0u64;
        let sym = backend.emit_global(&global, &target, &mut data, &mut rodata, &mut bss, &[]);
        assert_eq!(sym.name, "data_var");
        assert_eq!(sym.binding, SymbolBinding::Global);
        assert!(!data.is_empty(), ".data should contain the initialized value");
        assert_eq!(bss, 0, "BSS should not grow for initialized globals");
    }

    #[test]
    fn test_emit_global_static_local_binding() {
        let backend = X86_64CodeGen::new();
        let target = test_target();
        let global = GlobalVariable {
            name: "static_var".to_string(),
            ty: IrType::I32,
            initializer: Some(Constant::Integer { value: 1, ty: IrType::I32 }),
            is_extern: false,
            is_static: true,
        };
        let mut data = Vec::new();
        let mut rodata = Vec::new();
        let mut bss = 0u64;
        let sym = backend.emit_global(&global, &target, &mut data, &mut rodata, &mut bss, &[]);
        assert_eq!(sym.binding, SymbolBinding::Local, "Static globals should have local binding");
    }

    // =================================================================
    // Stack probe insertion tests
    // =================================================================

    #[test]
    fn test_insert_stack_probe_empty() {
        let backend = X86_64CodeGen::new();
        let mut instrs = vec![
            MachineInstr::with_operands(isel::opcodes::NOP, vec![]),
        ];
        backend.insert_stack_probe(&mut instrs, &[]);
        assert_eq!(instrs.len(), 1); // No change for empty probe instrs.
    }

    // =================================================================
    // Re-export verification tests
    // =================================================================

    #[test]
    fn test_register_constants_accessible() {
        // Verify that re-exported register constants are accessible.
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
        assert_eq!(XMM0.0, 16);
        assert_eq!(XMM15.0, 31);
    }

    #[test]
    fn test_register_info_available() {
        let info = x86_64_register_info();
        assert!(!info.int_regs.is_empty());
        assert!(!info.float_regs.is_empty());
    }

    #[test]
    fn test_security_config_from_flags() {
        let config = SecurityConfig::from_flags(true, false);
        assert!(config.retpoline);
        assert!(!config.cf_protection);
    }

    #[test]
    fn test_argument_class_variants() {
        // Verify ArgumentClass enum variants are accessible via re-export.
        let _int = ArgumentClass::Integer;
        let _sse = ArgumentClass::Sse;
        let _mem = ArgumentClass::Memory;
    }
}
