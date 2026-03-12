// DWARF v4 Debug Information Generator — Entry Point Module
//
// This module is the top-level coordinator for generating DWARF v4 debug
// information sections across all four target architectures (x86-64, i686,
// AArch64, RISC-V 64). It is conditionally invoked when the `-g` flag is
// specified on the command line.
//
// # Architecture
//
// The module declares and re-exports four submodules:
// - `dwarf`        — Core DWARF v4 structures, LEB128 encoding, abbreviation
//                    tables, string tables, and section builder.
// - `info`         — `.debug_info` section DIE (Debugging Information Entry)
//                    generation for compilation units, subprograms, variables,
//                    parameters, and types.
// - `line_program` — `.debug_line` section line number state machine program
//                    encoding source-to-address mappings.
// - `frame`        — `.debug_frame` section Call Frame Information (CFI) for
//                    stack unwinding support in gdb/lldb.
//
// # Public API
//
// The primary entry point is `DebugInfoGenerator::generate()`, which accepts
// a `CompilationUnitDebugInfo` struct describing the entire compilation unit
// and returns a `DwarfSections` struct containing all 7 DWARF section byte
// vectors ready for inclusion in ELF output by the linker.
//
// # Cross-Cutting Integration
//
// Per the AAP §0.4.1:
// "Debug Info Generator → Linker: DWARF sections (.debug_info, .debug_line,
//  .debug_abbrev, .debug_str, .debug_frame, .debug_aranges, .debug_loc) as
//  byte vectors with associated relocations. Contract: DWARF data conforms
//  to version 4 encoding; section cross-references use proper relocation
//  entries."
//
// # Constraints
//
// - Zero external dependencies: only `std` and internal crate modules.
// - No `unsafe` code: pure coordination and delegation.
// - DWARF v4 compliance: all sections conform to DWARF version 4.
// - Multi-architecture: handles x86-64 (addr=8), i686 (addr=4),
//   AArch64 (addr=8), and RISC-V 64 (addr=8).

// ---------------------------------------------------------------------------
// Submodule Declarations
// ---------------------------------------------------------------------------

/// DWARF v4 core structures: LEB128 encoding, abbreviation tables, string
/// tables, compilation unit headers, address range tables, location lists,
/// section builder, and byte encoding helpers.
pub mod dwarf;

/// `.debug_info` section generator: constructs Debugging Information Entries
/// (DIEs) for compilation units, subprograms, variables, parameters, and
/// types. Serialises the DIE tree into the `.debug_info` byte stream.
pub mod info;

/// `.debug_line` section generator: produces DWARF v4 line number state
/// machine programs mapping machine code addresses to source file/line/column
/// locations for source-level stepping in debuggers.
pub mod line_program;

/// `.debug_frame` section generator: produces Call Frame Information (CFI)
/// with architecture-specific Common Information Entries (CIEs) and
/// per-function Frame Description Entries (FDEs) for stack unwinding.
pub mod frame;

// ---------------------------------------------------------------------------
// Re-exports — public API surface for external consumers
// ---------------------------------------------------------------------------

pub use dwarf::{AbbreviationTable, DwarfSectionBuilder, DwarfSections, StringTable};
pub use frame::FrameInfoEmitter;
pub use info::DebugInfoEmitter;
pub use line_program::LineProgramEmitter;

// ---------------------------------------------------------------------------
// Import from codegen for Architecture enum
// ---------------------------------------------------------------------------

use crate::codegen::Architecture;

// ---------------------------------------------------------------------------
// DWARF v4 Language Constant
// ---------------------------------------------------------------------------

/// C11 language identifier used in the `DW_AT_language` attribute of
/// compilation unit DIEs.
pub const DW_LANG_C11: u16 = 0x001d;

// ===========================================================================
// Public Input Types — Pipeline-Facing API
// ===========================================================================
//
// These types define the interface between the compilation pipeline (driver,
// codegen, sema) and the debug info generator. They are defined here in the
// top-level module so that all consumers can use a single, clean API without
// depending on submodule-specific details.

/// A mapping from a machine code address to a source location.
///
/// Used by the debug info generator to construct `.debug_line` section entries
/// that map instruction addresses back to source file/line/column positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceMapping {
    /// Address in the generated machine code (offset from function start or
    /// absolute address in the `.text` section).
    pub address: u64,
    /// Source file ID (index into the compilation unit's file registry).
    pub file_id: u32,
    /// Source line number (1-based).
    pub line: u32,
    /// Source column number (1-based; 0 means unknown).
    pub column: u32,
    /// Whether this address is the beginning of a statement (recommended
    /// breakpoint position for debuggers).
    pub is_stmt: bool,
}

/// Debug information for a single compiled function.
///
/// Aggregates all debug metadata needed to produce DWARF subprogram DIEs,
/// line number program entries, and frame description entries for one function.
#[derive(Debug, Clone)]
pub struct FunctionDebugInfo {
    /// Function name as it appears in source code.
    pub name: String,
    /// Linkage name if different from the source name (e.g., mangled name).
    pub linkage_name: Option<String>,
    /// Start address of the function's machine code in the `.text` section.
    pub low_pc: u64,
    /// End address (exclusive) of the function's machine code.
    pub high_pc: u64,
    /// Source file ID (index into the compilation unit's file registry).
    pub file_id: u32,
    /// Source line number where the function is defined.
    pub line: u32,
    /// Return type description.
    pub return_type: DebugTypeRef,
    /// Formal parameters of the function.
    pub parameters: Vec<ParameterDebugInfo>,
    /// Local variables declared within the function body.
    pub local_variables: Vec<VariableDebugInfo>,
    /// Source-to-address mappings for instructions within this function.
    pub line_mappings: Vec<SourceMapping>,
    /// Frame unwinding information (CFA rules, callee-saved register saves).
    pub frame_info: FunctionFrameInfo,
}

/// Debug information for a formal function parameter.
#[derive(Debug, Clone)]
pub struct ParameterDebugInfo {
    /// Parameter name as it appears in source code.
    pub name: String,
    /// Type description for this parameter.
    pub type_ref: DebugTypeRef,
    /// Runtime location of this parameter (register, stack offset, etc.).
    pub location: VariableLocation,
}

/// Debug information for a local or global variable.
#[derive(Debug, Clone)]
pub struct VariableDebugInfo {
    /// Variable name as it appears in source code.
    pub name: String,
    /// Type description for this variable.
    pub type_ref: DebugTypeRef,
    /// Runtime location of this variable.
    pub location: VariableLocation,
    /// Start address of the scope in which this variable is valid.
    pub scope_low_pc: u64,
    /// End address (exclusive) of the scope in which this variable is valid.
    pub scope_high_pc: u64,
}

/// Describes the runtime location of a variable or parameter.
///
/// Different variants handle common location description patterns used
/// in DWARF location attributes.
#[derive(Debug, Clone, PartialEq)]
pub enum VariableLocation {
    /// Variable resides in a hardware register (DWARF register number).
    Register(u16),
    /// Variable resides at a fixed byte offset from the frame base pointer.
    FrameOffset(i64),
    /// Variable location described by a raw DWARF expression byte sequence.
    Expression(Vec<u8>),
    /// Variable location varies over the function's address range; the `u32`
    /// is a byte offset into the `.debug_loc` section.
    LocationList(u32),
}

/// A reference to a debug type, used to describe the types of variables,
/// parameters, and return values in the debug info.
///
/// These references are resolved during debug info generation into concrete
/// DWARF type DIEs (DW_TAG_base_type, DW_TAG_pointer_type, etc.).
#[derive(Debug, Clone, PartialEq)]
pub enum DebugTypeRef {
    /// Void type — no return value or typeless pointer target.
    Void,
    /// A fundamental C type identified by kind (int, char, float, etc.).
    BaseType(BaseTypeKind),
    /// Pointer to another type.
    Pointer(Box<DebugTypeRef>),
    /// Array of another type with an optional element count.
    Array(Box<DebugTypeRef>, Option<u64>),
    /// Struct or union type identified by its tag name.
    Struct(String),
    /// Typedef alias identified by its name.
    Typedef(String),
    /// Function type with return type and parameter types.
    Function {
        return_type: Box<DebugTypeRef>,
        param_types: Vec<DebugTypeRef>,
    },
}

/// Fundamental C type kinds used to select the appropriate DWARF base type
/// encoding (`DW_ATE_*`) and byte size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BaseTypeKind {
    /// `void` — used for void return types and untyped pointers.
    Void,
    /// `_Bool` — 1-byte boolean.
    Bool,
    /// `signed char` — 1-byte signed character.
    SignedChar,
    /// `unsigned char` — 1-byte unsigned character.
    UnsignedChar,
    /// `short` — 2-byte signed integer.
    Short,
    /// `unsigned short` — 2-byte unsigned integer.
    UnsignedShort,
    /// `int` — 4-byte signed integer.
    Int,
    /// `unsigned int` — 4-byte unsigned integer.
    UnsignedInt,
    /// `long` — target-width signed integer (4 bytes on i686, 8 on 64-bit).
    Long,
    /// `unsigned long` — target-width unsigned integer.
    UnsignedLong,
    /// `long long` — 8-byte signed integer.
    LongLong,
    /// `unsigned long long` — 8-byte unsigned integer.
    UnsignedLongLong,
    /// `float` — 4-byte IEEE 754 single precision.
    Float,
    /// `double` — 8-byte IEEE 754 double precision.
    Double,
    /// `long double` — extended precision (size varies by target).
    LongDouble,
}

/// Frame unwinding information for a single function, describing how a
/// debugger can reconstruct the call stack at any point within the function.
///
/// This is a simplified representation that the `DebugInfoGenerator` converts
/// into the submodule-specific `frame::FunctionFrameInfo` for CFI emission.
#[derive(Debug, Clone)]
pub struct FunctionFrameInfo {
    /// DWARF register number of the register used as the CFA (Canonical
    /// Frame Address) base. Typically RSP/ESP/SP for the target architecture.
    pub cfa_register: u16,
    /// Signed byte offset from the CFA register to the Canonical Frame Address.
    pub cfa_offset: i64,
    /// List of callee-saved register saves. Each tuple is (DWARF register
    /// number, byte offset from CFA where the register value is stored).
    pub saved_registers: Vec<(u16, i64)>,
}

/// Top-level metadata for a compilation unit's debug information.
///
/// This is the primary input to `DebugInfoGenerator::generate()` and
/// aggregates all information needed to produce the complete set of DWARF
/// sections for one compilation unit.
#[derive(Debug, Clone)]
pub struct CompilationUnitDebugInfo {
    /// Producer identification string (e.g., "bcc 0.1.0").
    pub producer: String,
    /// Source language (DW_LANG_C11 = 0x1d for C11 source code).
    pub language: u16,
    /// Main source file name (e.g., "main.c").
    pub source_file: String,
    /// Compilation directory (absolute path where the compiler was invoked).
    pub comp_dir: String,
    /// Lowest virtual address in this compilation unit's generated code.
    pub low_pc: u64,
    /// Highest virtual address (exclusive) in this compilation unit's code.
    pub high_pc: u64,
    /// Debug information for all functions in this compilation unit.
    pub functions: Vec<FunctionDebugInfo>,
    /// Debug information for global variables in this compilation unit.
    pub global_variables: Vec<VariableDebugInfo>,
    /// Struct/union type definitions in this compilation unit.
    pub struct_defs: Vec<StructDebugDef>,
    /// All source file paths referenced by this CU (for the line program
    /// file table). Index 0 is typically the main source file.
    pub source_files: Vec<String>,
    /// Include directories referenced by source files.
    pub include_directories: Vec<String>,
}

/// Debug information for a struct/union type definition.
#[derive(Debug, Clone)]
pub struct StructDebugDef {
    /// Struct tag name (e.g., "Point").
    pub name: String,
    /// Total byte size of the struct.
    pub byte_size: u64,
    /// Member fields.
    pub members: Vec<StructMemberDebugInfo>,
}

/// Debug information for a single struct/union member field.
#[derive(Debug, Clone)]
pub struct StructMemberDebugInfo {
    /// Member field name.
    pub name: String,
    /// Byte offset within the struct.
    pub byte_offset: u64,
    /// Type of this member.
    pub type_ref: DebugTypeRef,
}

// ===========================================================================
// DebugInfoGenerator — Top-Level Coordinator
// ===========================================================================

/// Orchestrates the generation of all DWARF v4 debug information sections
/// for a compilation unit.
///
/// The generator coordinates the four submodules (`dwarf`, `info`,
/// `line_program`, `frame`) to produce the complete set of 7 DWARF sections:
/// `.debug_info`, `.debug_abbrev`, `.debug_line`, `.debug_str`,
/// `.debug_aranges`, `.debug_frame`, and `.debug_loc`.
///
/// # Usage
///
/// ```ignore
/// use bcc::codegen::Architecture;
/// use bcc::debug::{DebugInfoGenerator, address_size_for_architecture};
///
/// let arch = Architecture::X86_64;
/// let addr_size = address_size_for_architecture(&arch);
/// let generator = DebugInfoGenerator::new(addr_size, arch);
/// let sections = generator.generate(&cu_info);
/// // Pass `sections` to the linker for ELF embedding.
/// ```
pub struct DebugInfoGenerator {
    /// Target address size in bytes (4 for i686, 8 for 64-bit targets).
    address_size: u8,
    /// Target architecture — used for architecture-specific register numbering
    /// in frame info generation and minimum instruction length determination.
    architecture: Architecture,
}

impl DebugInfoGenerator {
    /// Creates a new debug information generator configured for the specified
    /// target architecture.
    ///
    /// # Arguments
    ///
    /// * `address_size` — Target address size in bytes: 4 for i686 (ELF32),
    ///   8 for x86-64, AArch64, and RISC-V 64 (ELF64).
    /// * `architecture` — Target architecture enum variant, used for
    ///   architecture-specific register numbering and instruction parameters.
    pub fn new(address_size: u8, architecture: Architecture) -> Self {
        Self {
            address_size,
            architecture,
        }
    }

    /// Generates all DWARF v4 debug information sections for a compilation unit.
    ///
    /// This is the main entry point called by the driver/pipeline when the
    /// `-g` flag is specified. It orchestrates all four submodules to produce
    /// the complete set of 7 DWARF sections.
    ///
    /// # Arguments
    ///
    /// * `cu_info` — Complete debug information for the compilation unit,
    ///   including functions, variables, source files, and type information.
    ///
    /// # Returns
    ///
    /// A `DwarfSections` struct containing all 7 DWARF section byte vectors
    /// ready for inclusion in ELF output by the linker.
    ///
    /// # Generation Sequence
    ///
    /// 1. Create shared abbreviation table and string table.
    /// 2. Convert high-level types to submodule-specific input structures.
    /// 3. Generate `.debug_info` via `DebugInfoEmitter`.
    /// 4. Generate `.debug_line` via `LineProgramEmitter`.
    /// 5. Generate `.debug_frame` via `FrameInfoEmitter`.
    /// 6. Generate `.debug_aranges` from function address ranges.
    /// 7. Generate `.debug_loc` from variable location lists.
    /// 8. Serialise abbreviation table into `.debug_abbrev`.
    /// 9. Serialise string table into `.debug_str`.
    /// 10. Return all sections as `DwarfSections`.
    pub fn generate(&self, cu_info: &CompilationUnitDebugInfo) -> DwarfSections {
        // --- Phase A: Initialise shared tables ---
        let mut abbrev_table = AbbreviationTable::new();
        let mut string_table = StringTable::new();

        // --- Phase B: Build submodule input structures ---

        // B1: Compile unit info for info.rs
        let compile_unit_info = self.build_compile_unit_info(cu_info);

        // B2: Function debug info in info.rs format
        let info_functions = self.build_info_functions(cu_info);

        // B3: Build basic type DIEs for referenced types
        let type_dies = self.build_type_dies(cu_info);

        // --- Phase C: Generate .debug_info ---
        let info_emitter = DebugInfoEmitter::new(self.address_size);
        let debug_info_bytes = info_emitter.emit_compilation_unit(
            &compile_unit_info,
            &info_functions,
            &type_dies,
            &mut abbrev_table,
            &mut string_table,
        );

        // --- Phase D: Generate .debug_line ---
        let min_insn_len = min_instruction_length_for_architecture(&self.architecture);
        let line_emitter = LineProgramEmitter::new(self.address_size, min_insn_len);
        let line_mappings = self.build_line_mappings(cu_info);
        let source_map = self.build_source_map(cu_info);
        let debug_line_bytes = line_emitter.emit(&line_mappings, &source_map);

        // --- Phase E: Generate .debug_frame ---
        let frame_emitter = FrameInfoEmitter::new(self.architecture, self.address_size);
        let frame_functions = self.build_frame_functions(cu_info);
        let debug_frame_bytes = frame_emitter.emit(&frame_functions);

        // --- Phase F: Generate .debug_aranges ---
        let aranges_entries = derive_aranges_from(cu_info);
        let debug_aranges_bytes = dwarf::serialize_aranges(&aranges_entries, 0, self.address_size);

        // --- Phase G: Generate .debug_loc ---
        let debug_loc_bytes = generate_location_lists(cu_info, self.address_size);

        // --- Phase H: Finalise ---
        DwarfSections {
            debug_info: debug_info_bytes,
            debug_abbrev: abbrev_table.serialize(),
            debug_line: debug_line_bytes,
            debug_str: string_table.serialize(),
            debug_aranges: debug_aranges_bytes,
            debug_frame: debug_frame_bytes,
            debug_loc: debug_loc_bytes,
        }
    }

    // -----------------------------------------------------------------------
    // Private conversion helpers — bridge public API types to submodule types
    // -----------------------------------------------------------------------

    /// Builds the `info::CompileUnitInfo` structure from our public API type.
    fn build_compile_unit_info(&self, cu_info: &CompilationUnitDebugInfo) -> info::CompileUnitInfo {
        info::CompileUnitInfo {
            file_name: cu_info.source_file.clone(),
            comp_dir: cu_info.comp_dir.clone(),
            low_pc: cu_info.low_pc,
            high_pc: cu_info.high_pc,
            // Line program offset is 0 because we generate it in the same pass.
            line_program_offset: 0,
        }
    }

    /// Converts our public `FunctionDebugInfo` into the `info` submodule's
    /// `FunctionDebugInfo` format, which uses CU-relative type offsets and
    /// raw DWARF expressions rather than our high-level type references.
    fn build_info_functions(
        &self,
        cu_info: &CompilationUnitDebugInfo,
    ) -> Vec<info::FunctionDebugInfo> {
        cu_info
            .functions
            .iter()
            .map(|func| {
                let params: Vec<info::ParamDebugInfo> = func
                    .parameters
                    .iter()
                    .map(|p| info::ParamDebugInfo {
                        name: p.name.clone(),
                        type_offset: 0, // Placeholder — resolved by type DIE emission
                        location_expr: self.encode_variable_location(&p.location),
                    })
                    .collect();

                let locals: Vec<info::VariableDebugInfo> = func
                    .local_variables
                    .iter()
                    .map(|v| info::VariableDebugInfo {
                        name: v.name.clone(),
                        type_offset: 0, // Placeholder — resolved by type DIE emission
                        location_expr: self.encode_variable_location(&v.location),
                        decl_file: func.file_id,
                        decl_line: func.line,
                    })
                    .collect();

                // Build frame base expression for the function based on CFA register.
                let frame_base_expr = build_frame_base_expr(self.architecture);

                info::FunctionDebugInfo {
                    name: func.name.clone(),
                    low_pc: func.low_pc,
                    high_pc: func.high_pc,
                    return_type_offset: if matches!(func.return_type, DebugTypeRef::Void) {
                        None
                    } else {
                        Some(0) // Placeholder — resolved by type DIE emission
                    },
                    is_external: true, // Assume external linkage for top-level functions
                    decl_file: func.file_id,
                    decl_line: func.line,
                    frame_base_expr,
                    params,
                    locals,
                }
            })
            .collect()
    }

    /// Builds pre-constructed type DIEs for all types referenced in the CU.
    ///
    /// This scans all functions, parameters, variables, and return types to
    /// collect unique type references, then constructs the corresponding
    /// DWARF type DIEs.
    fn build_type_dies(&self, cu_info: &CompilationUnitDebugInfo) -> Vec<info::DebugInfoEntry> {
        let mut type_dies = Vec::new();

        // Collect all unique base types referenced in the CU.
        let mut seen_base_types = Vec::new();
        self.collect_base_types_from_cu(cu_info, &mut seen_base_types);

        for kind in &seen_base_types {
            let (name, byte_size, encoding) = base_type_dwarf_params(*kind, self.address_size);
            type_dies.push(info::build_base_type_die(name, byte_size, encoding));
        }

        // Collect all unique composite types (pointer, array, struct, typedef)
        // referenced in the CU, and emit DIEs for them.
        let mut seen_composite_keys: Vec<String> = Vec::new();
        let mut composite_refs: Vec<DebugTypeRef> = Vec::new();
        self.collect_composite_types_from_cu(
            cu_info,
            &mut seen_composite_keys,
            &mut composite_refs,
        );

        for type_ref in &composite_refs {
            match type_ref {
                DebugTypeRef::Pointer(_) => {
                    // DW_TAG_pointer_type with pointee offset (use 0 as placeholder).
                    type_dies.push(info::build_pointer_type_die(0, self.address_size));
                }
                DebugTypeRef::Array(_, count) => {
                    // DW_TAG_array_type with element offset (use 0 as placeholder).
                    type_dies.push(info::build_array_type_die(0, *count));
                }
                DebugTypeRef::Struct(name) => {
                    // DW_TAG_structure_type with member DIEs.
                    // Find struct info from CU if available.
                    let members = self.find_struct_members(cu_info, name);
                    type_dies.push(info::build_structure_type_die(
                        name, 0, // byte_size placeholder
                        members,
                    ));
                }
                DebugTypeRef::Typedef(name) => {
                    // DW_TAG_typedef with type offset (use 0 as placeholder).
                    type_dies.push(info::build_typedef_die(name, 0));
                }
                _ => {}
            }
        }

        type_dies
    }

    /// Collects all unique composite type references (Pointer, Array, Struct, Typedef)
    /// from a compilation unit.
    fn collect_composite_types_from_cu(
        &self,
        cu_info: &CompilationUnitDebugInfo,
        keys: &mut Vec<String>,
        refs: &mut Vec<DebugTypeRef>,
    ) {
        for func in &cu_info.functions {
            self.collect_composite_from_type_ref(&func.return_type, keys, refs);
            for param in &func.parameters {
                self.collect_composite_from_type_ref(&param.type_ref, keys, refs);
            }
            for var in &func.local_variables {
                self.collect_composite_from_type_ref(&var.type_ref, keys, refs);
            }
        }
        for var in &cu_info.global_variables {
            self.collect_composite_from_type_ref(&var.type_ref, keys, refs);
        }
    }

    /// Recursively extracts composite types from a DebugTypeRef.
    fn collect_composite_from_type_ref(
        &self,
        type_ref: &DebugTypeRef,
        keys: &mut Vec<String>,
        refs: &mut Vec<DebugTypeRef>,
    ) {
        match type_ref {
            DebugTypeRef::Void | DebugTypeRef::BaseType(_) => {}
            DebugTypeRef::Pointer(inner) => {
                let key = format!("ptr:{:?}", inner);
                if !keys.contains(&key) {
                    keys.push(key);
                    refs.push(type_ref.clone());
                }
                self.collect_composite_from_type_ref(inner, keys, refs);
            }
            DebugTypeRef::Array(inner, count) => {
                let key = format!("arr:{:?}:{:?}", inner, count);
                if !keys.contains(&key) {
                    keys.push(key);
                    refs.push(type_ref.clone());
                }
                self.collect_composite_from_type_ref(inner, keys, refs);
            }
            DebugTypeRef::Struct(name) => {
                let key = format!("struct:{}", name);
                if !keys.contains(&key) {
                    keys.push(key);
                    refs.push(type_ref.clone());
                }
            }
            DebugTypeRef::Typedef(name) => {
                let key = format!("typedef:{}", name);
                if !keys.contains(&key) {
                    keys.push(key);
                    refs.push(type_ref.clone());
                }
            }
            DebugTypeRef::Function {
                return_type,
                param_types,
            } => {
                self.collect_composite_from_type_ref(return_type, keys, refs);
                for pt in param_types {
                    self.collect_composite_from_type_ref(pt, keys, refs);
                }
            }
        }
    }

    /// Find struct member information from the CU's struct definitions.
    /// Extracts member information from the StructDebugDef entries passed
    /// through the compilation unit debug info.
    fn find_struct_members(
        &self,
        cu_info: &CompilationUnitDebugInfo,
        struct_name: &str,
    ) -> Vec<info::MemberDebugInfo> {
        // Look for struct definitions that were passed through.
        for def in &cu_info.struct_defs {
            if def.name == struct_name {
                return def
                    .members
                    .iter()
                    .map(|m| {
                        info::MemberDebugInfo {
                            name: m.name.clone(),
                            type_offset: 0, // placeholder — correct offset requires full type table
                            byte_offset: m.byte_offset as u32,
                        }
                    })
                    .collect();
            }
        }
        Vec::new()
    }

    /// Recursively collects all unique `BaseTypeKind` values from a CU.
    fn collect_base_types_from_cu(
        &self,
        cu_info: &CompilationUnitDebugInfo,
        seen: &mut Vec<BaseTypeKind>,
    ) {
        for func in &cu_info.functions {
            self.collect_base_types_from_type_ref(&func.return_type, seen);
            for param in &func.parameters {
                self.collect_base_types_from_type_ref(&param.type_ref, seen);
            }
            for var in &func.local_variables {
                self.collect_base_types_from_type_ref(&var.type_ref, seen);
            }
        }
        for var in &cu_info.global_variables {
            self.collect_base_types_from_type_ref(&var.type_ref, seen);
        }
    }

    /// Recursively extracts `BaseTypeKind` from a `DebugTypeRef`, avoiding
    /// duplicates.
    fn collect_base_types_from_type_ref(
        &self,
        type_ref: &DebugTypeRef,
        seen: &mut Vec<BaseTypeKind>,
    ) {
        match type_ref {
            DebugTypeRef::Void => {}
            DebugTypeRef::BaseType(kind) => {
                if !seen.contains(kind) {
                    seen.push(*kind);
                }
            }
            DebugTypeRef::Pointer(inner) => {
                self.collect_base_types_from_type_ref(inner, seen);
            }
            DebugTypeRef::Array(inner, _) => {
                self.collect_base_types_from_type_ref(inner, seen);
            }
            DebugTypeRef::Struct(_) | DebugTypeRef::Typedef(_) => {}
            DebugTypeRef::Function {
                return_type,
                param_types,
            } => {
                self.collect_base_types_from_type_ref(return_type, seen);
                for pt in param_types {
                    self.collect_base_types_from_type_ref(pt, seen);
                }
            }
        }
    }

    /// Converts line mappings from the public `SourceMapping` type to the
    /// `line_program::LineMappingEntry` type used by the line program emitter.
    fn build_line_mappings(
        &self,
        cu_info: &CompilationUnitDebugInfo,
    ) -> Vec<line_program::LineMappingEntry> {
        let mut mappings = Vec::new();
        for func in &cu_info.functions {
            let num_mappings = func.line_mappings.len();
            for (i, mapping) in func.line_mappings.iter().enumerate() {
                let is_last = i == num_mappings - 1;
                mappings.push(line_program::LineMappingEntry {
                    address: mapping.address,
                    file_id: mapping.file_id,
                    line: mapping.line,
                    column: mapping.column,
                    is_stmt: mapping.is_stmt,
                    is_prologue_end: false,
                    is_end_sequence: is_last,
                });
            }
        }
        mappings
    }

    /// Builds a minimal `SourceMap` for the line program emitter from the
    /// compilation unit's file list.
    fn build_source_map(
        &self,
        cu_info: &CompilationUnitDebugInfo,
    ) -> crate::common::source_map::SourceMap {
        let mut source_map = crate::common::source_map::SourceMap::new();
        for file_path in &cu_info.source_files {
            // Register each source file. The SourceMap assigns FileId values
            // sequentially, matching the order in cu_info.source_files.
            source_map.add_file(std::path::PathBuf::from(file_path), String::new());
        }
        source_map
    }

    /// Converts the public `FunctionFrameInfo` into the frame submodule's
    /// `frame::FunctionFrameInfo` format.
    fn build_frame_functions(
        &self,
        cu_info: &CompilationUnitDebugInfo,
    ) -> Vec<frame::FunctionFrameInfo> {
        cu_info
            .functions
            .iter()
            .map(|func| {
                let code_size = func.high_pc.saturating_sub(func.low_pc);
                frame::FunctionFrameInfo {
                    start_address: func.low_pc,
                    size: code_size,
                    frame_size: func.frame_info.cfa_offset.unsigned_abs(),
                    uses_frame_pointer: is_frame_pointer_register(
                        func.frame_info.cfa_register,
                        self.architecture,
                    ),
                    callee_saved_regs: func.frame_info.saved_registers.clone(),
                    prologue_size: estimate_prologue_size(self.architecture),
                    epilogue_offset: code_size
                        .saturating_sub(estimate_epilogue_size(self.architecture)),
                }
            })
            .collect()
    }

    /// Encodes a `VariableLocation` into a DWARF expression byte sequence.
    fn encode_variable_location(&self, location: &VariableLocation) -> Vec<u8> {
        match location {
            VariableLocation::Register(reg) => {
                let mut expr = Vec::new();
                if *reg < 32 {
                    // DW_OP_reg0..DW_OP_reg31
                    expr.push(dwarf::DW_OP_REG0 + *reg as u8);
                } else {
                    // DW_OP_regx for registers >= 32
                    expr.push(dwarf::DW_OP_REGX);
                    expr.extend_from_slice(&dwarf::encode_uleb128(*reg as u64));
                }
                expr
            }
            VariableLocation::FrameOffset(offset) => {
                let mut expr = Vec::new();
                expr.push(dwarf::DW_OP_FBREG);
                expr.extend_from_slice(&dwarf::encode_sleb128(*offset));
                expr
            }
            VariableLocation::Expression(bytes) => bytes.clone(),
            VariableLocation::LocationList(_) => {
                // Location list references are handled separately in .debug_loc
                // generation; the location attribute in .debug_info uses a
                // DW_FORM_sec_offset pointing into .debug_loc.
                Vec::new()
            }
        }
    }
}

// ===========================================================================
// Helper Functions
// ===========================================================================

/// Returns the DWARF address size for the given target architecture.
///
/// This is a convenience function that can be called by the driver to
/// determine the address size parameter when constructing a
/// `DebugInfoGenerator`.
///
/// # Returns
///
/// * `4` for i686 (32-bit ELF)
/// * `8` for x86-64, AArch64, and RISC-V 64 (64-bit ELF)
pub fn address_size_for_architecture(arch: &Architecture) -> u8 {
    match arch {
        Architecture::X86_64 => 8,
        Architecture::I686 => 4,
        Architecture::Aarch64 => 8,
        Architecture::Riscv64 => 8,
    }
}

/// Derives address range entries from a compilation unit for the
/// `.debug_aranges` section.
///
/// Creates one `AddressRangeEntry` per function, mapping its code region
/// to the compilation unit. Adjacent ranges are not merged for simplicity;
/// each function gets its own entry.
fn derive_aranges_from(cu_info: &CompilationUnitDebugInfo) -> Vec<dwarf::AddressRangeEntry> {
    let mut entries = Vec::new();
    for func in &cu_info.functions {
        let length = func.high_pc.saturating_sub(func.low_pc);
        if length > 0 {
            entries.push(dwarf::AddressRangeEntry {
                start_address: func.low_pc,
                length,
            });
        }
    }

    // If no functions but the CU has an address range, add a single entry.
    if entries.is_empty() && cu_info.high_pc > cu_info.low_pc {
        entries.push(dwarf::AddressRangeEntry {
            start_address: cu_info.low_pc,
            length: cu_info.high_pc - cu_info.low_pc,
        });
    }

    entries
}

/// Generates the `.debug_loc` section bytes from variable location lists.
///
/// Scans all variables in the compilation unit for `VariableLocation::LocationList`
/// references and serialises their location list entries. For variables that
/// do not use location lists, no data is emitted.
fn generate_location_lists(cu_info: &CompilationUnitDebugInfo, address_size: u8) -> Vec<u8> {
    let mut loc_bytes = Vec::new();

    // Scan functions for variables with location list references.
    for func in &cu_info.functions {
        for var in &func.local_variables {
            if let VariableLocation::LocationList(_offset) = &var.location {
                // Generate a simple location list entry covering the variable's scope.
                let entry = dwarf::LocationListEntry {
                    begin_offset: var.scope_low_pc.saturating_sub(func.low_pc),
                    end_offset: var.scope_high_pc.saturating_sub(func.low_pc),
                    expression: Vec::new(), // Empty expression — placeholder for actual location
                };
                loc_bytes.extend_from_slice(&dwarf::serialize_location_list(
                    &[entry],
                    func.low_pc,
                    address_size,
                ));
            }
        }
    }

    // Also check global variables.
    for var in &cu_info.global_variables {
        if let VariableLocation::LocationList(_offset) = &var.location {
            let entry = dwarf::LocationListEntry {
                begin_offset: var.scope_low_pc,
                end_offset: var.scope_high_pc,
                expression: Vec::new(),
            };
            loc_bytes.extend_from_slice(&dwarf::serialize_location_list(&[entry], 0, address_size));
        }
    }

    loc_bytes
}

/// Returns the minimum instruction length in bytes for the given architecture.
///
/// Used to configure the line number program's `minimum_instruction_length`
/// header field, which affects how address advances are factored.
fn min_instruction_length_for_architecture(arch: &Architecture) -> u8 {
    match arch {
        Architecture::X86_64 | Architecture::I686 => 1,
        Architecture::Aarch64 => 4,
        Architecture::Riscv64 => 2,
    }
}

/// Builds a DWARF expression for the frame base of a function, using the
/// architecture-specific frame pointer or stack pointer register.
///
/// # Returns
///
/// A byte sequence encoding a DW_OP_reg* expression for the frame base.
fn build_frame_base_expr(arch: Architecture) -> Vec<u8> {
    let reg = match arch {
        Architecture::X86_64 => frame::x86_64_regs::RBP,
        Architecture::I686 => frame::i686_regs::EBP,
        Architecture::Aarch64 => frame::aarch64_regs::X29,
        Architecture::Riscv64 => frame::riscv64_regs::X8,
    };

    let mut expr = Vec::new();
    if reg < 32 {
        // DW_OP_reg0..DW_OP_reg31
        expr.push(dwarf::DW_OP_REG0 + reg as u8);
    } else {
        // DW_OP_regx for registers >= 32
        expr.push(dwarf::DW_OP_REGX);
        expr.extend_from_slice(&dwarf::encode_uleb128(reg as u64));
    }
    expr
}

/// Determines whether the given DWARF register number corresponds to the
/// frame pointer for the specified architecture.
fn is_frame_pointer_register(reg: u16, arch: Architecture) -> bool {
    match arch {
        Architecture::X86_64 => reg == frame::x86_64_regs::RBP,
        Architecture::I686 => reg == frame::i686_regs::EBP,
        Architecture::Aarch64 => reg == frame::aarch64_regs::X29,
        Architecture::Riscv64 => reg == frame::riscv64_regs::X8,
    }
}

/// Returns a conservative estimate of prologue size in bytes for the given
/// architecture. Used to set the `prologue_size` field when exact information
/// is not available.
fn estimate_prologue_size(arch: Architecture) -> u64 {
    match arch {
        Architecture::X86_64 => 4,  // push rbp; mov rbp, rsp
        Architecture::I686 => 3,    // push ebp; mov ebp, esp
        Architecture::Aarch64 => 8, // stp x29, x30, [sp, #-N]!; mov x29, sp
        Architecture::Riscv64 => 8, // addi sp, sp, -N; sd ra, offset(sp)
    }
}

/// Returns a conservative estimate of epilogue size in bytes for the given
/// architecture.
fn estimate_epilogue_size(arch: Architecture) -> u64 {
    match arch {
        Architecture::X86_64 => 2,  // pop rbp; ret
        Architecture::I686 => 2,    // pop ebp; ret
        Architecture::Aarch64 => 8, // ldp x29, x30, [sp], #N; ret
        Architecture::Riscv64 => 8, // ld ra, offset(sp); addi sp, sp, N; ret
    }
}

/// Maps a `BaseTypeKind` to its DWARF parameters: (name, byte_size, DW_ATE encoding).
///
/// The byte size of `Long` and `UnsignedLong` varies by target: 4 bytes on
/// i686, 8 bytes on 64-bit targets.
fn base_type_dwarf_params(kind: BaseTypeKind, address_size: u8) -> (&'static str, u8, u8) {
    let long_size = if address_size == 4 { 4u8 } else { 8u8 };
    match kind {
        BaseTypeKind::Void => ("void", 0, dwarf::DW_ATE_UNSIGNED),
        BaseTypeKind::Bool => ("_Bool", 1, dwarf::DW_ATE_BOOLEAN),
        BaseTypeKind::SignedChar => ("signed char", 1, dwarf::DW_ATE_SIGNED_CHAR),
        BaseTypeKind::UnsignedChar => ("unsigned char", 1, dwarf::DW_ATE_UNSIGNED_CHAR),
        BaseTypeKind::Short => ("short", 2, dwarf::DW_ATE_SIGNED),
        BaseTypeKind::UnsignedShort => ("unsigned short", 2, dwarf::DW_ATE_UNSIGNED),
        BaseTypeKind::Int => ("int", 4, dwarf::DW_ATE_SIGNED),
        BaseTypeKind::UnsignedInt => ("unsigned int", 4, dwarf::DW_ATE_UNSIGNED),
        BaseTypeKind::Long => ("long", long_size, dwarf::DW_ATE_SIGNED),
        BaseTypeKind::UnsignedLong => ("unsigned long", long_size, dwarf::DW_ATE_UNSIGNED),
        BaseTypeKind::LongLong => ("long long", 8, dwarf::DW_ATE_SIGNED),
        BaseTypeKind::UnsignedLongLong => ("unsigned long long", 8, dwarf::DW_ATE_UNSIGNED),
        BaseTypeKind::Float => ("float", 4, dwarf::DW_ATE_FLOAT),
        BaseTypeKind::Double => ("double", 8, dwarf::DW_ATE_FLOAT),
        BaseTypeKind::LongDouble => ("long double", 16, dwarf::DW_ATE_FLOAT),
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::Architecture;

    // -----------------------------------------------------------------------
    // address_size_for_architecture tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_address_size_x86_64() {
        assert_eq!(address_size_for_architecture(&Architecture::X86_64), 8);
    }

    #[test]
    fn test_address_size_i686() {
        assert_eq!(address_size_for_architecture(&Architecture::I686), 4);
    }

    #[test]
    fn test_address_size_aarch64() {
        assert_eq!(address_size_for_architecture(&Architecture::Aarch64), 8);
    }

    #[test]
    fn test_address_size_riscv64() {
        assert_eq!(address_size_for_architecture(&Architecture::Riscv64), 8);
    }

    // -----------------------------------------------------------------------
    // DebugInfoGenerator construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_generator_construction_x86_64() {
        let gen = DebugInfoGenerator::new(8, Architecture::X86_64);
        assert_eq!(gen.address_size, 8);
    }

    #[test]
    fn test_generator_construction_i686() {
        let gen = DebugInfoGenerator::new(4, Architecture::I686);
        assert_eq!(gen.address_size, 4);
    }

    #[test]
    fn test_generator_construction_aarch64() {
        let gen = DebugInfoGenerator::new(8, Architecture::Aarch64);
        assert_eq!(gen.address_size, 8);
    }

    #[test]
    fn test_generator_construction_riscv64() {
        let gen = DebugInfoGenerator::new(8, Architecture::Riscv64);
        assert_eq!(gen.address_size, 8);
    }

    // -----------------------------------------------------------------------
    // Submodule accessibility
    // -----------------------------------------------------------------------

    #[test]
    fn test_submodule_dwarf_accessible() {
        // Verify we can access types from the dwarf submodule.
        let _sections = DwarfSections {
            debug_info: Vec::new(),
            debug_abbrev: Vec::new(),
            debug_line: Vec::new(),
            debug_str: Vec::new(),
            debug_aranges: Vec::new(),
            debug_frame: Vec::new(),
            debug_loc: Vec::new(),
        };
    }

    #[test]
    fn test_submodule_types_re_exported() {
        // Verify re-exported types are usable.
        let _builder = DwarfSectionBuilder::new(8);
        let _abbrev = AbbreviationTable::new();
        let _strings = StringTable::new();
    }

    // -----------------------------------------------------------------------
    // Helper function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_min_instruction_length_x86() {
        assert_eq!(
            min_instruction_length_for_architecture(&Architecture::X86_64),
            1
        );
        assert_eq!(
            min_instruction_length_for_architecture(&Architecture::I686),
            1
        );
    }

    #[test]
    fn test_min_instruction_length_aarch64() {
        assert_eq!(
            min_instruction_length_for_architecture(&Architecture::Aarch64),
            4
        );
    }

    #[test]
    fn test_min_instruction_length_riscv64() {
        assert_eq!(
            min_instruction_length_for_architecture(&Architecture::Riscv64),
            2
        );
    }

    #[test]
    fn test_base_type_params_int() {
        let (name, size, enc) = base_type_dwarf_params(BaseTypeKind::Int, 8);
        assert_eq!(name, "int");
        assert_eq!(size, 4);
        assert_eq!(enc, dwarf::DW_ATE_SIGNED);
    }

    #[test]
    fn test_base_type_params_long_64bit() {
        let (name, size, enc) = base_type_dwarf_params(BaseTypeKind::Long, 8);
        assert_eq!(name, "long");
        assert_eq!(size, 8);
        assert_eq!(enc, dwarf::DW_ATE_SIGNED);
    }

    #[test]
    fn test_base_type_params_long_32bit() {
        let (name, size, enc) = base_type_dwarf_params(BaseTypeKind::Long, 4);
        assert_eq!(name, "long");
        assert_eq!(size, 4);
        assert_eq!(enc, dwarf::DW_ATE_SIGNED);
    }

    #[test]
    fn test_base_type_params_float() {
        let (name, size, enc) = base_type_dwarf_params(BaseTypeKind::Float, 8);
        assert_eq!(name, "float");
        assert_eq!(size, 4);
        assert_eq!(enc, dwarf::DW_ATE_FLOAT);
    }

    #[test]
    fn test_base_type_params_double() {
        let (name, size, enc) = base_type_dwarf_params(BaseTypeKind::Double, 8);
        assert_eq!(name, "double");
        assert_eq!(size, 8);
        assert_eq!(enc, dwarf::DW_ATE_FLOAT);
    }

    #[test]
    fn test_base_type_params_bool() {
        let (name, size, enc) = base_type_dwarf_params(BaseTypeKind::Bool, 8);
        assert_eq!(name, "_Bool");
        assert_eq!(size, 1);
        assert_eq!(enc, dwarf::DW_ATE_BOOLEAN);
    }

    // -----------------------------------------------------------------------
    // derive_aranges_from tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_derive_aranges_empty_cu() {
        let cu = CompilationUnitDebugInfo {
            producer: "bcc 0.1.0".to_string(),
            language: DW_LANG_C11,
            source_file: "test.c".to_string(),
            comp_dir: "/tmp".to_string(),
            low_pc: 0x1000,
            high_pc: 0x2000,
            functions: Vec::new(),
            global_variables: Vec::new(),
            struct_defs: Vec::new(),
            source_files: vec!["test.c".to_string()],
            include_directories: Vec::new(),
        };
        let entries = derive_aranges_from(&cu);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].start_address, 0x1000);
        assert_eq!(entries[0].length, 0x1000);
    }

    #[test]
    fn test_derive_aranges_with_functions() {
        let cu = CompilationUnitDebugInfo {
            producer: "bcc 0.1.0".to_string(),
            language: DW_LANG_C11,
            source_file: "test.c".to_string(),
            comp_dir: "/tmp".to_string(),
            low_pc: 0x1000,
            high_pc: 0x2000,
            functions: vec![
                make_test_function("main", 0x1000, 0x1080),
                make_test_function("helper", 0x1080, 0x1100),
            ],
            global_variables: Vec::new(),
            struct_defs: Vec::new(),
            source_files: vec!["test.c".to_string()],
            include_directories: Vec::new(),
        };
        let entries = derive_aranges_from(&cu);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].start_address, 0x1000);
        assert_eq!(entries[0].length, 0x80);
        assert_eq!(entries[1].start_address, 0x1080);
        assert_eq!(entries[1].length, 0x80);
    }

    // -----------------------------------------------------------------------
    // VariableLocation encoding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_register_location() {
        let gen = DebugInfoGenerator::new(8, Architecture::X86_64);
        let expr = gen.encode_variable_location(&VariableLocation::Register(5));
        // DW_OP_reg5 = 0x50 + 5 = 0x55
        assert_eq!(expr, vec![0x55]);
    }

    #[test]
    fn test_encode_frame_offset_location() {
        let gen = DebugInfoGenerator::new(8, Architecture::X86_64);
        let expr = gen.encode_variable_location(&VariableLocation::FrameOffset(-8));
        // DW_OP_fbreg (0x91) followed by SLEB128(-8)
        assert_eq!(expr[0], 0x91);
        // SLEB128(-8) = 0x78
        assert_eq!(expr[1], 0x78);
    }

    #[test]
    fn test_encode_expression_location() {
        let gen = DebugInfoGenerator::new(8, Architecture::X86_64);
        let bytes = vec![0x91, 0x00]; // DW_OP_fbreg, 0
        let expr = gen.encode_variable_location(&VariableLocation::Expression(bytes.clone()));
        assert_eq!(expr, bytes);
    }

    #[test]
    fn test_encode_location_list_returns_empty() {
        let gen = DebugInfoGenerator::new(8, Architecture::X86_64);
        let expr = gen.encode_variable_location(&VariableLocation::LocationList(42));
        assert!(expr.is_empty());
    }

    // -----------------------------------------------------------------------
    // End-to-end smoke test
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_minimal_cu_x86_64() {
        let gen = DebugInfoGenerator::new(8, Architecture::X86_64);
        let cu = make_minimal_cu();
        let sections = gen.generate(&cu);

        // .debug_info should be non-empty and start with a valid CU header.
        assert!(
            !sections.debug_info.is_empty(),
            "debug_info should not be empty"
        );
        // DWARF version 4 is at bytes [4..6] of .debug_info (after 4-byte unit_length).
        if sections.debug_info.len() >= 6 {
            let version = u16::from_le_bytes([sections.debug_info[4], sections.debug_info[5]]);
            assert_eq!(version, 4, "DWARF version should be 4");
        }

        // .debug_abbrev should end with 0x00 null terminator.
        assert!(
            !sections.debug_abbrev.is_empty(),
            "debug_abbrev should not be empty"
        );
        assert_eq!(
            *sections.debug_abbrev.last().unwrap(),
            0x00,
            "debug_abbrev should end with null terminator"
        );

        // .debug_str should contain the producer string.
        assert!(
            !sections.debug_str.is_empty(),
            "debug_str should not be empty"
        );
        let str_content = String::from_utf8_lossy(&sections.debug_str);
        assert!(
            str_content.contains("bcc"),
            "debug_str should contain the producer string"
        );

        // .debug_line should be non-empty.
        assert!(
            !sections.debug_line.is_empty(),
            "debug_line should not be empty"
        );

        // .debug_aranges should be non-empty.
        assert!(
            !sections.debug_aranges.is_empty(),
            "debug_aranges should not be empty"
        );

        // .debug_frame should be non-empty (at least CIE).
        assert!(
            !sections.debug_frame.is_empty(),
            "debug_frame should not be empty"
        );
    }

    #[test]
    fn test_generate_minimal_cu_i686() {
        let gen = DebugInfoGenerator::new(4, Architecture::I686);
        let cu = make_minimal_cu();
        let sections = gen.generate(&cu);

        // All sections should be non-empty.
        assert!(!sections.debug_info.is_empty());
        assert!(!sections.debug_abbrev.is_empty());
        assert!(!sections.debug_line.is_empty());
        assert!(!sections.debug_str.is_empty());
        assert!(!sections.debug_aranges.is_empty());
        assert!(!sections.debug_frame.is_empty());

        // DWARF version should be 4.
        if sections.debug_info.len() >= 6 {
            let version = u16::from_le_bytes([sections.debug_info[4], sections.debug_info[5]]);
            assert_eq!(version, 4);
        }
    }

    #[test]
    fn test_generate_multi_arch_address_sizes() {
        // Generate sections for i686 (addr=4) and x86-64 (addr=8).
        let gen_32 = DebugInfoGenerator::new(4, Architecture::I686);
        let gen_64 = DebugInfoGenerator::new(8, Architecture::X86_64);
        let cu = make_minimal_cu();

        let sections_32 = gen_32.generate(&cu);
        let sections_64 = gen_64.generate(&cu);

        // The address size byte in the CU header is at offset 10 (byte 11).
        // .debug_info layout: [0..4] unit_length, [4..6] version,
        //                     [6..10] abbrev_offset, [10] address_size
        if sections_32.debug_info.len() > 10 && sections_64.debug_info.len() > 10 {
            assert_eq!(
                sections_32.debug_info[10], 4,
                "i686 address size should be 4"
            );
            assert_eq!(
                sections_64.debug_info[10], 8,
                "x86-64 address size should be 8"
            );
        }

        // .debug_aranges should differ in address field widths.
        assert_ne!(
            sections_32.debug_aranges.len(),
            sections_64.debug_aranges.len(),
            "32-bit and 64-bit aranges should have different sizes"
        );
    }

    #[test]
    fn test_generate_aarch64() {
        let gen = DebugInfoGenerator::new(8, Architecture::Aarch64);
        let cu = make_minimal_cu();
        let sections = gen.generate(&cu);
        assert!(!sections.debug_info.is_empty());
        assert!(!sections.debug_frame.is_empty());
    }

    #[test]
    fn test_generate_riscv64() {
        let gen = DebugInfoGenerator::new(8, Architecture::Riscv64);
        let cu = make_minimal_cu();
        let sections = gen.generate(&cu);
        assert!(!sections.debug_info.is_empty());
        assert!(!sections.debug_frame.is_empty());
    }

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Creates a minimal compilation unit with one simple function,
    /// equivalent to `int main() { return 0; }`.
    fn make_minimal_cu() -> CompilationUnitDebugInfo {
        CompilationUnitDebugInfo {
            producer: "bcc 0.1.0".to_string(),
            language: DW_LANG_C11,
            source_file: "main.c".to_string(),
            comp_dir: "/home/user/project".to_string(),
            low_pc: 0x401000,
            high_pc: 0x401020,
            functions: vec![make_test_function("main", 0x401000, 0x401020)],
            global_variables: Vec::new(),
            struct_defs: Vec::new(),
            source_files: vec!["main.c".to_string()],
            include_directories: Vec::new(),
        }
    }

    /// Creates a test function with the given name and address range.
    ///
    /// Uses `file_id: 0` because the SourceMap is 0-based (FileId(0) is the
    /// first registered file). The DWARF line program file table is 1-based,
    /// but the mapping from SourceMap FileId → line program file index is
    /// handled internally by `LineProgramEmitter`.
    fn make_test_function(name: &str, low_pc: u64, high_pc: u64) -> FunctionDebugInfo {
        FunctionDebugInfo {
            name: name.to_string(),
            linkage_name: None,
            low_pc,
            high_pc,
            file_id: 0, // 0-based SourceMap index
            line: 1,
            return_type: DebugTypeRef::BaseType(BaseTypeKind::Int),
            parameters: Vec::new(),
            local_variables: Vec::new(),
            line_mappings: vec![
                SourceMapping {
                    address: low_pc,
                    file_id: 0, // 0-based SourceMap index
                    line: 1,
                    column: 1,
                    is_stmt: true,
                },
                SourceMapping {
                    address: low_pc + 8,
                    file_id: 0, // 0-based SourceMap index
                    line: 2,
                    column: 1,
                    is_stmt: true,
                },
            ],
            frame_info: FunctionFrameInfo {
                cfa_register: 7, // RSP for x86-64
                cfa_offset: 8,
                saved_registers: Vec::new(),
            },
        }
    }
}
