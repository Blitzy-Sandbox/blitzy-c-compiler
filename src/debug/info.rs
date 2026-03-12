// Allow dead_code warnings because this module exports public types and
// functions that are consumed by other debug modules (debug/mod.rs) and
// the driver pipeline (driver/pipeline.rs) once wiring is complete.
// Without this attribute, every pub item triggers a warning since no
// intra-crate consumer currently references them outside of tests.
#![allow(dead_code)]

// .debug_info section generator for DWARF v4 debug information.
//
// This module generates Debugging Information Entries (DIEs) that describe
// the structure of the compiled program: compilation units, functions
// (subprograms), variables, parameters, and types. The generated data
// enables source-level debugging in `gdb` and `lldb`.
//
// # Architecture
//
// DIEs are first constructed as an in-memory tree of `DebugInfoEntry` nodes
// via the `build_*` family of functions.  The tree is then serialized into
// the `.debug_info` byte stream by `serialize_die`, which writes ULEB128
// abbreviation codes followed by attribute values encoded per their
// DW_FORM_* type.
//
// The `DebugInfoEmitter` struct provides a high-level entry point that
// coordinates DIE construction, abbreviation registration, string
// interning, and byte serialization for a complete compilation unit.
//
// # Cross-Architecture Support
//
// All four targets (x86-64, i686, AArch64, RISC-V 64) are supported.
// Target-dependent behaviour is parameterised by `address_size` (4 or 8)
// and the `Architecture` enum, which controls frame-base register
// selection in `build_frame_base_expr`.
//
// # Constraints
//
// - Zero external dependencies: only `std` and internal crate modules.
// - No `unsafe` code: pure data structure generation and byte encoding.
// - DWARF v4 compliance: tags, attributes, and forms use standard constants.

use crate::codegen::Architecture;
#[allow(unused_imports)]
use crate::debug::dwarf::{
    // LEB128 encoding utilities
    encode_sleb128_to,
    encode_uleb128_to,
    // Byte writing helpers
    write_address,
    write_string,
    write_u16_le,
    write_u32_le,
    write_u64_le,
    write_u8,
    // Abbreviation and string table types
    AbbreviationEntry,
    AbbreviationTable,
    // Children flag constants
    DW_CHILDREN_no,
    DW_CHILDREN_yes,
    StringTable,
    // Base type encoding constants
    DW_ATE_ADDRESS,
    DW_ATE_BOOLEAN,
    DW_ATE_FLOAT,
    DW_ATE_SIGNED,
    DW_ATE_SIGNED_CHAR,
    DW_ATE_UNSIGNED,
    DW_ATE_UNSIGNED_CHAR,
    // DWARF attribute constants
    DW_AT_BYTE_SIZE,
    DW_AT_COMP_DIR,
    DW_AT_DATA_MEMBER_LOCATION,
    DW_AT_DECL_FILE,
    DW_AT_DECL_LINE,
    DW_AT_ENCODING,
    DW_AT_EXTERNAL,
    DW_AT_FRAME_BASE,
    DW_AT_HIGH_PC,
    DW_AT_LANGUAGE,
    DW_AT_LOCATION,
    DW_AT_LOW_PC,
    DW_AT_NAME,
    DW_AT_PRODUCER,
    DW_AT_PROTOTYPED,
    DW_AT_STMT_LIST,
    DW_AT_TYPE,
    DW_AT_UPPER_BOUND,
    // DWARF form constants
    DW_FORM_ADDR,
    DW_FORM_DATA1,
    DW_FORM_DATA2,
    DW_FORM_DATA4,
    DW_FORM_DATA8,
    DW_FORM_EXPRLOC,
    DW_FORM_FLAG_PRESENT,
    DW_FORM_REF4,
    DW_FORM_SDATA,
    DW_FORM_SEC_OFFSET,
    DW_FORM_STRING,
    DW_FORM_STRP,
    DW_FORM_UDATA,
    // Language constant
    DW_LANG_C11,
    // Expression opcodes
    DW_OP_ADDR,
    DW_OP_FBREG,
    DW_OP_REG0,
    DW_OP_REGX,
    // DWARF tag constants
    DW_TAG_ARRAY_TYPE,
    DW_TAG_BASE_TYPE,
    DW_TAG_COMPILE_UNIT,
    DW_TAG_CONST_TYPE,
    DW_TAG_ENUMERATION_TYPE,
    DW_TAG_ENUMERATOR,
    DW_TAG_FORMAL_PARAMETER,
    DW_TAG_LEXICAL_BLOCK,
    DW_TAG_MEMBER,
    DW_TAG_POINTER_TYPE,
    DW_TAG_STRUCTURE_TYPE,
    DW_TAG_SUBPROGRAM,
    DW_TAG_SUBRANGE_TYPE,
    DW_TAG_TYPEDEF,
    DW_TAG_UNION_TYPE,
    DW_TAG_UNSPECIFIED_PARAMETERS,
    DW_TAG_VARIABLE,
    DW_TAG_VOLATILE_TYPE,
};

// ---------------------------------------------------------------------------
// Additional DWARF constants not defined in dwarf.rs
// ---------------------------------------------------------------------------

/// Non-standard void encoding used in some DWARF producers for void types.
/// While not part of the DWARF standard, this is commonly accepted by
/// debuggers when no encoding attribute is present.
#[allow(dead_code)]
pub const DW_ATE_VOID: u8 = 0x00;

/// Producer identification string embedded in compilation unit DIEs.
const BCC_PRODUCER: &str = "bcc 0.1.0";

// ---------------------------------------------------------------------------
// Attribute Value Representation
// ---------------------------------------------------------------------------

/// Represents the typed value of a single DWARF attribute within a DIE.
///
/// Each variant corresponds to a DW_FORM_* encoding that determines how the
/// value is serialized into the `.debug_info` byte stream.
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    /// Target-width address (DW_FORM_addr). 4 bytes on i686, 8 bytes on
    /// 64-bit targets.
    Addr(u64),
    /// Single unsigned byte (DW_FORM_data1).
    Data1(u8),
    /// Two-byte unsigned value, little-endian (DW_FORM_data2).
    Data2(u16),
    /// Four-byte unsigned value, little-endian (DW_FORM_data4).
    Data4(u32),
    /// Eight-byte unsigned value, little-endian (DW_FORM_data8).
    Data8(u64),
    /// Inline null-terminated string (DW_FORM_string).
    String(String),
    /// Offset into the `.debug_str` section (DW_FORM_strp). The string
    /// itself is stored in the shared string table for deduplication.
    StringOffset(u32),
    /// 4-byte reference to another DIE within the same compilation unit
    /// (DW_FORM_ref4). The offset is relative to the start of the CU.
    Ref(u32),
    /// Unsigned LEB128 encoded value (DW_FORM_udata).
    UData(u64),
    /// Signed LEB128 encoded value (DW_FORM_sdata).
    SData(i64),
    /// Flag whose mere presence indicates `true` (DW_FORM_flag_present).
    /// No data bytes are emitted.
    FlagPresent,
    /// DWARF expression location block (DW_FORM_exprloc). Serialized as
    /// ULEB128(length) followed by the raw expression bytes.
    ExprLoc(Vec<u8>),
    /// Section offset (DW_FORM_sec_offset). A 4-byte offset into another
    /// DWARF section (e.g., `.debug_line`).
    SecOffset(u32),
}

// ---------------------------------------------------------------------------
// DIE Attribute
// ---------------------------------------------------------------------------

/// A single attribute attached to a Debugging Information Entry (DIE).
///
/// Each attribute is a triple of (name, form, value) where `name` is a
/// DW_AT_* constant, `form` is a DW_FORM_* constant, and `value` carries
/// the typed data.
#[derive(Debug, Clone, PartialEq)]
pub struct DieAttribute {
    /// DW_AT_* attribute name constant.
    pub name: u16,
    /// DW_FORM_* encoding form constant.
    pub form: u16,
    /// Typed attribute value.
    pub value: AttributeValue,
}

// ---------------------------------------------------------------------------
// Debugging Information Entry (DIE)
// ---------------------------------------------------------------------------

/// A node in the DWARF Debugging Information Entry tree.
///
/// Each DIE has a tag (DW_TAG_*) indicating what it describes, a list of
/// attributes carrying its properties, and optionally a list of child DIEs
/// forming a subtree (e.g., a subprogram DIE containing parameter and
/// variable DIEs).
#[derive(Debug, Clone, PartialEq)]
pub struct DebugInfoEntry {
    /// DW_TAG_* constant identifying the kind of entity this DIE describes.
    pub tag: u16,
    /// Whether this DIE has child entries. Controls emission of the
    /// children terminator null byte during serialization.
    pub has_children: bool,
    /// Ordered list of attributes for this DIE.
    pub attributes: Vec<DieAttribute>,
    /// Child DIEs nested under this entry.
    pub children: Vec<DebugInfoEntry>,
}

// ---------------------------------------------------------------------------
// Input Data Structures
// ---------------------------------------------------------------------------

/// Information needed to build a `DW_TAG_compile_unit` DIE.
#[derive(Debug, Clone)]
pub struct CompileUnitInfo {
    /// Source file name (e.g., "main.c").
    pub file_name: String,
    /// Compilation directory (absolute path where the compiler was invoked).
    pub comp_dir: String,
    /// Lowest virtual address in this compilation unit's code.
    pub low_pc: u64,
    /// Highest virtual address (exclusive) in this compilation unit's code.
    pub high_pc: u64,
    /// Byte offset into the `.debug_line` section for this CU's line
    /// number program.
    pub line_program_offset: u32,
}

/// Debug information for a function (subprogram).
#[derive(Debug, Clone)]
pub struct FunctionDebugInfo {
    /// Function name as it appears in source.
    pub name: String,
    /// Start address of the function's machine code.
    pub low_pc: u64,
    /// End address (exclusive) of the function's machine code.
    pub high_pc: u64,
    /// Offset of the return type DIE within the CU, or `None` for void.
    pub return_type_offset: Option<u32>,
    /// Whether the function has external linkage.
    pub is_external: bool,
    /// Source file index (1-based, matching `.debug_line` file table).
    pub decl_file: u32,
    /// Source line number where the function is declared.
    pub decl_line: u32,
    /// DWARF expression bytes for the frame base (e.g., `DW_OP_reg6` for
    /// x86-64 rbp).
    pub frame_base_expr: Vec<u8>,
    /// Formal parameters of the function.
    pub params: Vec<ParamDebugInfo>,
    /// Local variables declared within the function body.
    pub locals: Vec<VariableDebugInfo>,
}

/// Debug information for a local or global variable.
#[derive(Debug, Clone)]
pub struct VariableDebugInfo {
    /// Variable name as it appears in source.
    pub name: String,
    /// Offset of the variable's type DIE within the CU.
    pub type_offset: u32,
    /// DWARF location expression describing where the variable resides
    /// (e.g., `DW_OP_fbreg` + offset for stack variables).
    pub location_expr: Vec<u8>,
    /// Source file index (1-based).
    pub decl_file: u32,
    /// Source line number of the declaration.
    pub decl_line: u32,
}

/// Debug information for a formal function parameter.
#[derive(Debug, Clone)]
pub struct ParamDebugInfo {
    /// Parameter name.
    pub name: String,
    /// Offset of the parameter's type DIE within the CU.
    pub type_offset: u32,
    /// DWARF location expression for the parameter.
    pub location_expr: Vec<u8>,
}

/// Debug information for a struct/union member field.
#[derive(Debug, Clone)]
pub struct MemberDebugInfo {
    /// Member field name.
    pub name: String,
    /// Offset of the member's type DIE within the CU.
    pub type_offset: u32,
    /// Byte offset of this member within the containing struct/union.
    pub byte_offset: u32,
}

// ---------------------------------------------------------------------------
// DIE Builder Functions — Compile Unit
// ---------------------------------------------------------------------------

/// Build a `DW_TAG_compile_unit` DIE for a compilation unit.
///
/// The resulting DIE carries producer identification, language, source file
/// name, compilation directory, address range, and a reference to the
/// `.debug_line` section's line number program.
///
/// `has_children` is always `true` because compilation units contain
/// subprogram, variable, and type DIEs.
pub fn build_compile_unit_die(unit_info: &CompileUnitInfo) -> DebugInfoEntry {
    let code_size = unit_info.high_pc.saturating_sub(unit_info.low_pc);

    let attributes = vec![
        DieAttribute {
            name: DW_AT_PRODUCER,
            form: DW_FORM_STRP,
            value: AttributeValue::StringOffset(0), // placeholder — patched by emitter
        },
        DieAttribute {
            name: DW_AT_LANGUAGE,
            form: DW_FORM_DATA2,
            value: AttributeValue::Data2(DW_LANG_C11),
        },
        DieAttribute {
            name: DW_AT_NAME,
            form: DW_FORM_STRP,
            value: AttributeValue::StringOffset(0), // placeholder — patched by emitter
        },
        DieAttribute {
            name: DW_AT_COMP_DIR,
            form: DW_FORM_STRP,
            value: AttributeValue::StringOffset(0), // placeholder — patched by emitter
        },
        DieAttribute {
            name: DW_AT_LOW_PC,
            form: DW_FORM_ADDR,
            value: AttributeValue::Addr(unit_info.low_pc),
        },
        DieAttribute {
            name: DW_AT_HIGH_PC,
            form: DW_FORM_DATA4,
            value: AttributeValue::Data4(code_size as u32),
        },
        DieAttribute {
            name: DW_AT_STMT_LIST,
            form: DW_FORM_SEC_OFFSET,
            value: AttributeValue::SecOffset(unit_info.line_program_offset),
        },
    ];

    DebugInfoEntry {
        tag: DW_TAG_COMPILE_UNIT,
        has_children: true,
        attributes,
        children: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// DIE Builder Functions — Subprogram (Functions)
// ---------------------------------------------------------------------------

/// Build a `DW_TAG_subprogram` DIE for a function.
///
/// The resulting DIE carries the function name, address range, return type
/// reference, linkage flags, prototyped flag, frame base expression, and
/// source location. Child DIEs for parameters and local variables are
/// added automatically from `func_info.params` and `func_info.locals`.
pub fn build_subprogram_die(func_info: &FunctionDebugInfo) -> DebugInfoEntry {
    let code_size = func_info.high_pc.saturating_sub(func_info.low_pc);

    let mut attributes = vec![
        DieAttribute {
            name: DW_AT_NAME,
            form: DW_FORM_STRP,
            value: AttributeValue::StringOffset(0), // placeholder
        },
        DieAttribute {
            name: DW_AT_LOW_PC,
            form: DW_FORM_ADDR,
            value: AttributeValue::Addr(func_info.low_pc),
        },
        DieAttribute {
            name: DW_AT_HIGH_PC,
            form: DW_FORM_DATA4,
            value: AttributeValue::Data4(code_size as u32),
        },
    ];

    // Return type reference — omitted for void functions.
    if let Some(ret_offset) = func_info.return_type_offset {
        attributes.push(DieAttribute {
            name: DW_AT_TYPE,
            form: DW_FORM_REF4,
            value: AttributeValue::Ref(ret_offset),
        });
    }

    // External linkage flag (presence implies true).
    if func_info.is_external {
        attributes.push(DieAttribute {
            name: DW_AT_EXTERNAL,
            form: DW_FORM_FLAG_PRESENT,
            value: AttributeValue::FlagPresent,
        });
    }

    // All functions are prototyped in C11.
    attributes.push(DieAttribute {
        name: DW_AT_PROTOTYPED,
        form: DW_FORM_FLAG_PRESENT,
        value: AttributeValue::FlagPresent,
    });

    // Frame base expression (architecture-specific register).
    if !func_info.frame_base_expr.is_empty() {
        attributes.push(DieAttribute {
            name: DW_AT_FRAME_BASE,
            form: DW_FORM_EXPRLOC,
            value: AttributeValue::ExprLoc(func_info.frame_base_expr.clone()),
        });
    }

    // Source location.
    attributes.push(DieAttribute {
        name: DW_AT_DECL_FILE,
        form: DW_FORM_DATA1,
        value: AttributeValue::Data1(func_info.decl_file as u8),
    });
    attributes.push(DieAttribute {
        name: DW_AT_DECL_LINE,
        form: DW_FORM_DATA4,
        value: AttributeValue::Data4(func_info.decl_line),
    });

    // Build child DIEs for parameters and local variables.
    let mut children = Vec::new();
    for param in &func_info.params {
        children.push(build_formal_parameter_die(param));
    }
    for local in &func_info.locals {
        children.push(build_variable_die(local));
    }

    DebugInfoEntry {
        tag: DW_TAG_SUBPROGRAM,
        has_children: !children.is_empty(),
        attributes,
        children,
    }
}

// ---------------------------------------------------------------------------
// DIE Builder Functions — Variables and Parameters
// ---------------------------------------------------------------------------

/// Build a `DW_TAG_variable` DIE for a local or global variable.
///
/// The resulting DIE carries the variable name, type reference, location
/// expression, and source location.
pub fn build_variable_die(var_info: &VariableDebugInfo) -> DebugInfoEntry {
    let mut attributes = vec![
        DieAttribute {
            name: DW_AT_NAME,
            form: DW_FORM_STRP,
            value: AttributeValue::StringOffset(0), // placeholder
        },
        DieAttribute {
            name: DW_AT_TYPE,
            form: DW_FORM_REF4,
            value: AttributeValue::Ref(var_info.type_offset),
        },
    ];

    // Location expression — only emitted if non-empty.
    if !var_info.location_expr.is_empty() {
        attributes.push(DieAttribute {
            name: DW_AT_LOCATION,
            form: DW_FORM_EXPRLOC,
            value: AttributeValue::ExprLoc(var_info.location_expr.clone()),
        });
    }

    attributes.push(DieAttribute {
        name: DW_AT_DECL_FILE,
        form: DW_FORM_DATA1,
        value: AttributeValue::Data1(var_info.decl_file as u8),
    });
    attributes.push(DieAttribute {
        name: DW_AT_DECL_LINE,
        form: DW_FORM_DATA4,
        value: AttributeValue::Data4(var_info.decl_line),
    });

    DebugInfoEntry {
        tag: DW_TAG_VARIABLE,
        has_children: false,
        attributes,
        children: Vec::new(),
    }
}

/// Build a `DW_TAG_formal_parameter` DIE for a function parameter.
///
/// The resulting DIE carries the parameter name, type reference, and
/// location expression describing where the parameter resides at runtime.
pub fn build_formal_parameter_die(param_info: &ParamDebugInfo) -> DebugInfoEntry {
    let mut attributes = vec![
        DieAttribute {
            name: DW_AT_NAME,
            form: DW_FORM_STRP,
            value: AttributeValue::StringOffset(0), // placeholder
        },
        DieAttribute {
            name: DW_AT_TYPE,
            form: DW_FORM_REF4,
            value: AttributeValue::Ref(param_info.type_offset),
        },
    ];

    if !param_info.location_expr.is_empty() {
        attributes.push(DieAttribute {
            name: DW_AT_LOCATION,
            form: DW_FORM_EXPRLOC,
            value: AttributeValue::ExprLoc(param_info.location_expr.clone()),
        });
    }

    DebugInfoEntry {
        tag: DW_TAG_FORMAL_PARAMETER,
        has_children: false,
        attributes,
        children: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// DIE Builder Functions — Type DIEs
// ---------------------------------------------------------------------------

/// Build a `DW_TAG_base_type` DIE for a fundamental C type.
///
/// # Arguments
/// * `name` — The type's source-level name (e.g., "int", "char", "float").
/// * `byte_size` — The type's size in bytes.
/// * `encoding` — A DW_ATE_* constant describing the type's encoding.
///
/// Common base types and their parameters:
/// - `int` → byte_size=4, encoding=DW_ATE_SIGNED
/// - `unsigned int` → byte_size=4, encoding=DW_ATE_UNSIGNED
/// - `char` → byte_size=1, encoding=DW_ATE_SIGNED_CHAR
/// - `float` → byte_size=4, encoding=DW_ATE_FLOAT
/// - `double` → byte_size=8, encoding=DW_ATE_FLOAT
/// - `_Bool` → byte_size=1, encoding=DW_ATE_BOOLEAN
/// - `long long` → byte_size=8, encoding=DW_ATE_SIGNED
pub fn build_base_type_die(name: &str, byte_size: u8, encoding: u8) -> DebugInfoEntry {
    let attributes = vec![
        DieAttribute {
            name: DW_AT_NAME,
            form: DW_FORM_STRING,
            value: AttributeValue::String(name.to_string()),
        },
        DieAttribute {
            name: DW_AT_BYTE_SIZE,
            form: DW_FORM_DATA1,
            value: AttributeValue::Data1(byte_size),
        },
        DieAttribute {
            name: DW_AT_ENCODING,
            form: DW_FORM_DATA1,
            value: AttributeValue::Data1(encoding),
        },
    ];

    DebugInfoEntry {
        tag: DW_TAG_BASE_TYPE,
        has_children: false,
        attributes,
        children: Vec::new(),
    }
}

/// Build a `DW_TAG_pointer_type` DIE.
///
/// # Arguments
/// * `pointee_offset` — Offset within the CU of the pointee's type DIE.
/// * `pointer_size` — Size of a pointer in bytes (4 for i686, 8 for 64-bit).
pub fn build_pointer_type_die(pointee_offset: u32, pointer_size: u8) -> DebugInfoEntry {
    let attributes = vec![
        DieAttribute {
            name: DW_AT_TYPE,
            form: DW_FORM_REF4,
            value: AttributeValue::Ref(pointee_offset),
        },
        DieAttribute {
            name: DW_AT_BYTE_SIZE,
            form: DW_FORM_DATA1,
            value: AttributeValue::Data1(pointer_size),
        },
    ];

    DebugInfoEntry {
        tag: DW_TAG_POINTER_TYPE,
        has_children: false,
        attributes,
        children: Vec::new(),
    }
}

/// Build a `DW_TAG_structure_type` DIE for a C struct.
///
/// # Arguments
/// * `name` — The struct's tag name.
/// * `byte_size` — Total size of the struct in bytes.
/// * `members` — Field descriptors; each is emitted as a `DW_TAG_member`
///   child DIE.
pub fn build_structure_type_die(
    name: &str,
    byte_size: u32,
    members: Vec<MemberDebugInfo>,
) -> DebugInfoEntry {
    let attributes = vec![
        DieAttribute {
            name: DW_AT_NAME,
            form: DW_FORM_STRING,
            value: AttributeValue::String(name.to_string()),
        },
        DieAttribute {
            name: DW_AT_BYTE_SIZE,
            form: DW_FORM_DATA4,
            value: AttributeValue::Data4(byte_size),
        },
    ];

    let children: Vec<DebugInfoEntry> = members.iter().map(build_member_die).collect();

    DebugInfoEntry {
        tag: DW_TAG_STRUCTURE_TYPE,
        has_children: !children.is_empty(),
        attributes,
        children,
    }
}

/// Build a `DW_TAG_member` DIE for a struct or union field.
///
/// The `DW_AT_data_member_location` attribute encodes the field's byte
/// offset within its containing aggregate.
pub fn build_member_die(member: &MemberDebugInfo) -> DebugInfoEntry {
    let attributes = vec![
        DieAttribute {
            name: DW_AT_NAME,
            form: DW_FORM_STRING,
            value: AttributeValue::String(member.name.clone()),
        },
        DieAttribute {
            name: DW_AT_TYPE,
            form: DW_FORM_REF4,
            value: AttributeValue::Ref(member.type_offset),
        },
        DieAttribute {
            name: DW_AT_DATA_MEMBER_LOCATION,
            form: DW_FORM_DATA4,
            value: AttributeValue::Data4(member.byte_offset),
        },
    ];

    DebugInfoEntry {
        tag: DW_TAG_MEMBER,
        has_children: false,
        attributes,
        children: Vec::new(),
    }
}

/// Build a `DW_TAG_array_type` DIE.
///
/// # Arguments
/// * `element_offset` — Offset within the CU of the element type DIE.
/// * `count` — Optional element count. If `Some(n)`, a `DW_TAG_subrange_type`
///   child is emitted with `DW_AT_upper_bound = n - 1`. If `None`, the
///   array is unbounded (flexible array member or `[]`).
pub fn build_array_type_die(element_offset: u32, count: Option<u64>) -> DebugInfoEntry {
    let attributes = vec![DieAttribute {
        name: DW_AT_TYPE,
        form: DW_FORM_REF4,
        value: AttributeValue::Ref(element_offset),
    }];

    let mut children = Vec::new();
    if let Some(n) = count {
        // Upper bound is count - 1 (zero-based indexing).
        let upper = if n > 0 { n - 1 } else { 0 };
        let subrange_attrs = vec![DieAttribute {
            name: DW_AT_UPPER_BOUND,
            form: DW_FORM_UDATA,
            value: AttributeValue::UData(upper),
        }];
        children.push(DebugInfoEntry {
            tag: DW_TAG_SUBRANGE_TYPE,
            has_children: false,
            attributes: subrange_attrs,
            children: Vec::new(),
        });
    }

    DebugInfoEntry {
        tag: DW_TAG_ARRAY_TYPE,
        has_children: !children.is_empty(),
        attributes,
        children,
    }
}

/// Build a `DW_TAG_typedef` DIE.
///
/// # Arguments
/// * `name` — The typedef alias name.
/// * `type_offset` — Offset within the CU of the underlying type DIE.
pub fn build_typedef_die(name: &str, type_offset: u32) -> DebugInfoEntry {
    let attributes = vec![
        DieAttribute {
            name: DW_AT_NAME,
            form: DW_FORM_STRING,
            value: AttributeValue::String(name.to_string()),
        },
        DieAttribute {
            name: DW_AT_TYPE,
            form: DW_FORM_REF4,
            value: AttributeValue::Ref(type_offset),
        },
    ];

    DebugInfoEntry {
        tag: DW_TAG_TYPEDEF,
        has_children: false,
        attributes,
        children: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// DIE Serialization
// ---------------------------------------------------------------------------

/// Serialize a single DIE and its children into the `.debug_info` byte stream.
///
/// This function writes the abbreviation code (ULEB128), followed by each
/// attribute value encoded per its DW_FORM_* type, then recursively
/// serializes all children. If the DIE has children, a null terminator byte
/// (0x00) is appended after the last child to mark the end of the children
/// list per DWARF v4 §7.5.2.
///
/// # Arguments
/// * `die` — The DIE to serialize.
/// * `abbrev_code` — The abbreviation code assigned to this DIE's shape.
/// * `buffer` — The output byte buffer (`.debug_info` section data).
/// * `address_size` — Target address size in bytes (4 or 8).
/// * `abbrev_table` — The abbreviation table for looking up child DIE codes.
///
/// # Recursive Children
///
/// For each child DIE, this function looks up or creates an abbreviation
/// entry in the provided table and recursively serializes the child.
pub fn serialize_die(
    die: &DebugInfoEntry,
    abbrev_code: u32,
    buffer: &mut Vec<u8>,
    address_size: u8,
    abbrev_table: &mut AbbreviationTable,
) {
    // Write abbreviation code.
    encode_uleb128_to(buffer, abbrev_code as u64);

    // Write each attribute value according to its form.
    for attr in &die.attributes {
        serialize_attribute_value(&attr.value, attr.form, buffer, address_size);
    }

    // Recursively serialize children.
    if die.has_children {
        for child in &die.children {
            let child_abbrev_attrs: Vec<(u16, u16)> =
                child.attributes.iter().map(|a| (a.name, a.form)).collect();
            let child_code =
                abbrev_table.add_abbreviation(child.tag, child.has_children, child_abbrev_attrs);
            serialize_die(child, child_code, buffer, address_size, abbrev_table);
        }
        // Null terminator marking end of children list.
        buffer.push(0x00);
    }
}

/// Serialize a single attribute value according to its DW_FORM_* encoding.
fn serialize_attribute_value(
    value: &AttributeValue,
    form: u16,
    buffer: &mut Vec<u8>,
    address_size: u8,
) {
    match (form, value) {
        (DW_FORM_ADDR, AttributeValue::Addr(addr)) => {
            write_address(buffer, *addr, address_size);
        }
        (DW_FORM_DATA1, AttributeValue::Data1(v)) => {
            write_u8(buffer, *v);
        }
        (DW_FORM_DATA2, AttributeValue::Data2(v)) => {
            write_u16_le(buffer, *v);
        }
        (DW_FORM_DATA4, AttributeValue::Data4(v)) => {
            write_u32_le(buffer, *v);
        }
        (DW_FORM_DATA8, AttributeValue::Data8(v)) => {
            write_u64_le(buffer, *v);
        }
        (DW_FORM_STRING, AttributeValue::String(s)) => {
            write_string(buffer, s);
        }
        (DW_FORM_STRP, AttributeValue::StringOffset(offset)) => {
            write_u32_le(buffer, *offset);
        }
        (DW_FORM_REF4, AttributeValue::Ref(offset)) => {
            write_u32_le(buffer, *offset);
        }
        (DW_FORM_UDATA, AttributeValue::UData(v)) => {
            encode_uleb128_to(buffer, *v);
        }
        (DW_FORM_SDATA, AttributeValue::SData(v)) => {
            encode_sleb128_to(buffer, *v);
        }
        (DW_FORM_FLAG_PRESENT, AttributeValue::FlagPresent) => {
            // No data emitted — presence of the attribute implies true.
        }
        (DW_FORM_EXPRLOC, AttributeValue::ExprLoc(expr)) => {
            // ULEB128(length) followed by the expression bytes.
            encode_uleb128_to(buffer, expr.len() as u64);
            buffer.extend_from_slice(expr);
        }
        (DW_FORM_SEC_OFFSET, AttributeValue::SecOffset(offset)) => {
            write_u32_le(buffer, *offset);
        }
        // Fallback: encode based on the value variant regardless of form mismatch.
        // This ensures robustness if a DIE builder uses a non-standard form/value
        // combination. The value is serialized using the most natural encoding.
        (_, AttributeValue::Addr(addr)) => {
            write_address(buffer, *addr, address_size);
        }
        (_, AttributeValue::Data1(v)) => {
            write_u8(buffer, *v);
        }
        (_, AttributeValue::Data2(v)) => {
            write_u16_le(buffer, *v);
        }
        (_, AttributeValue::Data4(v)) => {
            write_u32_le(buffer, *v);
        }
        (_, AttributeValue::Data8(v)) => {
            write_u64_le(buffer, *v);
        }
        (_, AttributeValue::String(s)) => {
            write_string(buffer, s);
        }
        (_, AttributeValue::StringOffset(offset)) => {
            write_u32_le(buffer, *offset);
        }
        (_, AttributeValue::Ref(offset)) => {
            write_u32_le(buffer, *offset);
        }
        (_, AttributeValue::UData(v)) => {
            encode_uleb128_to(buffer, *v);
        }
        (_, AttributeValue::SData(v)) => {
            encode_sleb128_to(buffer, *v);
        }
        (_, AttributeValue::FlagPresent) => {
            // No data.
        }
        (_, AttributeValue::ExprLoc(expr)) => {
            encode_uleb128_to(buffer, expr.len() as u64);
            buffer.extend_from_slice(expr);
        }
        (_, AttributeValue::SecOffset(offset)) => {
            write_u32_le(buffer, *offset);
        }
    }
}

// ---------------------------------------------------------------------------
// Location Expression Helpers
// ---------------------------------------------------------------------------

/// Build a DWARF location expression for a variable at a signed offset
/// from the frame base register.
///
/// Encodes `DW_OP_fbreg` followed by SLEB128(offset). This is the standard
/// location description for stack-allocated local variables and parameters
/// whose address is computed as `frame_base + offset`.
///
/// # Arguments
/// * `offset` — Signed byte offset from the frame base.
pub fn build_fbreg_location(offset: i64) -> Vec<u8> {
    let mut expr = Vec::new();
    expr.push(DW_OP_FBREG);
    encode_sleb128_to(&mut expr, offset);
    expr
}

/// Build a DWARF location expression for a variable at an absolute address.
///
/// Encodes `DW_OP_addr` followed by the target-width address. Used for
/// global variables whose address is fixed in the final binary.
///
/// # Arguments
/// * `address` — The absolute virtual address.
/// * `addr_size` — Target address size in bytes (4 or 8).
pub fn build_addr_location(address: u64, addr_size: u8) -> Vec<u8> {
    let mut expr = Vec::new();
    expr.push(DW_OP_ADDR);
    write_address(&mut expr, address, addr_size);
    expr
}

/// Build a DWARF location expression indicating a variable resides in a
/// specific register.
///
/// For register numbers 0–31, encodes `DW_OP_reg0 + reg_num` as a single
/// byte. For register numbers 32 and above, encodes `DW_OP_regx` followed
/// by ULEB128(reg_num).
///
/// # Arguments
/// * `reg_num` — Architecture-specific DWARF register number.
pub fn build_reg_location(reg_num: u16) -> Vec<u8> {
    let mut expr = Vec::new();
    if reg_num <= 31 {
        // DW_OP_reg0 through DW_OP_reg31 are consecutive single-byte opcodes.
        expr.push(DW_OP_REG0 + reg_num as u8);
    } else {
        // Extended register encoding.
        expr.push(DW_OP_REGX);
        encode_uleb128_to(&mut expr, reg_num as u64);
    }
    expr
}

/// Build the frame base DWARF expression for the given target architecture.
///
/// Each architecture uses a different register as its canonical frame
/// pointer:
/// - **x86-64**: `DW_OP_reg6` (rbp — DWARF register 6)
/// - **i686**: `DW_OP_reg5` (ebp — DWARF register 5)
/// - **AArch64**: `DW_OP_reg29` (x29/fp — DWARF register 29)
/// - **RISC-V 64**: `DW_OP_reg8` (s0/fp — DWARF register 8)
///
/// The returned expression is suitable for use as the `DW_AT_frame_base`
/// attribute value on a `DW_TAG_subprogram` DIE.
pub fn build_frame_base_expr(arch: Architecture) -> Vec<u8> {
    match arch {
        Architecture::X86_64 => build_reg_location(6),   // rbp
        Architecture::I686 => build_reg_location(5),     // ebp
        Architecture::Aarch64 => build_reg_location(29), // x29/fp
        Architecture::Riscv64 => build_reg_location(8),  // s0/fp
    }
}

// ---------------------------------------------------------------------------
// DebugInfoEmitter — High-Level Compilation Unit Emitter
// ---------------------------------------------------------------------------

/// High-level emitter for the `.debug_info` section content.
///
/// The emitter coordinates DIE construction, abbreviation table
/// registration, string interning, and byte serialization for complete
/// DWARF v4 compilation units.
///
/// # Usage
///
/// ```ignore
/// let emitter = DebugInfoEmitter::new(8); // 64-bit target
/// let bytes = emitter.emit_compilation_unit(
///     &cu_info, &functions, &type_dies,
///     &mut abbrev_table, &mut string_table,
/// );
/// ```
pub struct DebugInfoEmitter {
    /// Target address size in bytes (4 for i686, 8 for 64-bit targets).
    pub address_size: u8,
}

impl DebugInfoEmitter {
    /// Create a new emitter for the given target address size.
    ///
    /// # Arguments
    /// * `address_size` — 4 for i686 (ELF32), 8 for x86-64/AArch64/RISC-V 64
    ///   (ELF64).
    pub fn new(address_size: u8) -> Self {
        Self { address_size }
    }

    /// Emit a complete `.debug_info` compilation unit.
    ///
    /// This method:
    /// 1. Interns all strings (producer, file name, directory, function names,
    ///    variable names, type names) into the shared string table.
    /// 2. Builds the root `DW_TAG_compile_unit` DIE.
    /// 3. Attaches function DIEs (with parameters and locals) as children.
    /// 4. Attaches pre-built type DIEs as children.
    /// 5. Registers all abbreviation entries.
    /// 6. Serializes the entire DIE tree into bytes.
    /// 7. Prepends the 11-byte compilation unit header.
    /// 8. Patches the `unit_length` field in the header.
    ///
    /// # Arguments
    /// * `cu_info` — Compilation unit metadata (file, directory, address range).
    /// * `functions` — Debug info for each function in the CU.
    /// * `type_dies` — Pre-constructed type DIEs (base types, pointers,
    ///   structs, arrays, typedefs).
    /// * `abbrev_table` — Shared abbreviation table (mutated to register
    ///   new entries).
    /// * `string_table` — Shared string table (mutated to intern new strings).
    ///
    /// # Returns
    /// The serialized `.debug_info` bytes for this compilation unit,
    /// including the CU header.
    pub fn emit_compilation_unit(
        &self,
        cu_info: &CompileUnitInfo,
        functions: &[FunctionDebugInfo],
        type_dies: &[DebugInfoEntry],
        abbrev_table: &mut AbbreviationTable,
        string_table: &mut StringTable,
    ) -> Vec<u8> {
        // --- Step 1: Intern strings and build root CU DIE ---
        let producer_offset = string_table.add(BCC_PRODUCER);
        let name_offset = string_table.add(&cu_info.file_name);
        let comp_dir_offset = string_table.add(&cu_info.comp_dir);

        let mut cu_die = build_compile_unit_die(cu_info);

        // Patch string offsets in the CU DIE attributes.
        patch_string_offset(&mut cu_die.attributes, DW_AT_PRODUCER, producer_offset);
        patch_string_offset(&mut cu_die.attributes, DW_AT_NAME, name_offset);
        patch_string_offset(&mut cu_die.attributes, DW_AT_COMP_DIR, comp_dir_offset);

        // --- Step 2: Add type DIEs as children ---
        for type_die in type_dies {
            let mut patched = type_die.clone();
            self.intern_die_strings(&mut patched, string_table);
            cu_die.children.push(patched);
        }

        // --- Step 3: Add function DIEs as children ---
        for func in functions {
            let mut func_die = build_subprogram_die(func);
            // Intern function name.
            let func_name_offset = string_table.add(&func.name);
            patch_string_offset(&mut func_die.attributes, DW_AT_NAME, func_name_offset);

            // Intern parameter names.
            for (i, param) in func.params.iter().enumerate() {
                if i < func_die.children.len() {
                    let param_name_offset = string_table.add(&param.name);
                    patch_string_offset(
                        &mut func_die.children[i].attributes,
                        DW_AT_NAME,
                        param_name_offset,
                    );
                }
            }

            // Intern local variable names.
            let param_count = func.params.len();
            for (i, local) in func.locals.iter().enumerate() {
                let child_idx = param_count + i;
                if child_idx < func_die.children.len() {
                    let local_name_offset = string_table.add(&local.name);
                    patch_string_offset(
                        &mut func_die.children[child_idx].attributes,
                        DW_AT_NAME,
                        local_name_offset,
                    );
                }
            }

            cu_die.children.push(func_die);
        }

        // The CU DIE now has children (type dies + function dies).
        cu_die.has_children = !cu_die.children.is_empty() || true; // CU always has_children

        // --- Step 4: Register abbreviation for root CU DIE ---
        let cu_abbrev_attrs: Vec<(u16, u16)> =
            cu_die.attributes.iter().map(|a| (a.name, a.form)).collect();
        let cu_abbrev_code =
            abbrev_table.add_abbreviation(cu_die.tag, cu_die.has_children, cu_abbrev_attrs);

        // --- Step 5: Serialize the CU ---
        // First, write the CU header (11 bytes for 32-bit DWARF).
        let mut result = Vec::new();

        // Placeholder for unit_length (4 bytes).
        let length_pos = result.len();
        write_u32_le(&mut result, 0); // placeholder

        // Version (2 bytes).
        write_u16_le(&mut result, 4); // DWARF version 4

        // Debug abbrev offset (4 bytes) — always 0 for now (single CU).
        write_u32_le(&mut result, 0);

        // Address size (1 byte).
        write_u8(&mut result, self.address_size);

        // Serialize the DIE tree.
        serialize_die(
            &cu_die,
            cu_abbrev_code,
            &mut result,
            self.address_size,
            abbrev_table,
        );

        // --- Step 6: Patch unit_length ---
        // unit_length = total size minus the 4-byte length field itself.
        let unit_length = (result.len() - (length_pos + 4)) as u32;
        result[length_pos..length_pos + 4].copy_from_slice(&unit_length.to_le_bytes());

        result
    }

    /// Recursively intern all inline string attributes (`DW_FORM_STRING`)
    /// in a DIE tree, converting them to string table references
    /// (`DW_FORM_STRP`) with proper offsets.
    ///
    /// This handles type DIEs and member DIEs that were constructed with
    /// inline `AttributeValue::String` values by `build_base_type_die`,
    /// `build_structure_type_die`, `build_member_die`, and
    /// `build_typedef_die`. Each string is added to the shared string
    /// table for deduplication, and the attribute is converted in-place.
    fn intern_die_strings(&self, die: &mut DebugInfoEntry, string_table: &mut StringTable) {
        for attr in &mut die.attributes {
            if attr.form == DW_FORM_STRING {
                if let AttributeValue::String(ref s) = attr.value {
                    let offset = string_table.add(s);
                    attr.form = DW_FORM_STRP;
                    attr.value = AttributeValue::StringOffset(offset);
                }
            }
        }
        for child in &mut die.children {
            self.intern_die_strings(child, string_table);
        }
    }
}

/// Patch a `DW_FORM_STRP` attribute with the given string table offset.
///
/// Searches the attribute list for the first attribute with the given
/// `attr_name` and `DW_FORM_STRP` form, then replaces its value with
/// `AttributeValue::StringOffset(offset)`.
fn patch_string_offset(attributes: &mut [DieAttribute], attr_name: u16, offset: u32) {
    for attr in attributes.iter_mut() {
        if attr.name == attr_name && attr.form == DW_FORM_STRP {
            attr.value = AttributeValue::StringOffset(offset);
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Compile Unit DIE
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_compile_unit_die_tag_and_children() {
        let info = CompileUnitInfo {
            file_name: "main.c".to_string(),
            comp_dir: "/home/user/project".to_string(),
            low_pc: 0x1000,
            high_pc: 0x2000,
            line_program_offset: 0,
        };
        let die = build_compile_unit_die(&info);
        assert_eq!(die.tag, DW_TAG_COMPILE_UNIT);
        assert!(die.has_children);
    }

    #[test]
    fn test_compile_unit_die_attributes() {
        let info = CompileUnitInfo {
            file_name: "test.c".to_string(),
            comp_dir: "/tmp".to_string(),
            low_pc: 0x400000,
            high_pc: 0x401000,
            line_program_offset: 42,
        };
        let die = build_compile_unit_die(&info);

        // Check that required attributes are present.
        let attr_names: Vec<u16> = die.attributes.iter().map(|a| a.name).collect();
        assert!(attr_names.contains(&DW_AT_PRODUCER));
        assert!(attr_names.contains(&DW_AT_LANGUAGE));
        assert!(attr_names.contains(&DW_AT_NAME));
        assert!(attr_names.contains(&DW_AT_COMP_DIR));
        assert!(attr_names.contains(&DW_AT_LOW_PC));
        assert!(attr_names.contains(&DW_AT_HIGH_PC));
        assert!(attr_names.contains(&DW_AT_STMT_LIST));

        // Verify low_pc value.
        let low_pc_attr = die
            .attributes
            .iter()
            .find(|a| a.name == DW_AT_LOW_PC)
            .unwrap();
        assert_eq!(low_pc_attr.value, AttributeValue::Addr(0x400000));

        // Verify high_pc is the code size (not absolute address).
        let high_pc_attr = die
            .attributes
            .iter()
            .find(|a| a.name == DW_AT_HIGH_PC)
            .unwrap();
        assert_eq!(high_pc_attr.value, AttributeValue::Data4(0x1000));

        // Verify stmt_list offset.
        let stmt_attr = die
            .attributes
            .iter()
            .find(|a| a.name == DW_AT_STMT_LIST)
            .unwrap();
        assert_eq!(stmt_attr.value, AttributeValue::SecOffset(42));
    }

    // -----------------------------------------------------------------------
    // Subprogram DIE
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_subprogram_die_basic() {
        let func = FunctionDebugInfo {
            name: "main".to_string(),
            low_pc: 0x1000,
            high_pc: 0x1100,
            return_type_offset: Some(0x30),
            is_external: true,
            decl_file: 1,
            decl_line: 10,
            frame_base_expr: vec![DW_OP_REG0 + 6], // DW_OP_reg6 (rbp)
            params: vec![],
            locals: vec![],
        };
        let die = build_subprogram_die(&func);
        assert_eq!(die.tag, DW_TAG_SUBPROGRAM);

        let attr_names: Vec<u16> = die.attributes.iter().map(|a| a.name).collect();
        assert!(attr_names.contains(&DW_AT_NAME));
        assert!(attr_names.contains(&DW_AT_LOW_PC));
        assert!(attr_names.contains(&DW_AT_HIGH_PC));
        assert!(attr_names.contains(&DW_AT_TYPE));
        assert!(attr_names.contains(&DW_AT_EXTERNAL));
        assert!(attr_names.contains(&DW_AT_PROTOTYPED));
        assert!(attr_names.contains(&DW_AT_FRAME_BASE));
        assert!(attr_names.contains(&DW_AT_DECL_FILE));
        assert!(attr_names.contains(&DW_AT_DECL_LINE));
    }

    #[test]
    fn test_subprogram_void_return_omits_type() {
        let func = FunctionDebugInfo {
            name: "void_func".to_string(),
            low_pc: 0x2000,
            high_pc: 0x2080,
            return_type_offset: None,
            is_external: false,
            decl_file: 1,
            decl_line: 20,
            frame_base_expr: vec![],
            params: vec![],
            locals: vec![],
        };
        let die = build_subprogram_die(&func);

        let attr_names: Vec<u16> = die.attributes.iter().map(|a| a.name).collect();
        assert!(!attr_names.contains(&DW_AT_TYPE));
        assert!(!attr_names.contains(&DW_AT_EXTERNAL));
    }

    #[test]
    fn test_subprogram_with_params_and_locals() {
        let func = FunctionDebugInfo {
            name: "add".to_string(),
            low_pc: 0x3000,
            high_pc: 0x3040,
            return_type_offset: Some(0x20),
            is_external: true,
            decl_file: 1,
            decl_line: 5,
            frame_base_expr: build_frame_base_expr(Architecture::X86_64),
            params: vec![
                ParamDebugInfo {
                    name: "a".to_string(),
                    type_offset: 0x20,
                    location_expr: build_fbreg_location(-8),
                },
                ParamDebugInfo {
                    name: "b".to_string(),
                    type_offset: 0x20,
                    location_expr: build_fbreg_location(-16),
                },
            ],
            locals: vec![VariableDebugInfo {
                name: "result".to_string(),
                type_offset: 0x20,
                location_expr: build_fbreg_location(-24),
                decl_file: 1,
                decl_line: 6,
            }],
        };
        let die = build_subprogram_die(&func);

        assert!(die.has_children);
        assert_eq!(die.children.len(), 3); // 2 params + 1 local
        assert_eq!(die.children[0].tag, DW_TAG_FORMAL_PARAMETER);
        assert_eq!(die.children[1].tag, DW_TAG_FORMAL_PARAMETER);
        assert_eq!(die.children[2].tag, DW_TAG_VARIABLE);
    }

    // -----------------------------------------------------------------------
    // Variable DIE
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_variable_die() {
        let var = VariableDebugInfo {
            name: "counter".to_string(),
            type_offset: 0x40,
            location_expr: build_fbreg_location(-4),
            decl_file: 1,
            decl_line: 15,
        };
        let die = build_variable_die(&var);
        assert_eq!(die.tag, DW_TAG_VARIABLE);
        assert!(!die.has_children);

        let attr_names: Vec<u16> = die.attributes.iter().map(|a| a.name).collect();
        assert!(attr_names.contains(&DW_AT_NAME));
        assert!(attr_names.contains(&DW_AT_TYPE));
        assert!(attr_names.contains(&DW_AT_LOCATION));
        assert!(attr_names.contains(&DW_AT_DECL_FILE));
        assert!(attr_names.contains(&DW_AT_DECL_LINE));
    }

    // -----------------------------------------------------------------------
    // Formal Parameter DIE
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_formal_parameter_die() {
        let param = ParamDebugInfo {
            name: "argc".to_string(),
            type_offset: 0x20,
            location_expr: build_fbreg_location(16),
        };
        let die = build_formal_parameter_die(&param);
        assert_eq!(die.tag, DW_TAG_FORMAL_PARAMETER);
        assert!(!die.has_children);

        let attr_names: Vec<u16> = die.attributes.iter().map(|a| a.name).collect();
        assert!(attr_names.contains(&DW_AT_NAME));
        assert!(attr_names.contains(&DW_AT_TYPE));
        assert!(attr_names.contains(&DW_AT_LOCATION));
    }

    // -----------------------------------------------------------------------
    // Base Type DIE
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_base_type_die_int() {
        let die = build_base_type_die("int", 4, DW_ATE_SIGNED);
        assert_eq!(die.tag, DW_TAG_BASE_TYPE);
        assert!(!die.has_children);

        let byte_size_attr = die
            .attributes
            .iter()
            .find(|a| a.name == DW_AT_BYTE_SIZE)
            .unwrap();
        assert_eq!(byte_size_attr.value, AttributeValue::Data1(4));

        let encoding_attr = die
            .attributes
            .iter()
            .find(|a| a.name == DW_AT_ENCODING)
            .unwrap();
        assert_eq!(encoding_attr.value, AttributeValue::Data1(DW_ATE_SIGNED));
    }

    #[test]
    fn test_build_base_type_die_char() {
        let die = build_base_type_die("char", 1, DW_ATE_SIGNED_CHAR);
        let encoding_attr = die
            .attributes
            .iter()
            .find(|a| a.name == DW_AT_ENCODING)
            .unwrap();
        assert_eq!(
            encoding_attr.value,
            AttributeValue::Data1(DW_ATE_SIGNED_CHAR)
        );
    }

    #[test]
    fn test_build_base_type_die_float() {
        let die = build_base_type_die("float", 4, DW_ATE_FLOAT);
        let byte_size = die
            .attributes
            .iter()
            .find(|a| a.name == DW_AT_BYTE_SIZE)
            .unwrap();
        assert_eq!(byte_size.value, AttributeValue::Data1(4));
    }

    #[test]
    fn test_build_base_type_die_double() {
        let die = build_base_type_die("double", 8, DW_ATE_FLOAT);
        let byte_size = die
            .attributes
            .iter()
            .find(|a| a.name == DW_AT_BYTE_SIZE)
            .unwrap();
        assert_eq!(byte_size.value, AttributeValue::Data1(8));
    }

    #[test]
    fn test_build_base_type_die_bool() {
        let die = build_base_type_die("_Bool", 1, DW_ATE_BOOLEAN);
        let encoding_attr = die
            .attributes
            .iter()
            .find(|a| a.name == DW_AT_ENCODING)
            .unwrap();
        assert_eq!(encoding_attr.value, AttributeValue::Data1(DW_ATE_BOOLEAN));
    }

    // -----------------------------------------------------------------------
    // Pointer Type DIE
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_pointer_type_die_64bit() {
        let die = build_pointer_type_die(0x30, 8);
        assert_eq!(die.tag, DW_TAG_POINTER_TYPE);
        assert!(!die.has_children);

        let type_attr = die
            .attributes
            .iter()
            .find(|a| a.name == DW_AT_TYPE)
            .unwrap();
        assert_eq!(type_attr.value, AttributeValue::Ref(0x30));

        let size_attr = die
            .attributes
            .iter()
            .find(|a| a.name == DW_AT_BYTE_SIZE)
            .unwrap();
        assert_eq!(size_attr.value, AttributeValue::Data1(8));
    }

    #[test]
    fn test_build_pointer_type_die_32bit() {
        let die = build_pointer_type_die(0x20, 4);
        let size_attr = die
            .attributes
            .iter()
            .find(|a| a.name == DW_AT_BYTE_SIZE)
            .unwrap();
        assert_eq!(size_attr.value, AttributeValue::Data1(4));
    }

    // -----------------------------------------------------------------------
    // Structure Type DIE
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_structure_type_die() {
        let members = vec![
            MemberDebugInfo {
                name: "x".to_string(),
                type_offset: 0x20,
                byte_offset: 0,
            },
            MemberDebugInfo {
                name: "y".to_string(),
                type_offset: 0x20,
                byte_offset: 4,
            },
        ];
        let die = build_structure_type_die("point", 8, members);
        assert_eq!(die.tag, DW_TAG_STRUCTURE_TYPE);
        assert!(die.has_children);
        assert_eq!(die.children.len(), 2);
        assert_eq!(die.children[0].tag, DW_TAG_MEMBER);
        assert_eq!(die.children[1].tag, DW_TAG_MEMBER);

        let size_attr = die
            .attributes
            .iter()
            .find(|a| a.name == DW_AT_BYTE_SIZE)
            .unwrap();
        assert_eq!(size_attr.value, AttributeValue::Data4(8));
    }

    #[test]
    fn test_build_structure_type_die_empty() {
        let die = build_structure_type_die("empty_struct", 0, vec![]);
        assert!(!die.has_children);
        assert!(die.children.is_empty());
    }

    // -----------------------------------------------------------------------
    // Member DIE
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_member_die() {
        let member = MemberDebugInfo {
            name: "field".to_string(),
            type_offset: 0x30,
            byte_offset: 8,
        };
        let die = build_member_die(&member);
        assert_eq!(die.tag, DW_TAG_MEMBER);
        assert!(!die.has_children);

        let offset_attr = die
            .attributes
            .iter()
            .find(|a| a.name == DW_AT_DATA_MEMBER_LOCATION)
            .unwrap();
        assert_eq!(offset_attr.value, AttributeValue::Data4(8));
    }

    // -----------------------------------------------------------------------
    // Array Type DIE
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_array_type_die_with_count() {
        let die = build_array_type_die(0x20, Some(10));
        assert_eq!(die.tag, DW_TAG_ARRAY_TYPE);
        assert!(die.has_children);
        assert_eq!(die.children.len(), 1);
        assert_eq!(die.children[0].tag, DW_TAG_SUBRANGE_TYPE);

        let bound_attr = die.children[0]
            .attributes
            .iter()
            .find(|a| a.name == DW_AT_UPPER_BOUND)
            .unwrap();
        // Upper bound = count - 1 = 9.
        assert_eq!(bound_attr.value, AttributeValue::UData(9));
    }

    #[test]
    fn test_build_array_type_die_unbounded() {
        let die = build_array_type_die(0x20, None);
        assert_eq!(die.tag, DW_TAG_ARRAY_TYPE);
        assert!(!die.has_children);
        assert!(die.children.is_empty());
    }

    // -----------------------------------------------------------------------
    // Typedef DIE
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_typedef_die() {
        let die = build_typedef_die("size_t", 0x40);
        assert_eq!(die.tag, DW_TAG_TYPEDEF);
        assert!(!die.has_children);

        let type_attr = die
            .attributes
            .iter()
            .find(|a| a.name == DW_AT_TYPE)
            .unwrap();
        assert_eq!(type_attr.value, AttributeValue::Ref(0x40));
    }

    // -----------------------------------------------------------------------
    // Location Expression Helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_fbreg_location_negative_offset() {
        let expr = build_fbreg_location(-8);
        assert_eq!(expr[0], DW_OP_FBREG);
        // -8 in SLEB128 = 0x78
        assert_eq!(expr[1], 0x78);
        assert_eq!(expr.len(), 2);
    }

    #[test]
    fn test_build_fbreg_location_positive_offset() {
        let expr = build_fbreg_location(16);
        assert_eq!(expr[0], DW_OP_FBREG);
        // 16 in SLEB128 = 0x10
        assert_eq!(expr[1], 0x10);
        assert_eq!(expr.len(), 2);
    }

    #[test]
    fn test_build_fbreg_location_zero() {
        let expr = build_fbreg_location(0);
        assert_eq!(expr[0], DW_OP_FBREG);
        assert_eq!(expr[1], 0x00);
        assert_eq!(expr.len(), 2);
    }

    #[test]
    fn test_build_fbreg_location_large_negative() {
        let expr = build_fbreg_location(-128);
        assert_eq!(expr[0], DW_OP_FBREG);
        // -128 in SLEB128 = 0x80, 0x7f
        assert_eq!(expr[1], 0x80);
        assert_eq!(expr[2], 0x7f);
    }

    #[test]
    fn test_build_addr_location_8byte() {
        let expr = build_addr_location(0x400000, 8);
        assert_eq!(expr[0], DW_OP_ADDR);
        // 0x400000 as 8-byte LE.
        let addr_bytes = &expr[1..9];
        let addr = u64::from_le_bytes(addr_bytes.try_into().unwrap());
        assert_eq!(addr, 0x400000);
        assert_eq!(expr.len(), 9);
    }

    #[test]
    fn test_build_addr_location_4byte() {
        let expr = build_addr_location(0x8048000, 4);
        assert_eq!(expr[0], DW_OP_ADDR);
        let addr_bytes = &expr[1..5];
        let addr = u32::from_le_bytes(addr_bytes.try_into().unwrap());
        assert_eq!(addr, 0x8048000);
        assert_eq!(expr.len(), 5);
    }

    #[test]
    fn test_build_reg_location_small() {
        // Register 6 (rbp on x86-64).
        let expr = build_reg_location(6);
        assert_eq!(expr.len(), 1);
        assert_eq!(expr[0], DW_OP_REG0 + 6);
    }

    #[test]
    fn test_build_reg_location_reg31() {
        let expr = build_reg_location(31);
        assert_eq!(expr.len(), 1);
        assert_eq!(expr[0], DW_OP_REG0 + 31);
    }

    #[test]
    fn test_build_reg_location_extended() {
        // Register 32 requires DW_OP_regx encoding.
        let expr = build_reg_location(32);
        assert_eq!(expr[0], DW_OP_REGX);
        assert_eq!(expr[1], 32); // 32 as ULEB128 = single byte 0x20
    }

    #[test]
    fn test_build_reg_location_large_reg() {
        let expr = build_reg_location(128);
        assert_eq!(expr[0], DW_OP_REGX);
        // 128 as ULEB128 = 0x80, 0x01
        assert_eq!(expr[1], 0x80);
        assert_eq!(expr[2], 0x01);
    }

    // -----------------------------------------------------------------------
    // Frame Base Expression (Architecture-Specific)
    // -----------------------------------------------------------------------

    #[test]
    fn test_frame_base_expr_x86_64() {
        let expr = build_frame_base_expr(Architecture::X86_64);
        assert_eq!(expr, vec![DW_OP_REG0 + 6]); // DW_OP_reg6 (rbp)
    }

    #[test]
    fn test_frame_base_expr_i686() {
        let expr = build_frame_base_expr(Architecture::I686);
        assert_eq!(expr, vec![DW_OP_REG0 + 5]); // DW_OP_reg5 (ebp)
    }

    #[test]
    fn test_frame_base_expr_aarch64() {
        let expr = build_frame_base_expr(Architecture::Aarch64);
        assert_eq!(expr, vec![DW_OP_REG0 + 29]); // DW_OP_reg29 (x29/fp)
    }

    #[test]
    fn test_frame_base_expr_riscv64() {
        let expr = build_frame_base_expr(Architecture::Riscv64);
        assert_eq!(expr, vec![DW_OP_REG0 + 8]); // DW_OP_reg8 (s0/fp)
    }

    // -----------------------------------------------------------------------
    // DIE Serialization
    // -----------------------------------------------------------------------

    #[test]
    fn test_serialize_die_simple_base_type() {
        let die = build_base_type_die("int", 4, DW_ATE_SIGNED);
        let mut abbrev = AbbreviationTable::new();
        let abbrev_attrs: Vec<(u16, u16)> =
            die.attributes.iter().map(|a| (a.name, a.form)).collect();
        let code = abbrev.add_abbreviation(die.tag, die.has_children, abbrev_attrs);

        let mut buffer = Vec::new();
        serialize_die(&die, code, &mut buffer, 8, &mut abbrev);

        // Should start with ULEB128(1) for the first abbreviation code.
        assert_eq!(buffer[0], 1);
        // buffer should be non-empty and contain the attribute data.
        assert!(buffer.len() > 1);
    }

    #[test]
    fn test_serialize_die_with_children_terminator() {
        let parent = DebugInfoEntry {
            tag: DW_TAG_STRUCTURE_TYPE,
            has_children: true,
            attributes: vec![DieAttribute {
                name: DW_AT_BYTE_SIZE,
                form: DW_FORM_DATA4,
                value: AttributeValue::Data4(8),
            }],
            children: vec![DebugInfoEntry {
                tag: DW_TAG_MEMBER,
                has_children: false,
                attributes: vec![DieAttribute {
                    name: DW_AT_DATA_MEMBER_LOCATION,
                    form: DW_FORM_DATA4,
                    value: AttributeValue::Data4(0),
                }],
                children: vec![],
            }],
        };

        let mut abbrev = AbbreviationTable::new();
        let parent_attrs: Vec<(u16, u16)> =
            parent.attributes.iter().map(|a| (a.name, a.form)).collect();
        let parent_code = abbrev.add_abbreviation(parent.tag, parent.has_children, parent_attrs);

        let mut buffer = Vec::new();
        serialize_die(&parent, parent_code, &mut buffer, 8, &mut abbrev);

        // The last byte must be 0x00 (children list terminator).
        assert_eq!(*buffer.last().unwrap(), 0x00);
    }

    #[test]
    fn test_serialize_die_flag_present_emits_no_bytes() {
        let die = DebugInfoEntry {
            tag: DW_TAG_SUBPROGRAM,
            has_children: false,
            attributes: vec![
                DieAttribute {
                    name: DW_AT_EXTERNAL,
                    form: DW_FORM_FLAG_PRESENT,
                    value: AttributeValue::FlagPresent,
                },
                DieAttribute {
                    name: DW_AT_PROTOTYPED,
                    form: DW_FORM_FLAG_PRESENT,
                    value: AttributeValue::FlagPresent,
                },
            ],
            children: vec![],
        };

        let mut abbrev = AbbreviationTable::new();
        let attrs: Vec<(u16, u16)> = die.attributes.iter().map(|a| (a.name, a.form)).collect();
        let code = abbrev.add_abbreviation(die.tag, die.has_children, attrs);

        let mut buffer = Vec::new();
        serialize_die(&die, code, &mut buffer, 8, &mut abbrev);

        // Only the abbreviation code byte (ULEB128(1)) should be present,
        // since flag_present emits no data and there are no other attributes.
        assert_eq!(buffer.len(), 1);
        assert_eq!(buffer[0], code as u8);
    }

    #[test]
    fn test_serialize_die_exprloc_encoding() {
        let expr = vec![DW_OP_FBREG, 0x78]; // DW_OP_fbreg, -8
        let die = DebugInfoEntry {
            tag: DW_TAG_VARIABLE,
            has_children: false,
            attributes: vec![DieAttribute {
                name: DW_AT_LOCATION,
                form: DW_FORM_EXPRLOC,
                value: AttributeValue::ExprLoc(expr.clone()),
            }],
            children: vec![],
        };

        let mut abbrev = AbbreviationTable::new();
        let attrs: Vec<(u16, u16)> = die.attributes.iter().map(|a| (a.name, a.form)).collect();
        let code = abbrev.add_abbreviation(die.tag, die.has_children, attrs);

        let mut buffer = Vec::new();
        serialize_die(&die, code, &mut buffer, 8, &mut abbrev);

        // buffer[0] = abbrev code (1)
        // buffer[1] = ULEB128(2) — length of expression
        // buffer[2] = DW_OP_FBREG (0x91)
        // buffer[3] = 0x78 (-8 as SLEB128)
        assert_eq!(buffer[0], code as u8);
        assert_eq!(buffer[1], 2); // length
        assert_eq!(buffer[2], DW_OP_FBREG);
        assert_eq!(buffer[3], 0x78);
    }

    #[test]
    fn test_serialize_die_addr_8byte() {
        let die = DebugInfoEntry {
            tag: DW_TAG_SUBPROGRAM,
            has_children: false,
            attributes: vec![DieAttribute {
                name: DW_AT_LOW_PC,
                form: DW_FORM_ADDR,
                value: AttributeValue::Addr(0x400000),
            }],
            children: vec![],
        };

        let mut abbrev = AbbreviationTable::new();
        let attrs: Vec<(u16, u16)> = die.attributes.iter().map(|a| (a.name, a.form)).collect();
        let code = abbrev.add_abbreviation(die.tag, die.has_children, attrs);

        let mut buffer = Vec::new();
        serialize_die(&die, code, &mut buffer, 8, &mut abbrev);

        // buffer[0] = abbrev code
        // buffer[1..9] = 0x400000 as 8-byte LE
        assert_eq!(buffer.len(), 9); // 1 + 8
        let addr = u64::from_le_bytes(buffer[1..9].try_into().unwrap());
        assert_eq!(addr, 0x400000);
    }

    #[test]
    fn test_serialize_die_addr_4byte() {
        let die = DebugInfoEntry {
            tag: DW_TAG_SUBPROGRAM,
            has_children: false,
            attributes: vec![DieAttribute {
                name: DW_AT_LOW_PC,
                form: DW_FORM_ADDR,
                value: AttributeValue::Addr(0x8048000),
            }],
            children: vec![],
        };

        let mut abbrev = AbbreviationTable::new();
        let attrs: Vec<(u16, u16)> = die.attributes.iter().map(|a| (a.name, a.form)).collect();
        let code = abbrev.add_abbreviation(die.tag, die.has_children, attrs);

        let mut buffer = Vec::new();
        serialize_die(&die, code, &mut buffer, 4, &mut abbrev);

        // buffer[0] = abbrev code
        // buffer[1..5] = 0x8048000 as 4-byte LE
        assert_eq!(buffer.len(), 5); // 1 + 4
        let addr = u32::from_le_bytes(buffer[1..5].try_into().unwrap());
        assert_eq!(addr, 0x8048000);
    }

    // -----------------------------------------------------------------------
    // DebugInfoEmitter
    // -----------------------------------------------------------------------

    #[test]
    fn test_emitter_new() {
        let emitter = DebugInfoEmitter::new(8);
        assert_eq!(emitter.address_size, 8);
    }

    #[test]
    fn test_emitter_emit_compilation_unit_basic() {
        let emitter = DebugInfoEmitter::new(8);
        let cu_info = CompileUnitInfo {
            file_name: "test.c".to_string(),
            comp_dir: "/tmp".to_string(),
            low_pc: 0x1000,
            high_pc: 0x2000,
            line_program_offset: 0,
        };

        let int_type = build_base_type_die("int", 4, DW_ATE_SIGNED);
        let mut abbrev = AbbreviationTable::new();
        let mut strings = StringTable::new();

        // Intern the type name before passing.
        let mut int_die = int_type;
        let int_name_offset = strings.add("int");
        patch_string_offset(&mut int_die.attributes, DW_AT_NAME, int_name_offset);

        let bytes =
            emitter.emit_compilation_unit(&cu_info, &[], &[int_die], &mut abbrev, &mut strings);

        // The result should start with the CU header:
        // bytes[0..4] = unit_length (non-zero)
        let unit_length = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert!(unit_length > 0);

        // bytes[4..6] = version = 4
        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        assert_eq!(version, 4);

        // bytes[10] = address_size = 8
        assert_eq!(bytes[10], 8);

        // Total length should match header claim.
        assert_eq!(bytes.len() as u32, unit_length + 4);
    }

    #[test]
    fn test_emitter_emit_with_functions() {
        let emitter = DebugInfoEmitter::new(8);
        let cu_info = CompileUnitInfo {
            file_name: "main.c".to_string(),
            comp_dir: "/home".to_string(),
            low_pc: 0x1000,
            high_pc: 0x2000,
            line_program_offset: 0,
        };

        let func = FunctionDebugInfo {
            name: "main".to_string(),
            low_pc: 0x1000,
            high_pc: 0x1100,
            return_type_offset: None,
            is_external: true,
            decl_file: 1,
            decl_line: 1,
            frame_base_expr: build_frame_base_expr(Architecture::X86_64),
            params: vec![],
            locals: vec![],
        };

        let mut abbrev = AbbreviationTable::new();
        let mut strings = StringTable::new();

        let bytes =
            emitter.emit_compilation_unit(&cu_info, &[func], &[], &mut abbrev, &mut strings);

        // Should produce non-empty output.
        assert!(!bytes.is_empty());

        // The abbreviation table should have entries (at least CU + subprogram).
        assert!(abbrev.get(1).is_some());
        assert!(abbrev.get(2).is_some());
    }

    #[test]
    fn test_emitter_32bit_address_size() {
        let emitter = DebugInfoEmitter::new(4);
        assert_eq!(emitter.address_size, 4);

        let cu_info = CompileUnitInfo {
            file_name: "test.c".to_string(),
            comp_dir: "/tmp".to_string(),
            low_pc: 0x8048000,
            high_pc: 0x8049000,
            line_program_offset: 0,
        };

        let mut abbrev = AbbreviationTable::new();
        let mut strings = StringTable::new();

        let bytes = emitter.emit_compilation_unit(&cu_info, &[], &[], &mut abbrev, &mut strings);

        // address_size byte in CU header should be 4.
        assert_eq!(bytes[10], 4);
    }

    // -----------------------------------------------------------------------
    // ULEB128 / SLEB128 encoding in serialized output
    // -----------------------------------------------------------------------

    #[test]
    fn test_serialize_die_udata_encoding() {
        let die = DebugInfoEntry {
            tag: DW_TAG_SUBRANGE_TYPE,
            has_children: false,
            attributes: vec![DieAttribute {
                name: DW_AT_UPPER_BOUND,
                form: DW_FORM_UDATA,
                value: AttributeValue::UData(255),
            }],
            children: vec![],
        };

        let mut abbrev = AbbreviationTable::new();
        let attrs: Vec<(u16, u16)> = die.attributes.iter().map(|a| (a.name, a.form)).collect();
        let code = abbrev.add_abbreviation(die.tag, die.has_children, attrs);

        let mut buffer = Vec::new();
        serialize_die(&die, code, &mut buffer, 8, &mut abbrev);

        // buffer[0] = abbrev code (1)
        // buffer[1..] = ULEB128(255) = 0xFF, 0x01
        assert_eq!(buffer[1], 0xFF);
        assert_eq!(buffer[2], 0x01);
    }

    #[test]
    fn test_serialize_die_sdata_encoding() {
        let die = DebugInfoEntry {
            tag: DW_TAG_VARIABLE,
            has_children: false,
            attributes: vec![DieAttribute {
                name: DW_AT_DECL_LINE,
                form: DW_FORM_SDATA,
                value: AttributeValue::SData(-100),
            }],
            children: vec![],
        };

        let mut abbrev = AbbreviationTable::new();
        let attrs: Vec<(u16, u16)> = die.attributes.iter().map(|a| (a.name, a.form)).collect();
        let code = abbrev.add_abbreviation(die.tag, die.has_children, attrs);

        let mut buffer = Vec::new();
        serialize_die(&die, code, &mut buffer, 8, &mut abbrev);

        // buffer[0] = abbrev code
        // buffer[1..] = SLEB128(-100) = 0x9C, 0x7F
        assert_eq!(buffer[1], 0x9C);
        assert_eq!(buffer[2], 0x7F);
    }

    // -----------------------------------------------------------------------
    // String interning integration
    // -----------------------------------------------------------------------

    #[test]
    fn test_patch_string_offset() {
        let mut attrs = vec![DieAttribute {
            name: DW_AT_NAME,
            form: DW_FORM_STRP,
            value: AttributeValue::StringOffset(0),
        }];

        patch_string_offset(&mut attrs, DW_AT_NAME, 42);
        assert_eq!(attrs[0].value, AttributeValue::StringOffset(42));
    }

    #[test]
    fn test_patch_string_offset_no_match() {
        let mut attrs = vec![DieAttribute {
            name: DW_AT_TYPE,
            form: DW_FORM_REF4,
            value: AttributeValue::Ref(0x10),
        }];

        // Should not modify anything since there's no DW_AT_NAME + DW_FORM_STRP.
        patch_string_offset(&mut attrs, DW_AT_NAME, 42);
        assert_eq!(attrs[0].value, AttributeValue::Ref(0x10));
    }
}
