// DWARF v4 Core Structures for bcc compiler debug info generation.
//
// This is the foundational module providing:
// - LEB128 encoding utilities (unsigned and signed)
// - Byte encoding helpers (little-endian primitives, address, string)
// - Abbreviation table construction and serialization
// - Compilation unit header emission
// - String table (.debug_str) management with deduplication
// - Address range table (.debug_aranges) serialization
// - Location list (.debug_loc) serialization
// - DWARF section builder coordinating all section generation
//
// All other debug modules (info.rs, line_program.rs, frame.rs) depend on
// types and utilities defined here.
//
// CONSTRAINTS:
// - Zero external dependencies: only std imports.
// - No unsafe code: pure data structure generation and byte encoding.
// - DWARF v4 compliance: 32-bit DWARF format throughout.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// DWARF v4 Constants
// ---------------------------------------------------------------------------

/// DWARF version used by this compiler (version 4).
pub const DWARF_VERSION: u16 = 4;

/// Indicates that a DIE using this abbreviation has children.
#[allow(non_upper_case_globals)]
pub const DW_CHILDREN_yes: u8 = 1;

/// Indicates that a DIE using this abbreviation has no children.
#[allow(non_upper_case_globals)]
pub const DW_CHILDREN_no: u8 = 0;

// -- DWARF Tag constants (DW_TAG_*) -----------------------------------------

/// Compilation unit tag.
pub const DW_TAG_COMPILE_UNIT: u16 = 0x11;
/// Subprogram (function) tag.
pub const DW_TAG_SUBPROGRAM: u16 = 0x2e;
/// Variable tag.
pub const DW_TAG_VARIABLE: u16 = 0x34;
/// Formal parameter tag.
pub const DW_TAG_FORMAL_PARAMETER: u16 = 0x05;
/// Base (fundamental) type tag.
pub const DW_TAG_BASE_TYPE: u16 = 0x24;
/// Pointer type tag.
pub const DW_TAG_POINTER_TYPE: u16 = 0x0f;
/// Structure type tag.
pub const DW_TAG_STRUCTURE_TYPE: u16 = 0x13;
/// Array type tag.
pub const DW_TAG_ARRAY_TYPE: u16 = 0x01;
/// Typedef tag.
pub const DW_TAG_TYPEDEF: u16 = 0x16;
/// Member (struct/union field) tag.
pub const DW_TAG_MEMBER: u16 = 0x0d;
/// Subrange type tag (array bounds).
pub const DW_TAG_SUBRANGE_TYPE: u16 = 0x21;
/// Enumeration type tag.
pub const DW_TAG_ENUMERATION_TYPE: u16 = 0x04;
/// Enumerator (enum value) tag.
pub const DW_TAG_ENUMERATOR: u16 = 0x28;
/// Union type tag.
pub const DW_TAG_UNION_TYPE: u16 = 0x17;
/// Const-qualified type tag.
pub const DW_TAG_CONST_TYPE: u16 = 0x26;
/// Volatile-qualified type tag.
pub const DW_TAG_VOLATILE_TYPE: u16 = 0x35;
/// Lexical block tag.
pub const DW_TAG_LEXICAL_BLOCK: u16 = 0x0b;
/// Unspecified parameters (variadic) tag.
pub const DW_TAG_UNSPECIFIED_PARAMETERS: u16 = 0x18;
/// Subroutine (function pointer) type tag.
pub const DW_TAG_SUBROUTINE_TYPE: u16 = 0x15;
/// Restrict-qualified type tag.
pub const DW_TAG_RESTRICT_TYPE: u16 = 0x37;

// -- DWARF Attribute constants (DW_AT_*) ------------------------------------

/// Name attribute.
pub const DW_AT_NAME: u16 = 0x03;
/// Compilation directory attribute.
pub const DW_AT_COMP_DIR: u16 = 0x1b;
/// Producer (compiler identification) attribute.
pub const DW_AT_PRODUCER: u16 = 0x25;
/// Source language attribute.
pub const DW_AT_LANGUAGE: u16 = 0x13;
/// Low program counter attribute.
pub const DW_AT_LOW_PC: u16 = 0x11;
/// High program counter attribute.
pub const DW_AT_HIGH_PC: u16 = 0x12;
/// Statement list (offset into .debug_line) attribute.
pub const DW_AT_STMT_LIST: u16 = 0x10;
/// Type reference attribute.
pub const DW_AT_TYPE: u16 = 0x49;
/// Byte size attribute.
pub const DW_AT_BYTE_SIZE: u16 = 0x0b;
/// Type encoding attribute.
pub const DW_AT_ENCODING: u16 = 0x3e;
/// Location attribute.
pub const DW_AT_LOCATION: u16 = 0x02;
/// External linkage flag attribute.
pub const DW_AT_EXTERNAL: u16 = 0x3f;
/// Declaration file index attribute.
pub const DW_AT_DECL_FILE: u16 = 0x3a;
/// Declaration line number attribute.
pub const DW_AT_DECL_LINE: u16 = 0x3b;
/// Data member location (byte offset) attribute.
pub const DW_AT_DATA_MEMBER_LOCATION: u16 = 0x38;
/// Upper bound attribute (array dimensions).
pub const DW_AT_UPPER_BOUND: u16 = 0x2f;
/// Prototyped flag attribute.
pub const DW_AT_PROTOTYPED: u16 = 0x27;
/// Frame base attribute.
pub const DW_AT_FRAME_BASE: u16 = 0x40;
/// Linkage name attribute.
pub const DW_AT_LINKAGE_NAME: u16 = 0x6e;
/// Const value attribute.
pub const DW_AT_CONST_VALUE: u16 = 0x1c;
/// Bit size attribute.
pub const DW_AT_BIT_SIZE: u16 = 0x0d;
/// Bit offset attribute.
pub const DW_AT_BIT_OFFSET: u16 = 0x0c;

// -- DWARF Attribute Form constants (DW_FORM_*) -----------------------------

/// Address form (4 or 8 bytes depending on target).
pub const DW_FORM_ADDR: u16 = 0x01;
/// 1-byte unsigned data form.
pub const DW_FORM_DATA1: u16 = 0x0b;
/// 2-byte unsigned data form.
pub const DW_FORM_DATA2: u16 = 0x05;
/// 4-byte unsigned data form.
pub const DW_FORM_DATA4: u16 = 0x06;
/// 8-byte unsigned data form.
pub const DW_FORM_DATA8: u16 = 0x07;
/// Inline null-terminated string form.
pub const DW_FORM_STRING: u16 = 0x08;
/// Offset into .debug_str form.
pub const DW_FORM_STRP: u16 = 0x0e;
/// 4-byte reference within CU form.
pub const DW_FORM_REF4: u16 = 0x13;
/// Unsigned LEB128 data form.
pub const DW_FORM_UDATA: u16 = 0x0f;
/// Signed LEB128 data form.
pub const DW_FORM_SDATA: u16 = 0x0d;
/// Flag present form (presence implies true; no data).
pub const DW_FORM_FLAG_PRESENT: u16 = 0x19;
/// Expression location (ULEB128 length + bytes) form.
pub const DW_FORM_EXPRLOC: u16 = 0x18;
/// Section offset form.
pub const DW_FORM_SEC_OFFSET: u16 = 0x17;
/// Block1 form (1-byte length + data bytes).
pub const DW_FORM_BLOCK1: u16 = 0x0a;

// -- DWARF Base Type Encoding constants (DW_ATE_*) --------------------------

/// Address encoding.
pub const DW_ATE_ADDRESS: u8 = 0x01;
/// Boolean encoding.
pub const DW_ATE_BOOLEAN: u8 = 0x02;
/// Floating-point encoding.
pub const DW_ATE_FLOAT: u8 = 0x04;
/// Signed integer encoding.
pub const DW_ATE_SIGNED: u8 = 0x05;
/// Signed character encoding.
pub const DW_ATE_SIGNED_CHAR: u8 = 0x06;
/// Unsigned integer encoding.
pub const DW_ATE_UNSIGNED: u8 = 0x07;
/// Unsigned character encoding.
pub const DW_ATE_UNSIGNED_CHAR: u8 = 0x08;

// -- DWARF Language constant ------------------------------------------------

/// C11 language identifier.
pub const DW_LANG_C11: u16 = 0x001d;

// -- DWARF Expression Opcodes (DW_OP_*) -------------------------------------

/// Frame base register + signed offset.
pub const DW_OP_FBREG: u8 = 0x91;
/// Absolute address.
pub const DW_OP_ADDR: u8 = 0x03;
/// Base register for DW_OP_reg0..DW_OP_reg31 (reg0 = 0x50).
pub const DW_OP_REG0: u8 = 0x50;
/// Extended register (register number as ULEB128 operand).
pub const DW_OP_REGX: u8 = 0x90;
/// Base for DW_OP_breg0..DW_OP_breg31 (breg0 = 0x70).
pub const DW_OP_BREG0: u8 = 0x70;
/// Plus unsigned constant.
pub const DW_OP_PLUS_UCONST: u8 = 0x23;
/// Stack value (marks top of expression stack as the value itself).
pub const DW_OP_STACK_VALUE: u8 = 0x9f;

// ---------------------------------------------------------------------------
// LEB128 Encoding Utilities
// ---------------------------------------------------------------------------

/// Encode an unsigned 64-bit value as ULEB128, returning a new `Vec<u8>`.
///
/// ULEB128 (Unsigned Little-Endian Base 128) encodes values in 7-bit groups,
/// where the high bit of each byte indicates whether more bytes follow.
pub fn encode_uleb128(mut value: u64) -> Vec<u8> {
    let mut result = Vec::new();
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80; // Set continuation bit
        }
        result.push(byte);
        if value == 0 {
            break;
        }
    }
    result
}

/// Encode a signed 64-bit value as SLEB128, returning a new `Vec<u8>`.
///
/// SLEB128 (Signed Little-Endian Base 128) encodes signed values in 7-bit
/// groups. The sign bit of the last byte's 7-bit payload indicates the sign
/// extension: if bit 6 is set the remaining (unencoded) bits are all ones,
/// otherwise all zeros.
pub fn encode_sleb128(mut value: i64) -> Vec<u8> {
    let mut result = Vec::new();
    let mut more = true;
    while more {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        // Termination condition: remaining bits are all-zero (positive) or
        // all-ones (negative) AND the sign bit of the current payload byte
        // matches accordingly.
        if (value == 0 && (byte & 0x40) == 0) || (value == -1 && (byte & 0x40) != 0) {
            more = false;
        } else {
            byte |= 0x80; // Set continuation bit
        }
        result.push(byte);
    }
    result
}

/// Encode an unsigned 64-bit value as ULEB128 directly into `buf`.
///
/// This avoids a separate allocation compared to [`encode_uleb128`].
pub fn encode_uleb128_to(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Encode a signed 64-bit value as SLEB128 directly into `buf`.
///
/// This avoids a separate allocation compared to [`encode_sleb128`].
pub fn encode_sleb128_to(buf: &mut Vec<u8>, mut value: i64) {
    let mut more = true;
    while more {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if (value == 0 && (byte & 0x40) == 0) || (value == -1 && (byte & 0x40) != 0) {
            more = false;
        } else {
            byte |= 0x80;
        }
        buf.push(byte);
    }
}

// ---------------------------------------------------------------------------
// Byte Encoding Helpers (little-endian)
// ---------------------------------------------------------------------------

/// Append a single byte to `buf`.
#[inline]
pub fn write_u8(buf: &mut Vec<u8>, value: u8) {
    buf.push(value);
}

/// Append a 16-bit value in little-endian byte order to `buf`.
#[inline]
pub fn write_u16_le(buf: &mut Vec<u8>, value: u16) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Append a 32-bit value in little-endian byte order to `buf`.
#[inline]
pub fn write_u32_le(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Append a 64-bit value in little-endian byte order to `buf`.
#[inline]
pub fn write_u64_le(buf: &mut Vec<u8>, value: u64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Write a target-width address to `buf`.
///
/// - For `addr_size == 4`: writes a 32-bit LE value (truncating the upper 32
///   bits of `value`).
/// - For `addr_size == 8`: writes a 64-bit LE value.
/// - Other sizes are unsupported and will panic in debug builds.
pub fn write_address(buf: &mut Vec<u8>, value: u64, addr_size: u8) {
    match addr_size {
        4 => write_u32_le(buf, value as u32),
        8 => write_u64_le(buf, value),
        _ => {
            // Graceful fallback: treat unknown sizes as 8-byte for robustness,
            // but this should never happen with valid target configurations.
            debug_assert!(
                false,
                "Unsupported address size: {}. Expected 4 or 8.",
                addr_size
            );
            write_u64_le(buf, value);
        }
    }
}

/// Append a null-terminated UTF-8 string to `buf`.
///
/// The string bytes are followed by a single `0x00` terminator byte.
pub fn write_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(s.as_bytes());
    buf.push(0x00);
}

// ---------------------------------------------------------------------------
// Abbreviation Table
// ---------------------------------------------------------------------------

/// A single abbreviation entry describing the shape of a DIE.
///
/// Each entry records the DWARF tag, whether the DIE has children, and the
/// ordered list of (attribute-name, attribute-form) pairs. The `code` field
/// is a 1-based index assigned sequentially when the entry is added to an
/// [`AbbreviationTable`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbbreviationEntry {
    /// Abbreviation code (1-based, sequentially assigned).
    pub code: u32,
    /// DWARF tag (DW_TAG_*).
    pub tag: u16,
    /// Whether DIEs using this abbreviation have children.
    pub has_children: bool,
    /// Ordered list of (attribute name, attribute form) pairs.
    pub attributes: Vec<(u16, u16)>,
}

/// A collection of abbreviation entries that will be serialized into the
/// `.debug_abbrev` ELF section.
///
/// Supports deduplication: adding an abbreviation with the same tag,
/// children flag, and attribute list as an existing entry will return the
/// existing code instead of creating a new entry.
#[derive(Debug, Clone)]
pub struct AbbreviationTable {
    entries: Vec<AbbreviationEntry>,
    next_code: u32,
}

impl AbbreviationTable {
    /// Create an empty abbreviation table.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_code: 1,
        }
    }

    /// Add a new abbreviation or return the code of an existing identical one.
    ///
    /// Deduplication compares `tag`, `has_children`, and the full `attributes`
    /// list. If a match is found the existing code is returned without
    /// allocating a new entry.
    pub fn add_abbreviation(
        &mut self,
        tag: u16,
        has_children: bool,
        attributes: Vec<(u16, u16)>,
    ) -> u32 {
        // Check for existing identical abbreviation (deduplication).
        for entry in &self.entries {
            if entry.tag == tag
                && entry.has_children == has_children
                && entry.attributes == attributes
            {
                return entry.code;
            }
        }

        let code = self.next_code;
        self.entries.push(AbbreviationEntry {
            code,
            tag,
            has_children,
            attributes,
        });
        self.next_code += 1;
        code
    }

    /// Look up an abbreviation entry by its code.
    ///
    /// Returns `None` if no entry with the given code exists.
    pub fn get(&self, code: u32) -> Option<&AbbreviationEntry> {
        self.entries.iter().find(|e| e.code == code)
    }

    /// Serialize the abbreviation table into the `.debug_abbrev` section
    /// byte format per DWARF v4 §7.5.3.
    ///
    /// Layout per entry:
    /// - ULEB128(code)
    /// - ULEB128(tag)
    /// - u8(DW_CHILDREN_yes | DW_CHILDREN_no)
    /// - For each attribute: ULEB128(attr_name), ULEB128(attr_form)
    /// - 0x00, 0x00 (attribute list terminator)
    ///
    /// After all entries: 0x00 (table terminator).
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        for entry in &self.entries {
            encode_uleb128_to(&mut buf, entry.code as u64);
            encode_uleb128_to(&mut buf, entry.tag as u64);
            buf.push(if entry.has_children {
                DW_CHILDREN_yes
            } else {
                DW_CHILDREN_no
            });
            for &(attr_name, attr_form) in &entry.attributes {
                encode_uleb128_to(&mut buf, attr_name as u64);
                encode_uleb128_to(&mut buf, attr_form as u64);
            }
            // Attribute list terminator.
            buf.push(0x00);
            buf.push(0x00);
        }
        // Table terminator (null abbreviation code).
        buf.push(0x00);
        buf
    }
}

impl Default for AbbreviationTable {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Compilation Unit Header
// ---------------------------------------------------------------------------

/// A DWARF v4 compilation unit header (32-bit DWARF format).
///
/// This header appears at the start of each compilation unit in the
/// `.debug_info` section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompilationUnitHeader {
    /// Total length of the CU (excluding this 4-byte length field).
    /// Typically set to a placeholder (0) and patched after the full CU
    /// is serialized.
    pub unit_length: u32,
    /// DWARF version (always 4 for DWARF v4).
    pub version: u16,
    /// Byte offset into the `.debug_abbrev` section for this CU's
    /// abbreviation table.
    pub debug_abbrev_offset: u32,
    /// Size of an address on the target in bytes (4 for i686, 8 for
    /// x86-64 / AArch64 / RISC-V 64).
    pub address_size: u8,
}

/// Serialize a [`CompilationUnitHeader`] into its 11-byte on-disk format.
///
/// Layout (32-bit DWARF format):
/// - 4 bytes: `unit_length` (LE)
/// - 2 bytes: `version` (LE)
/// - 4 bytes: `debug_abbrev_offset` (LE)
/// - 1 byte:  `address_size`
pub fn serialize_cu_header(header: &CompilationUnitHeader) -> Vec<u8> {
    let mut buf = Vec::with_capacity(11);
    write_u32_le(&mut buf, header.unit_length);
    write_u16_le(&mut buf, header.version);
    write_u32_le(&mut buf, header.debug_abbrev_offset);
    write_u8(&mut buf, header.address_size);
    buf
}

/// Build a DWARF v4 compilation unit header with a placeholder
/// `unit_length` of 0.
///
/// After the full compilation unit's DIE tree is serialized, the caller
/// must patch the `unit_length` field (bytes 0..4 of the CU) to reflect
/// the actual number of bytes following the length field (i.e., the CU
/// size minus 4).
pub fn build_cu_header(address_size: u8, abbrev_offset: u32) -> CompilationUnitHeader {
    CompilationUnitHeader {
        unit_length: 0, // Placeholder — patched after full CU serialization.
        version: DWARF_VERSION,
        debug_abbrev_offset: abbrev_offset,
        address_size,
    }
}

// ---------------------------------------------------------------------------
// String Table (.debug_str)
// ---------------------------------------------------------------------------

/// A deduplicated string table that will be serialized into the
/// `.debug_str` ELF section.
///
/// Strings are stored as null-terminated bytes. Each string is recorded
/// only once; subsequent additions of the same string return the
/// previously recorded offset. The offset is used with `DW_FORM_strp` in
/// `.debug_info` attribute values.
#[derive(Debug, Clone)]
pub struct StringTable {
    /// Concatenated null-terminated string data.
    data: Vec<u8>,
    /// Map from string content to its byte offset within `data`.
    offsets: HashMap<String, u32>,
}

impl StringTable {
    /// Create an empty string table.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            offsets: HashMap::new(),
        }
    }

    /// Add a string to the table, returning its byte offset.
    ///
    /// If the string already exists in the table the existing offset is
    /// returned without appending a duplicate.
    pub fn add(&mut self, s: &str) -> u32 {
        if self.offsets.contains_key(s) {
            return *self.offsets.get(s).unwrap();
        }
        let offset = self.data.len() as u32;
        self.data.extend_from_slice(s.as_bytes());
        self.data.push(0x00); // Null terminator.
        self.offsets.insert(s.to_owned(), offset);
        offset
    }

    /// Look up the byte offset of a previously added string.
    ///
    /// Returns `None` if the string has not been added.
    pub fn get_offset(&self, s: &str) -> Option<u32> {
        self.offsets.get(s).copied()
    }

    /// Serialize the string table into the raw `.debug_str` section bytes.
    ///
    /// The returned vector contains all added strings concatenated as
    /// null-terminated byte sequences.
    pub fn serialize(&self) -> Vec<u8> {
        self.data.clone()
    }
}

impl Default for StringTable {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Address Range Entry (.debug_aranges)
// ---------------------------------------------------------------------------

/// A single address range mapping a code region to a compilation unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddressRangeEntry {
    /// Start address of the range.
    pub start_address: u64,
    /// Length of the range in bytes.
    pub length: u64,
}

/// Serialize a list of address range entries into a `.debug_aranges`
/// section per DWARF v4 §6.1.2.
///
/// # Arguments
/// - `entries` — The address ranges to encode.
/// - `cu_offset` — Byte offset of the associated compilation unit in
///   `.debug_info`.
/// - `address_size` — Target address size (4 or 8).
///
/// # Layout
/// ```text
/// u32  unit_length        (patched)
/// u16  version            (2)
/// u32  debug_info_offset  (cu_offset)
/// u8   address_size
/// u8   segment_size       (0)
/// [padding to 2*address_size alignment]
/// (address, length) pairs
/// (0, 0)                  terminator
/// ```
pub fn serialize_aranges(
    entries: &[AddressRangeEntry],
    cu_offset: u32,
    address_size: u8,
) -> Vec<u8> {
    let mut buf = Vec::new();

    // Reserve space for unit_length (will be patched).
    let length_pos = buf.len();
    write_u32_le(&mut buf, 0); // placeholder

    // Header fields.
    write_u16_le(&mut buf, 2); // version (always 2 for .debug_aranges)
    write_u32_le(&mut buf, cu_offset); // debug_info_offset
    write_u8(&mut buf, address_size);
    write_u8(&mut buf, 0); // segment_size

    // Pad to align the first tuple to a 2*address_size boundary.
    // The header after the unit_length field occupies:
    //   2 (version) + 4 (debug_info_offset) + 1 (address_size) +
    //   1 (segment_size) = 8 bytes.
    // The alignment target is 2 * address_size bytes relative to the
    // start of the tuples (i.e. from the byte after the header).
    let tuple_alignment = (address_size as usize) * 2;
    let header_bytes_after_length = buf.len() - (length_pos + 4);
    let remainder = header_bytes_after_length % tuple_alignment;
    if remainder != 0 {
        let padding = tuple_alignment - remainder;
        for _ in 0..padding {
            buf.push(0x00);
        }
    }

    // Address/length tuples.
    for entry in entries {
        write_address(&mut buf, entry.start_address, address_size);
        write_address(&mut buf, entry.length, address_size);
    }

    // Terminator: pair of zeros.
    write_address(&mut buf, 0, address_size);
    write_address(&mut buf, 0, address_size);

    // Patch unit_length: total size minus the 4-byte length field itself.
    let unit_length = (buf.len() - (length_pos + 4)) as u32;
    buf[length_pos..length_pos + 4].copy_from_slice(&unit_length.to_le_bytes());

    buf
}

// ---------------------------------------------------------------------------
// Location List Entry (.debug_loc)
// ---------------------------------------------------------------------------

/// A single entry in a DWARF location list describing where a variable
/// resides over an address sub-range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocationListEntry {
    /// Beginning offset (relative to the compilation unit's base address).
    pub begin_offset: u64,
    /// Ending offset (exclusive, relative to the base address).
    pub end_offset: u64,
    /// DWARF expression bytes describing the variable's location in this
    /// sub-range.
    pub expression: Vec<u8>,
}

/// Serialize a location list into `.debug_loc` section bytes per DWARF
/// v4 §2.6.2.
///
/// Each entry is encoded as:
/// - `begin_offset` (4 or 8 bytes)
/// - `end_offset`   (4 or 8 bytes)
/// - `u16` expression length
/// - expression bytes
///
/// The list is terminated by a pair of zero addresses.
///
/// `base_address` is currently reserved for base-address selection
/// entries; standard entries are emitted relative to the CU base.
pub fn serialize_location_list(
    entries: &[LocationListEntry],
    _base_address: u64,
    address_size: u8,
) -> Vec<u8> {
    let mut buf = Vec::new();
    for entry in entries {
        write_address(&mut buf, entry.begin_offset, address_size);
        write_address(&mut buf, entry.end_offset, address_size);
        // Expression length as u16 LE.
        write_u16_le(&mut buf, entry.expression.len() as u16);
        buf.extend_from_slice(&entry.expression);
    }
    // Terminator: two zero addresses.
    write_address(&mut buf, 0, address_size);
    write_address(&mut buf, 0, address_size);
    buf
}

// ---------------------------------------------------------------------------
// DWARF Sections Output
// ---------------------------------------------------------------------------

/// The finalized byte content of all DWARF debug sections, ready to be
/// included in the ELF output by the linker.
#[derive(Debug, Clone)]
pub struct DwarfSections {
    /// `.debug_info` section bytes.
    pub debug_info: Vec<u8>,
    /// `.debug_abbrev` section bytes.
    pub debug_abbrev: Vec<u8>,
    /// `.debug_line` section bytes.
    pub debug_line: Vec<u8>,
    /// `.debug_str` section bytes.
    pub debug_str: Vec<u8>,
    /// `.debug_aranges` section bytes.
    pub debug_aranges: Vec<u8>,
    /// `.debug_frame` section bytes.
    pub debug_frame: Vec<u8>,
    /// `.debug_loc` section bytes.
    pub debug_loc: Vec<u8>,
}

// ---------------------------------------------------------------------------
// DWARF Section Builder (Coordinator)
// ---------------------------------------------------------------------------

/// Builder that accumulates data for all DWARF debug sections throughout
/// the debug-info generation process.
///
/// The builder owns the shared [`AbbreviationTable`] and [`StringTable`],
/// and provides per-section byte buffers that submodules (`info.rs`,
/// `line_program.rs`, `frame.rs`) append to. Once generation is complete
/// the caller invokes [`DwarfSectionBuilder::finalize`] to produce the
/// immutable [`DwarfSections`] output.
#[derive(Debug, Clone)]
pub struct DwarfSectionBuilder {
    /// Abbreviation table shared across all compilation units.
    pub abbrev_table: AbbreviationTable,
    /// String table shared across all compilation units.
    pub string_table: StringTable,
    /// Accumulated `.debug_info` section bytes.
    pub debug_info: Vec<u8>,
    /// Accumulated `.debug_line` section bytes.
    pub debug_line: Vec<u8>,
    /// Accumulated `.debug_frame` section bytes.
    pub debug_frame: Vec<u8>,
    /// Accumulated `.debug_aranges` section bytes.
    pub debug_aranges: Vec<u8>,
    /// Accumulated `.debug_loc` section bytes.
    pub debug_loc: Vec<u8>,
    /// Target address size in bytes (4 for i686, 8 for 64-bit targets).
    pub address_size: u8,
}

impl DwarfSectionBuilder {
    /// Create a new empty builder for the given target address size.
    pub fn new(address_size: u8) -> Self {
        Self {
            abbrev_table: AbbreviationTable::new(),
            string_table: StringTable::new(),
            debug_info: Vec::new(),
            debug_line: Vec::new(),
            debug_frame: Vec::new(),
            debug_aranges: Vec::new(),
            debug_loc: Vec::new(),
            address_size,
        }
    }

    /// Consume the builder and return the finalized [`DwarfSections`].
    ///
    /// The abbreviation table and string table are serialized at this
    /// point into their respective section byte vectors.
    pub fn finalize(self) -> DwarfSections {
        DwarfSections {
            debug_info: self.debug_info,
            debug_abbrev: self.abbrev_table.serialize(),
            debug_line: self.debug_line,
            debug_str: self.string_table.serialize(),
            debug_aranges: self.debug_aranges,
            debug_frame: self.debug_frame,
            debug_loc: self.debug_loc,
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
    // LEB128 unsigned encoding
    // -----------------------------------------------------------------------

    #[test]
    fn test_uleb128_zero() {
        assert_eq!(encode_uleb128(0), vec![0x00]);
    }

    #[test]
    fn test_uleb128_one() {
        assert_eq!(encode_uleb128(1), vec![0x01]);
    }

    #[test]
    fn test_uleb128_127() {
        assert_eq!(encode_uleb128(127), vec![0x7f]);
    }

    #[test]
    fn test_uleb128_128() {
        assert_eq!(encode_uleb128(128), vec![0x80, 0x01]);
    }

    #[test]
    fn test_uleb128_129() {
        assert_eq!(encode_uleb128(129), vec![0x81, 0x01]);
    }

    #[test]
    fn test_uleb128_16256() {
        // 0x3F80 = 16256
        assert_eq!(encode_uleb128(16256), vec![0x80, 0x7f]);
    }

    #[test]
    fn test_uleb128_624485() {
        assert_eq!(encode_uleb128(624485), vec![0xe5, 0x8e, 0x26]);
    }

    #[test]
    fn test_uleb128_large() {
        // 2^32 - 1
        let encoded = encode_uleb128(0xFFFF_FFFF);
        assert!(!encoded.is_empty());
        // Should be 5 bytes for a 32-bit max value.
        assert_eq!(encoded.len(), 5);
    }

    // -----------------------------------------------------------------------
    // LEB128 signed encoding
    // -----------------------------------------------------------------------

    #[test]
    fn test_sleb128_zero() {
        assert_eq!(encode_sleb128(0), vec![0x00]);
    }

    #[test]
    fn test_sleb128_one() {
        assert_eq!(encode_sleb128(1), vec![0x01]);
    }

    #[test]
    fn test_sleb128_neg_one() {
        assert_eq!(encode_sleb128(-1), vec![0x7f]);
    }

    #[test]
    fn test_sleb128_63() {
        assert_eq!(encode_sleb128(63), vec![0x3f]);
    }

    #[test]
    fn test_sleb128_64() {
        assert_eq!(encode_sleb128(64), vec![0xc0, 0x00]);
    }

    #[test]
    fn test_sleb128_neg_64() {
        assert_eq!(encode_sleb128(-64), vec![0x40]);
    }

    #[test]
    fn test_sleb128_neg_65() {
        assert_eq!(encode_sleb128(-65), vec![0xbf, 0x7f]);
    }

    #[test]
    fn test_sleb128_neg_128() {
        assert_eq!(encode_sleb128(-128), vec![0x80, 0x7f]);
    }

    #[test]
    fn test_sleb128_128() {
        assert_eq!(encode_sleb128(128), vec![0x80, 0x01]);
    }

    // -----------------------------------------------------------------------
    // LEB128 "to" variants write same bytes
    // -----------------------------------------------------------------------

    #[test]
    fn test_uleb128_to_matches_allocating() {
        for value in [0u64, 1, 127, 128, 16256, 624485, 0xFFFF_FFFF, u64::MAX] {
            let expected = encode_uleb128(value);
            let mut buf = Vec::new();
            encode_uleb128_to(&mut buf, value);
            assert_eq!(buf, expected, "ULEB128 mismatch for value {}", value);
        }
    }

    #[test]
    fn test_sleb128_to_matches_allocating() {
        for value in [0i64, 1, -1, 63, 64, -64, -65, -128, 128, i64::MIN, i64::MAX] {
            let expected = encode_sleb128(value);
            let mut buf = Vec::new();
            encode_sleb128_to(&mut buf, value);
            assert_eq!(buf, expected, "SLEB128 mismatch for value {}", value);
        }
    }

    // -----------------------------------------------------------------------
    // Byte encoding helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_write_u16_le() {
        let mut buf = Vec::new();
        write_u16_le(&mut buf, 0x0102);
        assert_eq!(buf, vec![0x02, 0x01]);
    }

    #[test]
    fn test_write_u32_le() {
        let mut buf = Vec::new();
        write_u32_le(&mut buf, 0x04030201);
        assert_eq!(buf, vec![0x01, 0x02, 0x03, 0x04]);
    }

    #[test]
    fn test_write_u64_le() {
        let mut buf = Vec::new();
        write_u64_le(&mut buf, 0x0807060504030201);
        assert_eq!(buf, vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);
    }

    #[test]
    fn test_write_address_4byte() {
        let mut buf = Vec::new();
        write_address(&mut buf, 0x12345678_AABBCCDD, 4);
        assert_eq!(buf.len(), 4);
        // Lower 32 bits: 0xAABBCCDD in LE.
        assert_eq!(buf, vec![0xDD, 0xCC, 0xBB, 0xAA]);
    }

    #[test]
    fn test_write_address_8byte() {
        let mut buf = Vec::new();
        write_address(&mut buf, 0x0000_0000_DEAD_BEEF, 8);
        assert_eq!(buf.len(), 8);
        assert_eq!(
            buf,
            vec![0xEF, 0xBE, 0xAD, 0xDE, 0x00, 0x00, 0x00, 0x00]
        );
    }

    #[test]
    fn test_write_string() {
        let mut buf = Vec::new();
        write_string(&mut buf, "hello");
        assert_eq!(buf, b"hello\0");
    }

    #[test]
    fn test_write_string_empty() {
        let mut buf = Vec::new();
        write_string(&mut buf, "");
        assert_eq!(buf, vec![0x00]);
    }

    // -----------------------------------------------------------------------
    // Abbreviation Table
    // -----------------------------------------------------------------------

    #[test]
    fn test_abbrev_first_code_is_one() {
        let mut table = AbbreviationTable::new();
        let code = table.add_abbreviation(
            DW_TAG_COMPILE_UNIT,
            true,
            vec![(DW_AT_NAME, DW_FORM_STRP)],
        );
        assert_eq!(code, 1);
    }

    #[test]
    fn test_abbrev_second_code_is_two() {
        let mut table = AbbreviationTable::new();
        table.add_abbreviation(
            DW_TAG_COMPILE_UNIT,
            true,
            vec![(DW_AT_NAME, DW_FORM_STRP)],
        );
        let code2 = table.add_abbreviation(
            DW_TAG_SUBPROGRAM,
            false,
            vec![(DW_AT_NAME, DW_FORM_STRP)],
        );
        assert_eq!(code2, 2);
    }

    #[test]
    fn test_abbrev_dedup_returns_same_code() {
        let mut table = AbbreviationTable::new();
        let code1 = table.add_abbreviation(
            DW_TAG_COMPILE_UNIT,
            true,
            vec![(DW_AT_NAME, DW_FORM_STRP), (DW_AT_LANGUAGE, DW_FORM_DATA2)],
        );
        let code2 = table.add_abbreviation(
            DW_TAG_COMPILE_UNIT,
            true,
            vec![(DW_AT_NAME, DW_FORM_STRP), (DW_AT_LANGUAGE, DW_FORM_DATA2)],
        );
        assert_eq!(code1, code2);
    }

    #[test]
    fn test_abbrev_get() {
        let mut table = AbbreviationTable::new();
        let code = table.add_abbreviation(DW_TAG_VARIABLE, false, vec![]);
        let entry = table.get(code).unwrap();
        assert_eq!(entry.tag, DW_TAG_VARIABLE);
        assert!(!entry.has_children);
        assert!(entry.attributes.is_empty());
    }

    #[test]
    fn test_abbrev_get_missing() {
        let table = AbbreviationTable::new();
        assert!(table.get(99).is_none());
    }

    #[test]
    fn test_abbrev_serialize_single_entry() {
        let mut table = AbbreviationTable::new();
        table.add_abbreviation(
            DW_TAG_COMPILE_UNIT,
            true,
            vec![(DW_AT_NAME, DW_FORM_STRP)],
        );
        let bytes = table.serialize();

        // Expected layout:
        // ULEB128(1) = 0x01               -- code
        // ULEB128(0x11) = 0x11            -- DW_TAG_compile_unit
        // 0x01                            -- DW_CHILDREN_yes
        // ULEB128(0x03) = 0x03            -- DW_AT_name
        // ULEB128(0x0e) = 0x0e            -- DW_FORM_strp
        // 0x00 0x00                       -- attr list terminator
        // 0x00                            -- table terminator
        assert_eq!(
            bytes,
            vec![0x01, 0x11, 0x01, 0x03, 0x0e, 0x00, 0x00, 0x00]
        );
    }

    #[test]
    fn test_abbrev_serialize_ends_with_null() {
        let table = AbbreviationTable::new();
        let bytes = table.serialize();
        // Empty table: only the terminator byte.
        assert_eq!(bytes, vec![0x00]);
    }

    #[test]
    fn test_abbrev_serialize_no_attrs_entry() {
        let mut table = AbbreviationTable::new();
        table.add_abbreviation(DW_TAG_BASE_TYPE, false, vec![]);
        let bytes = table.serialize();
        // code=1, tag=0x24, children=0, attr-terminator(0x00,0x00), table-terminator(0x00)
        assert_eq!(bytes, vec![0x01, 0x24, 0x00, 0x00, 0x00, 0x00]);
    }

    // -----------------------------------------------------------------------
    // String Table
    // -----------------------------------------------------------------------

    #[test]
    fn test_string_table_first_offset_zero() {
        let mut st = StringTable::new();
        let off = st.add("hello");
        assert_eq!(off, 0);
    }

    #[test]
    fn test_string_table_second_offset() {
        let mut st = StringTable::new();
        st.add("hello"); // occupies bytes 0..5 + null = 6 bytes total
        let off2 = st.add("world");
        assert_eq!(off2, 6); // starts right after "hello\0"
    }

    #[test]
    fn test_string_table_dedup() {
        let mut st = StringTable::new();
        let off1 = st.add("hello");
        let off2 = st.add("hello");
        assert_eq!(off1, off2);
    }

    #[test]
    fn test_string_table_get_offset() {
        let mut st = StringTable::new();
        st.add("hello");
        assert_eq!(st.get_offset("hello"), Some(0));
        assert_eq!(st.get_offset("nope"), None);
    }

    #[test]
    fn test_string_table_serialize() {
        let mut st = StringTable::new();
        st.add("hello");
        st.add("world");
        let data = st.serialize();
        assert_eq!(data, b"hello\0world\0");
    }

    #[test]
    fn test_string_table_empty() {
        let st = StringTable::new();
        let data = st.serialize();
        assert!(data.is_empty());
    }

    // -----------------------------------------------------------------------
    // Compilation Unit Header
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_cu_header_defaults() {
        let header = build_cu_header(8, 0);
        assert_eq!(header.version, DWARF_VERSION);
        assert_eq!(header.address_size, 8);
        assert_eq!(header.debug_abbrev_offset, 0);
        assert_eq!(header.unit_length, 0); // placeholder
    }

    #[test]
    fn test_serialize_cu_header_size() {
        let header = build_cu_header(4, 42);
        let bytes = serialize_cu_header(&header);
        // 4 (unit_length) + 2 (version) + 4 (abbrev_offset) + 1 (addr_size) = 11
        assert_eq!(bytes.len(), 11);
    }

    #[test]
    fn test_serialize_cu_header_version() {
        let header = build_cu_header(8, 0);
        let bytes = serialize_cu_header(&header);
        // Version field at offset 4..6, LE.
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        assert_eq!(version, 4);
    }

    #[test]
    fn test_serialize_cu_header_addr_size() {
        let header = build_cu_header(4, 0);
        let bytes = serialize_cu_header(&header);
        assert_eq!(bytes[10], 4);

        let header8 = build_cu_header(8, 0);
        let bytes8 = serialize_cu_header(&header8);
        assert_eq!(bytes8[10], 8);
    }

    #[test]
    fn test_serialize_cu_header_abbrev_offset() {
        let header = build_cu_header(8, 0x00ABCDEF);
        let bytes = serialize_cu_header(&header);
        let offset = u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]);
        assert_eq!(offset, 0x00ABCDEF);
    }

    // -----------------------------------------------------------------------
    // .debug_aranges serialization
    // -----------------------------------------------------------------------

    #[test]
    fn test_aranges_empty() {
        let bytes = serialize_aranges(&[], 0, 8);
        // Header: 4 (len) + 2 (ver) + 4 (info_off) + 1 (addr) + 1 (seg) = 12
        // Plus padding to align tuples to 16-byte boundary (8 bytes after
        // header = remainder 8%16=8 → 8 bytes padding)
        // Plus terminator: 16 bytes (two 8-byte zeros)
        // Total after length field should be >= 12 + padding + 16
        assert!(!bytes.is_empty());

        // Verify version at offset 4..6.
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        assert_eq!(version, 2);
    }

    #[test]
    fn test_aranges_single_entry_8() {
        let entries = vec![AddressRangeEntry {
            start_address: 0x1000,
            length: 0x200,
        }];
        let bytes = serialize_aranges(&entries, 0, 8);

        // Version field.
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        assert_eq!(version, 2);

        // Address size field.
        assert_eq!(bytes[10], 8);

        // Segment size field.
        assert_eq!(bytes[11], 0);

        // The unit_length field should be consistent.
        let unit_length = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(unit_length as usize, bytes.len() - 4);
    }

    #[test]
    fn test_aranges_single_entry_4() {
        let entries = vec![AddressRangeEntry {
            start_address: 0x1000,
            length: 0x200,
        }];
        let bytes = serialize_aranges(&entries, 0, 4);

        // Address size field.
        assert_eq!(bytes[10], 4);

        // Verify unit_length consistency.
        let unit_length = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(unit_length as usize, bytes.len() - 4);
    }

    #[test]
    fn test_aranges_cu_offset() {
        let bytes = serialize_aranges(&[], 0xDEAD, 4);
        // debug_info_offset at bytes 6..10.
        let offset = u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]);
        assert_eq!(offset, 0xDEAD);
    }

    // -----------------------------------------------------------------------
    // Location list serialization
    // -----------------------------------------------------------------------

    #[test]
    fn test_location_list_empty() {
        let bytes = serialize_location_list(&[], 0, 8);
        // Just the terminator: two 8-byte zero addresses = 16 bytes.
        assert_eq!(bytes.len(), 16);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_location_list_single_entry_8() {
        let entries = vec![LocationListEntry {
            begin_offset: 0x10,
            end_offset: 0x20,
            expression: vec![DW_OP_FBREG, 0x00], // DW_OP_fbreg + sleb128(0)
        }];
        let bytes = serialize_location_list(&entries, 0, 8);

        // 8 (begin) + 8 (end) + 2 (expr_len) + 2 (expr) + 16 (terminator) = 36
        assert_eq!(bytes.len(), 36);

        // Verify begin_offset.
        let begin = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        assert_eq!(begin, 0x10);

        // Verify end_offset.
        let end = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(end, 0x20);

        // Verify expression length.
        let expr_len = u16::from_le_bytes([bytes[16], bytes[17]]);
        assert_eq!(expr_len, 2);

        // Verify expression bytes.
        assert_eq!(bytes[18], DW_OP_FBREG);
        assert_eq!(bytes[19], 0x00);

        // Verify terminator at end.
        let term_start = 20;
        assert!(bytes[term_start..term_start + 16].iter().all(|&b| b == 0));
    }

    #[test]
    fn test_location_list_single_entry_4() {
        let entries = vec![LocationListEntry {
            begin_offset: 0x10,
            end_offset: 0x20,
            expression: vec![0xAB],
        }];
        let bytes = serialize_location_list(&entries, 0, 4);

        // 4 (begin) + 4 (end) + 2 (expr_len) + 1 (expr) + 8 (terminator) = 19
        assert_eq!(bytes.len(), 19);
    }

    // -----------------------------------------------------------------------
    // DwarfSectionBuilder / DwarfSections
    // -----------------------------------------------------------------------

    #[test]
    fn test_builder_new() {
        let builder = DwarfSectionBuilder::new(8);
        assert_eq!(builder.address_size, 8);
        assert!(builder.debug_info.is_empty());
        assert!(builder.debug_line.is_empty());
        assert!(builder.debug_frame.is_empty());
        assert!(builder.debug_aranges.is_empty());
        assert!(builder.debug_loc.is_empty());
    }

    #[test]
    fn test_builder_finalize_empty() {
        let builder = DwarfSectionBuilder::new(4);
        let sections = builder.finalize();
        assert!(sections.debug_info.is_empty());
        // Abbreviation table with zero entries serializes to just the
        // terminator byte.
        assert_eq!(sections.debug_abbrev, vec![0x00]);
        assert!(sections.debug_line.is_empty());
        assert!(sections.debug_str.is_empty());
        assert!(sections.debug_aranges.is_empty());
        assert!(sections.debug_frame.is_empty());
        assert!(sections.debug_loc.is_empty());
    }

    #[test]
    fn test_builder_with_strings_and_abbrevs() {
        let mut builder = DwarfSectionBuilder::new(8);
        builder.string_table.add("test.c");
        builder.string_table.add("bcc 0.1.0");
        builder.abbrev_table.add_abbreviation(
            DW_TAG_COMPILE_UNIT,
            true,
            vec![(DW_AT_NAME, DW_FORM_STRP)],
        );
        let sections = builder.finalize();

        // .debug_str should contain both strings.
        assert!(sections.debug_str.len() > 0);
        assert!(sections.debug_str.windows(6).any(|w| w == b"test.c"));
        assert!(sections
            .debug_str
            .windows(9)
            .any(|w| w == b"bcc 0.1.0"));

        // .debug_abbrev should have one entry + terminator.
        assert!(sections.debug_abbrev.len() > 1);
        assert_eq!(*sections.debug_abbrev.last().unwrap(), 0x00);
    }

    // -----------------------------------------------------------------------
    // DWARF constants sanity checks
    // -----------------------------------------------------------------------

    #[test]
    fn test_dwarf_version_constant() {
        assert_eq!(DWARF_VERSION, 4);
    }

    #[test]
    fn test_children_constants() {
        assert_eq!(DW_CHILDREN_yes, 1);
        assert_eq!(DW_CHILDREN_no, 0);
    }
}
