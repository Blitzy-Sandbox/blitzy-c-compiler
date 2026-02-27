//! Integration tests for DWARF v4 debug information generation.
//!
//! These tests verify that the `bcc` compiler generates correct DWARF v4 debug
//! sections when the `-g` flag is specified, with sufficient fidelity for
//! source-level debugging in GDB and LLDB.
//!
//! # Test Categories
//!
//! 1. **Section Presence** — Verifies `.debug_info`, `.debug_abbrev`, `.debug_line`,
//!    `.debug_str`, `.debug_aranges`, `.debug_frame` sections exist with `-g` and
//!    are absent without it.
//! 2. **DWARF Version** — Confirms the DWARF version field is exactly 4.
//! 3. **Compilation Unit DIEs** — Verifies `DW_TAG_compile_unit` with required
//!    attributes (`DW_AT_name`, `DW_AT_language`, `DW_AT_producer`, etc.).
//! 4. **Subprogram DIEs** — Verifies `DW_TAG_subprogram` entries for functions,
//!    `DW_TAG_formal_parameter` for parameters, and `DW_TAG_variable` for locals.
//! 5. **Type DIEs** — Verifies `DW_TAG_base_type`, `DW_TAG_pointer_type`,
//!    `DW_TAG_structure_type`, `DW_TAG_array_type`, and `DW_TAG_typedef`.
//! 6. **Line Number Program** — Verifies `.debug_line` header and source mappings.
//! 7. **Frame Information** — Verifies `.debug_frame` with CIE and FDE entries.
//! 8. **Multi-Architecture** — Verifies DWARF generation across all four targets.
//!
//! # Zero-Dependency Guarantee
//!
//! This module uses ONLY the Rust standard library (`std`). No external crates.

mod common;

use std::fs;
use std::path::Path;
use std::process::Command;

// ============================================================================
// DWARF v4 Tag Constants (per DWARF Debugging Information Format Version 4)
// ============================================================================

#[allow(non_upper_case_globals)]
const DW_TAG_array_type: u64 = 0x01;
#[allow(non_upper_case_globals)]
const DW_TAG_formal_parameter: u64 = 0x05;
#[allow(non_upper_case_globals)]
const DW_TAG_member: u64 = 0x0d;
#[allow(non_upper_case_globals)]
const DW_TAG_pointer_type: u64 = 0x0f;
#[allow(non_upper_case_globals)]
const DW_TAG_compile_unit: u64 = 0x11;
#[allow(non_upper_case_globals)]
const DW_TAG_structure_type: u64 = 0x13;
#[allow(non_upper_case_globals)]
const DW_TAG_typedef: u64 = 0x16;
#[allow(non_upper_case_globals)]
const DW_TAG_subrange_type: u64 = 0x21;
#[allow(non_upper_case_globals)]
const DW_TAG_base_type: u64 = 0x24;
#[allow(non_upper_case_globals)]
const DW_TAG_subprogram: u64 = 0x2e;
#[allow(non_upper_case_globals)]
const DW_TAG_variable: u64 = 0x34;

// ============================================================================
// DWARF v4 Attribute Constants
// ============================================================================

#[allow(non_upper_case_globals)]
const DW_AT_name: u64 = 0x03;
#[allow(non_upper_case_globals)]
const DW_AT_byte_size: u64 = 0x0b;
#[allow(non_upper_case_globals)]
const DW_AT_stmt_list: u64 = 0x10;
#[allow(non_upper_case_globals)]
const DW_AT_low_pc: u64 = 0x11;
#[allow(non_upper_case_globals)]
const DW_AT_high_pc: u64 = 0x12;
#[allow(non_upper_case_globals)]
const DW_AT_language: u64 = 0x13;
#[allow(non_upper_case_globals)]
const DW_AT_comp_dir: u64 = 0x1b;
#[allow(non_upper_case_globals)]
const DW_AT_producer: u64 = 0x25;
#[allow(non_upper_case_globals)]
const DW_AT_encoding: u64 = 0x3e;
#[allow(non_upper_case_globals)]
const DW_AT_type: u64 = 0x49;

// ============================================================================
// DWARF v4 Form Constants
// ============================================================================

const DW_FORM_ADDR: u64 = 0x01;
const DW_FORM_BLOCK2: u64 = 0x03;
const DW_FORM_BLOCK4: u64 = 0x04;
const DW_FORM_DATA2: u64 = 0x05;
const DW_FORM_DATA4: u64 = 0x06;
const DW_FORM_DATA8: u64 = 0x07;
const DW_FORM_STRING: u64 = 0x08;
const DW_FORM_BLOCK: u64 = 0x09;
const DW_FORM_BLOCK1: u64 = 0x0a;
const DW_FORM_DATA1: u64 = 0x0b;
const DW_FORM_FLAG: u64 = 0x0c;
const DW_FORM_SDATA: u64 = 0x0d;
const DW_FORM_STRP: u64 = 0x0e;
const DW_FORM_UDATA: u64 = 0x0f;
const DW_FORM_REF_ADDR: u64 = 0x10;
const DW_FORM_REF1: u64 = 0x11;
const DW_FORM_REF2: u64 = 0x12;
const DW_FORM_REF4: u64 = 0x13;
const DW_FORM_REF8: u64 = 0x14;
const DW_FORM_REF_UDATA: u64 = 0x15;
const DW_FORM_INDIRECT: u64 = 0x16;
const DW_FORM_SEC_OFFSET: u64 = 0x17;
const DW_FORM_EXPRLOC: u64 = 0x18;
const DW_FORM_FLAG_PRESENT: u64 = 0x19;
const DW_FORM_REF_SIG8: u64 = 0x20;

// ============================================================================
// DWARF Language Constants
// ============================================================================

/// DW_LANG_C99 = 0x000c
#[allow(dead_code)]
const DW_LANG_C99: u16 = 0x000c;
/// DW_LANG_C11 = 0x001d
#[allow(dead_code)]
const DW_LANG_C11: u16 = 0x001d;

// ============================================================================
// Helper Structs
// ============================================================================

/// Parsed DWARF v4 compilation unit header from `.debug_info`.
#[derive(Debug, Clone)]
struct DwarfHeader {
    /// Total length of the compilation unit (excluding the length field itself).
    unit_length: u64,
    /// DWARF version number (should be 4 for DWARF v4).
    version: u16,
    /// Offset into the `.debug_abbrev` section for this CU's abbreviation table.
    debug_abbrev_offset: u64,
    /// Size of an address on the target architecture (4 or 8 bytes).
    address_size: u8,
    /// Total number of bytes consumed by this header (for advancing the read position).
    header_size: usize,
    /// Whether this is a 64-bit DWARF format (uses 12-byte initial length).
    is_64bit_dwarf: bool,
}

/// A single entry in the DWARF abbreviation table.
#[derive(Debug, Clone)]
struct AbbrevEntry {
    /// The abbreviation code (1-based; 0 terminates the table).
    code: u64,
    /// The DWARF tag (DW_TAG_*) this abbreviation represents.
    tag: u64,
    /// Whether this DIE has children.
    #[allow(dead_code)]
    has_children: bool,
    /// List of (attribute, form) pairs for this abbreviation.
    attrs: Vec<(u64, u64)>,
}

// ============================================================================
// Byte-Level Reading Helpers (Little-Endian)
// ============================================================================

/// Read a little-endian u16 from `data` at byte offset `off`.
fn read_u16_le(data: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([data[off], data[off + 1]])
}

/// Read a little-endian u32 from `data` at byte offset `off`.
fn read_u32_le(data: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

/// Read a little-endian u64 from `data` at byte offset `off`.
fn read_u64_le(data: &[u8], off: usize) -> u64 {
    u64::from_le_bytes([
        data[off],
        data[off + 1],
        data[off + 2],
        data[off + 3],
        data[off + 4],
        data[off + 5],
        data[off + 6],
        data[off + 7],
    ])
}

// ============================================================================
// LEB128 Decoding
// ============================================================================

/// Decode an unsigned LEB128 value from `data` starting at `*pos`.
/// Advances `*pos` past the encoded bytes.
fn read_uleb128(data: &[u8], pos: &mut usize) -> u64 {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    loop {
        if *pos >= data.len() {
            break;
        }
        let byte = data[*pos];
        *pos += 1;
        result |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            break;
        }
    }
    result
}

/// Decode a signed LEB128 value from `data` starting at `*pos`.
/// Advances `*pos` past the encoded bytes.
#[allow(dead_code)]
fn read_sleb128(data: &[u8], pos: &mut usize) -> i64 {
    let mut result: i64 = 0;
    let mut shift: u32 = 0;
    let mut byte: u8;
    loop {
        if *pos >= data.len() {
            break;
        }
        byte = data[*pos];
        *pos += 1;
        result |= ((byte & 0x7f) as i64) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            // Sign extend if the high bit of the last byte is set.
            if shift < 64 && (byte & 0x40) != 0 {
                result |= !0i64 << shift;
            }
            break;
        }
        if shift >= 64 {
            break;
        }
    }
    result
}

// ============================================================================
// ELF Section Parsing Helpers
// ============================================================================

/// Find a named section in an ELF binary, returning `(file_offset, size)`.
///
/// Supports both ELF32 and ELF64 formats by inspecting the ELF class byte.
/// Returns `None` if the section is not found or the binary is malformed.
fn find_elf_section(binary: &[u8], name: &str) -> Option<(usize, usize)> {
    if binary.len() < 52 {
        return None;
    }
    // Verify ELF magic.
    if &binary[0..4] != b"\x7fELF" {
        return None;
    }
    let elf_class = binary[4]; // 1 = ELF32, 2 = ELF64

    // Read ELF header fields based on class.
    let (sh_off, sh_entsize, sh_num, sh_strndx) = if elf_class == 2 {
        // ELF64: e_shoff at 40 (8 bytes), e_shentsize at 58, e_shnum at 60, e_shstrndx at 62
        if binary.len() < 64 {
            return None;
        }
        let sh_off = read_u64_le(binary, 40) as usize;
        let sh_entsize = read_u16_le(binary, 58) as usize;
        let sh_num = read_u16_le(binary, 60) as usize;
        let sh_strndx = read_u16_le(binary, 62) as usize;
        (sh_off, sh_entsize, sh_num, sh_strndx)
    } else if elf_class == 1 {
        // ELF32: e_shoff at 32 (4 bytes), e_shentsize at 46, e_shnum at 48, e_shstrndx at 50
        if binary.len() < 52 {
            return None;
        }
        let sh_off = read_u32_le(binary, 32) as usize;
        let sh_entsize = read_u16_le(binary, 46) as usize;
        let sh_num = read_u16_le(binary, 48) as usize;
        let sh_strndx = read_u16_le(binary, 50) as usize;
        (sh_off, sh_entsize, sh_num, sh_strndx)
    } else {
        return None;
    };

    if sh_off == 0 || sh_num == 0 || sh_strndx >= sh_num {
        return None;
    }
    if sh_off + sh_num * sh_entsize > binary.len() {
        return None;
    }

    // Locate the section header string table (.shstrtab).
    let shstrtab_hdr_off = sh_off + sh_strndx * sh_entsize;
    let (shstrtab_off, shstrtab_sz) = if elf_class == 2 {
        (
            read_u64_le(binary, shstrtab_hdr_off + 24) as usize,
            read_u64_le(binary, shstrtab_hdr_off + 32) as usize,
        )
    } else {
        (
            read_u32_le(binary, shstrtab_hdr_off + 16) as usize,
            read_u32_le(binary, shstrtab_hdr_off + 20) as usize,
        )
    };

    if shstrtab_off + shstrtab_sz > binary.len() {
        return None;
    }
    let shstrtab = &binary[shstrtab_off..shstrtab_off + shstrtab_sz];

    // Iterate over all section headers to find the one matching `name`.
    for i in 0..sh_num {
        let hdr_off = sh_off + i * sh_entsize;
        let name_idx = read_u32_le(binary, hdr_off) as usize;

        if name_idx >= shstrtab.len() {
            continue;
        }

        // Extract the null-terminated section name from the string table.
        let sec_name_end = shstrtab[name_idx..]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(shstrtab.len() - name_idx);
        let sec_name = &shstrtab[name_idx..name_idx + sec_name_end];

        if sec_name == name.as_bytes() {
            let (offset, size) = if elf_class == 2 {
                (
                    read_u64_le(binary, hdr_off + 24) as usize,
                    read_u64_le(binary, hdr_off + 32) as usize,
                )
            } else {
                (
                    read_u32_le(binary, hdr_off + 16) as usize,
                    read_u32_le(binary, hdr_off + 20) as usize,
                )
            };
            return Some((offset, size));
        }
    }

    None
}

/// Retrieve a slice of the ELF binary corresponding to a named section.
///
/// Returns `None` if the section is not found.
fn get_section_data<'a>(binary: &'a [u8], name: &str) -> Option<&'a [u8]> {
    find_elf_section(binary, name).and_then(|(off, sz)| {
        if off + sz <= binary.len() {
            Some(&binary[off..off + sz])
        } else {
            None
        }
    })
}

/// Check whether an ELF binary file on disk contains a section with the given name.
///
/// Reads the binary from `path`, parses ELF headers, and searches for the section.
/// Returns `false` if the file cannot be read or the section is not found.
#[allow(dead_code)]
fn has_section(path: &Path, section_name: &str) -> bool {
    match fs::read(path) {
        Ok(data) => find_elf_section(&data, section_name).is_some(),
        Err(_) => false,
    }
}

// ============================================================================
// DWARF Parsing Helpers
// ============================================================================

/// Parse a DWARF v4 compilation unit header from the beginning of `.debug_info` data.
fn parse_debug_info_header(data: &[u8]) -> DwarfHeader {
    assert!(
        data.len() >= 11,
        "Data too short for DWARF compilation unit header: {} bytes",
        data.len()
    );

    let mut pos = 0;
    let initial_word = read_u32_le(data, pos);
    pos += 4;

    let (unit_length, is_64bit_dwarf) = if initial_word == 0xFFFF_FFFF {
        // 64-bit DWARF format: 12-byte initial length field.
        assert!(
            data.len() >= 23,
            "Data too short for 64-bit DWARF CU header"
        );
        let length = read_u64_le(data, pos);
        pos += 8;
        (length, true)
    } else {
        (initial_word as u64, false)
    };

    let version = read_u16_le(data, pos);
    pos += 2;

    let debug_abbrev_offset = if is_64bit_dwarf {
        let off = read_u64_le(data, pos);
        pos += 8;
        off
    } else {
        let off = read_u32_le(data, pos) as u64;
        pos += 4;
        off
    };

    let address_size = data[pos];
    pos += 1;

    DwarfHeader {
        unit_length,
        version,
        debug_abbrev_offset,
        address_size,
        header_size: pos,
        is_64bit_dwarf,
    }
}

/// Parse the abbreviation table from `.debug_abbrev` section data.
fn parse_abbrev_table(data: &[u8]) -> Vec<AbbrevEntry> {
    let mut entries = Vec::new();
    let mut pos = 0;

    loop {
        if pos >= data.len() {
            break;
        }
        let code = read_uleb128(data, &mut pos);
        if code == 0 {
            break;
        }

        let tag = read_uleb128(data, &mut pos);
        let has_children = if pos < data.len() {
            let b = data[pos];
            pos += 1;
            b != 0
        } else {
            false
        };

        let mut attrs = Vec::new();
        loop {
            if pos >= data.len() {
                break;
            }
            let attr = read_uleb128(data, &mut pos);
            let form = read_uleb128(data, &mut pos);
            if attr == 0 && form == 0 {
                break;
            }
            attrs.push((attr, form));
        }

        entries.push(AbbrevEntry {
            code,
            tag,
            has_children,
            attrs,
        });
    }

    entries
}

/// Parse the abbreviation table starting at a specific offset within the section.
fn parse_abbrev_table_at(data: &[u8], offset: u64) -> Vec<AbbrevEntry> {
    let off = offset as usize;
    if off >= data.len() {
        return Vec::new();
    }
    parse_abbrev_table(&data[off..])
}

/// Check whether any abbreviation in the table has the specified DWARF tag.
fn abbrev_has_tag(table: &[AbbrevEntry], tag: u64) -> bool {
    table.iter().any(|e| e.tag == tag)
}

/// Find the first abbreviation entry with the specified DWARF tag.
fn abbrev_with_tag<'a>(table: &'a [AbbrevEntry], tag: u64) -> Option<&'a AbbrevEntry> {
    table.iter().find(|e| e.tag == tag)
}

/// Check whether an abbreviation with the given tag declares the given attribute.
fn abbrev_tag_has_attr(table: &[AbbrevEntry], tag: u64, attr: u64) -> bool {
    table
        .iter()
        .filter(|e| e.tag == tag)
        .any(|e| e.attrs.iter().any(|&(a, _)| a == attr))
}

/// Skip past a DWARF form value at `*pos` in the data buffer.
fn skip_form_value(
    form: u64,
    address_size: u8,
    is_dwarf64: bool,
    data: &[u8],
    pos: &mut usize,
) {
    match form {
        DW_FORM_ADDR => *pos += address_size as usize,
        DW_FORM_BLOCK1 => {
            if *pos < data.len() {
                let len = data[*pos] as usize;
                *pos += 1 + len;
            }
        }
        DW_FORM_BLOCK2 => {
            if *pos + 2 <= data.len() {
                let len = read_u16_le(data, *pos) as usize;
                *pos += 2 + len;
            }
        }
        DW_FORM_BLOCK4 => {
            if *pos + 4 <= data.len() {
                let len = read_u32_le(data, *pos) as usize;
                *pos += 4 + len;
            }
        }
        DW_FORM_BLOCK | DW_FORM_EXPRLOC => {
            let len = read_uleb128(data, pos) as usize;
            *pos += len;
        }
        DW_FORM_DATA1 | DW_FORM_REF1 | DW_FORM_FLAG => *pos += 1,
        DW_FORM_DATA2 | DW_FORM_REF2 => *pos += 2,
        DW_FORM_DATA4 | DW_FORM_REF4 => *pos += 4,
        DW_FORM_DATA8 | DW_FORM_REF8 | DW_FORM_REF_SIG8 => *pos += 8,
        DW_FORM_SDATA => {
            let _ = read_sleb128(data, pos);
        }
        DW_FORM_UDATA | DW_FORM_REF_UDATA => {
            let _ = read_uleb128(data, pos);
        }
        DW_FORM_STRING => {
            while *pos < data.len() && data[*pos] != 0 {
                *pos += 1;
            }
            if *pos < data.len() {
                *pos += 1;
            }
        }
        DW_FORM_STRP | DW_FORM_SEC_OFFSET => {
            *pos += if is_dwarf64 { 8 } else { 4 };
        }
        DW_FORM_REF_ADDR => {
            *pos += if is_dwarf64 { 8 } else { 4 };
        }
        DW_FORM_FLAG_PRESENT => { /* zero bytes */ }
        DW_FORM_INDIRECT => {
            let actual_form = read_uleb128(data, pos);
            skip_form_value(actual_form, address_size, is_dwarf64, data, pos);
        }
        _ => { /* unknown form — cannot advance safely */ }
    }
}

/// Walk DIEs in a `.debug_info` compilation unit and check if any has the target tag.
fn debug_info_contains_tag(
    debug_info: &[u8],
    debug_abbrev: &[u8],
    target_tag: u64,
) -> bool {
    if debug_info.is_empty() || debug_abbrev.is_empty() {
        return false;
    }

    let header = parse_debug_info_header(debug_info);
    let abbrev_table = parse_abbrev_table_at(debug_abbrev, header.debug_abbrev_offset);

    if !abbrev_has_tag(&abbrev_table, target_tag) {
        return false;
    }

    // Compute CU end boundary.
    let length_field_size = if header.is_64bit_dwarf { 12 } else { 4 };
    let cu_end = (length_field_size + header.unit_length as usize).min(debug_info.len());
    let mut pos = header.header_size;

    while pos < cu_end {
        let abbrev_code = read_uleb128(debug_info, &mut pos);
        if abbrev_code == 0 {
            continue;
        }

        let entry = match abbrev_table.iter().find(|e| e.code == abbrev_code) {
            Some(e) => e,
            None => break,
        };

        if entry.tag == target_tag {
            return true;
        }

        for &(_, form) in &entry.attrs {
            skip_form_value(
                form,
                header.address_size,
                header.is_64bit_dwarf,
                debug_info,
                &mut pos,
            );
        }
    }

    false
}

/// Check whether the `.debug_str` section contains a specific substring.
fn debug_str_contains(binary: &[u8], needle: &str) -> bool {
    if let Some(debug_str) = get_section_data(binary, ".debug_str") {
        let needle_bytes = needle.as_bytes();
        if needle_bytes.is_empty() {
            return true;
        }
        debug_str
            .windows(needle_bytes.len())
            .any(|window| window == needle_bytes)
    } else {
        false
    }
}

/// Check if `.debug_info` section raw bytes contain a given string (DW_FORM_string).
fn debug_info_raw_contains(binary: &[u8], needle: &str) -> bool {
    if let Some(debug_info) = get_section_data(binary, ".debug_info") {
        let needle_bytes = needle.as_bytes();
        if needle_bytes.is_empty() {
            return true;
        }
        debug_info
            .windows(needle_bytes.len())
            .any(|window| window == needle_bytes)
    } else {
        false
    }
}

/// Check if any debug section (`.debug_info` or `.debug_str`) contains a string.
fn debug_sections_contain_string(binary: &[u8], needle: &str) -> bool {
    debug_str_contains(binary, needle) || debug_info_raw_contains(binary, needle)
}

/// Count ELF sections whose name starts with a given prefix.
fn count_sections_with_prefix(binary: &[u8], prefix: &str) -> usize {
    if binary.len() < 52 || &binary[0..4] != b"\x7fELF" {
        return 0;
    }

    let elf_class = binary[4];
    let (sh_off, sh_entsize, sh_num, sh_strndx) = if elf_class == 2 {
        if binary.len() < 64 {
            return 0;
        }
        (
            read_u64_le(binary, 40) as usize,
            read_u16_le(binary, 58) as usize,
            read_u16_le(binary, 60) as usize,
            read_u16_le(binary, 62) as usize,
        )
    } else if elf_class == 1 {
        (
            read_u32_le(binary, 32) as usize,
            read_u16_le(binary, 46) as usize,
            read_u16_le(binary, 48) as usize,
            read_u16_le(binary, 50) as usize,
        )
    } else {
        return 0;
    };

    if sh_off == 0 || sh_num == 0 || sh_strndx >= sh_num {
        return 0;
    }
    if sh_off + sh_num * sh_entsize > binary.len() {
        return 0;
    }

    let shstrtab_hdr_off = sh_off + sh_strndx * sh_entsize;
    let (shstrtab_off, shstrtab_sz) = if elf_class == 2 {
        (
            read_u64_le(binary, shstrtab_hdr_off + 24) as usize,
            read_u64_le(binary, shstrtab_hdr_off + 32) as usize,
        )
    } else {
        (
            read_u32_le(binary, shstrtab_hdr_off + 16) as usize,
            read_u32_le(binary, shstrtab_hdr_off + 20) as usize,
        )
    };

    if shstrtab_off + shstrtab_sz > binary.len() {
        return 0;
    }
    let shstrtab = &binary[shstrtab_off..shstrtab_off + shstrtab_sz];

    let mut count = 0;
    for i in 0..sh_num {
        let hdr_off = sh_off + i * sh_entsize;
        let name_idx = read_u32_le(binary, hdr_off) as usize;
        if name_idx >= shstrtab.len() {
            continue;
        }
        let name_end = shstrtab[name_idx..]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(shstrtab.len() - name_idx);
        let sec_name = &shstrtab[name_idx..name_idx + name_end];
        if sec_name.starts_with(prefix.as_bytes()) {
            count += 1;
        }
    }

    count
}

// ============================================================================
// DWARF Line Program Header Helper
// ============================================================================

/// Parsed DWARF v4 line number program header from `.debug_line`.
#[derive(Debug)]
#[allow(dead_code)]
struct LineHeader {
    unit_length: u64,
    version: u16,
    header_length: u64,
    minimum_instruction_length: u8,
    maximum_operations_per_instruction: u8,
    default_is_stmt: u8,
    line_base: i8,
    line_range: u8,
    opcode_base: u8,
    header_size: usize,
    is_64bit_dwarf: bool,
}

/// Parse a DWARF v4 line number program header.
fn parse_line_header(data: &[u8]) -> Option<LineHeader> {
    if data.len() < 15 {
        return None;
    }

    let mut pos = 0;
    let initial_word = read_u32_le(data, pos);
    pos += 4;

    let (unit_length, is_64bit_dwarf) = if initial_word == 0xFFFF_FFFF {
        if data.len() < 23 {
            return None;
        }
        let length = read_u64_le(data, pos);
        pos += 8;
        (length, true)
    } else {
        (initial_word as u64, false)
    };

    if pos + 2 > data.len() {
        return None;
    }
    let version = read_u16_le(data, pos);
    pos += 2;

    let header_length = if is_64bit_dwarf {
        if pos + 8 > data.len() {
            return None;
        }
        let hl = read_u64_le(data, pos);
        pos += 8;
        hl
    } else {
        if pos + 4 > data.len() {
            return None;
        }
        let hl = read_u32_le(data, pos) as u64;
        pos += 4;
        hl
    };

    if pos + 6 > data.len() {
        return None;
    }

    let minimum_instruction_length = data[pos];
    pos += 1;

    let maximum_operations_per_instruction = if version >= 4 {
        let v = data[pos];
        pos += 1;
        v
    } else {
        1
    };

    let default_is_stmt = data[pos];
    pos += 1;

    let line_base = data[pos] as i8;
    pos += 1;

    let line_range = data[pos];
    pos += 1;

    let opcode_base = data[pos];
    pos += 1;

    Some(LineHeader {
        unit_length,
        version,
        header_length,
        minimum_instruction_length,
        maximum_operations_per_instruction,
        default_is_stmt,
        line_base,
        line_range,
        opcode_base,
        header_size: pos,
        is_64bit_dwarf,
    })
}

// ============================================================================
// Compilation Helpers for Tests
// ============================================================================

/// Compile C source with `-g` flag, returning the raw binary bytes.
fn compile_with_debug(source: &str, extra_flags: &[&str]) -> Vec<u8> {
    let mut flags: Vec<&str> = vec!["-g"];
    flags.extend_from_slice(extra_flags);
    let result = common::compile_source(source, &flags);
    assert!(
        result.success,
        "Compilation with -g failed:\nstderr: {}\nstdout: {}",
        result.stderr, result.stdout
    );
    let binary_path = result
        .output_path
        .as_ref()
        .expect("No output binary produced after successful compilation");
    fs::read(binary_path).unwrap_or_else(|e| {
        panic!(
            "Failed to read compiled binary '{}': {}",
            binary_path.display(),
            e
        )
    })
}

/// Compile C source WITHOUT `-g` flag, returning the raw binary bytes.
fn compile_without_debug(source: &str) -> Vec<u8> {
    let result = common::compile_source(source, &[]);
    assert!(
        result.success,
        "Compilation without -g failed:\nstderr: {}\nstdout: {}",
        result.stderr, result.stdout
    );
    let binary_path = result
        .output_path
        .as_ref()
        .expect("No output binary produced");
    fs::read(binary_path).unwrap_or_else(|e| {
        panic!(
            "Failed to read compiled binary '{}': {}",
            binary_path.display(),
            e
        )
    })
}

/// Compile C source with `-g` and a specific target architecture.
fn compile_debug_for_target(source: &str, target: &str) -> Vec<u8> {
    compile_with_debug(source, &["--target", target])
}

// ============================================================================
// C Source Code Snippets for Tests
// ============================================================================

/// Minimal C program for basic DWARF section presence tests.
const SIMPLE_MAIN: &str = r#"
int main(void) {
    return 0;
}
"#;

/// C program with a named function and parameters for subprogram DIE tests.
const FUNCTION_WITH_PARAMS: &str = r#"
int add(int a, int b) {
    return a + b;
}

int main(void) {
    return add(1, 2);
}
"#;

/// C program with local variables for variable DIE tests.
const FUNCTION_WITH_LOCALS: &str = r#"
int compute(int x) {
    int result = x * 2;
    int offset = 10;
    return result + offset;
}

int main(void) {
    return compute(5);
}
"#;

/// C program with multiple types for type DIE tests.
const TYPES_PROGRAM: &str = r#"
typedef unsigned long size_type;

struct Point {
    int x;
    int y;
};

int main(void) {
    int i = 42;
    char c = 'A';
    float f = 3.14f;
    double d = 2.718;
    long l = 100000L;
    int *p = &i;
    struct Point pt;
    pt.x = 1;
    pt.y = 2;
    int arr[10];
    arr[0] = i;
    size_type s = 42;
    (void)c;
    (void)f;
    (void)d;
    (void)l;
    (void)p;
    (void)s;
    return pt.x + pt.y + arr[0];
}
"#;

/// Multi-line, multi-function C program for line number program tests.
const MULTILINE_PROGRAM: &str = r#"
int square(int x) {
    return x * x;
}

int cube(int x) {
    int sq = square(x);
    return sq * x;
}

int main(void) {
    int a = 3;
    int b = square(a);
    int c = cube(a);
    return b + c;
}
"#;

/// C program with function calls for frame information tests.
const FRAME_PROGRAM: &str = r#"
int deep_function(int a, int b, int c, int d) {
    int local1 = a + b;
    int local2 = c + d;
    int local3 = local1 * local2;
    return local3;
}

int main(void) {
    return deep_function(1, 2, 3, 4);
}
"#;

// ============================================================================
// Phase 2: Debug Section Presence Tests
// ============================================================================

/// Compile with `-g` and verify that the output binary contains all expected
/// DWARF v4 debug sections.
#[test]
fn debug_sections_present_with_g_flag() {
    let binary = compile_with_debug(SIMPLE_MAIN, &[]);

    assert!(
        find_elf_section(&binary, ".debug_info").is_some(),
        "Missing .debug_info section in binary compiled with -g"
    );
    assert!(
        find_elf_section(&binary, ".debug_abbrev").is_some(),
        "Missing .debug_abbrev section in binary compiled with -g"
    );
    assert!(
        find_elf_section(&binary, ".debug_line").is_some(),
        "Missing .debug_line section in binary compiled with -g"
    );
    assert!(
        find_elf_section(&binary, ".debug_str").is_some(),
        "Missing .debug_str section in binary compiled with -g"
    );
    assert!(
        find_elf_section(&binary, ".debug_frame").is_some(),
        "Missing .debug_frame section in binary compiled with -g"
    );

    // .debug_aranges is optional but expected.
    if find_elf_section(&binary, ".debug_aranges").is_none() {
        eprintln!(
            "Note: .debug_aranges section not present (optional but recommended)"
        );
    }
}

/// Compile WITHOUT `-g` and verify that no `.debug_*` sections exist.
#[test]
fn no_debug_sections_without_g_flag() {
    let binary = compile_without_debug(SIMPLE_MAIN);

    let debug_section_count = count_sections_with_prefix(&binary, ".debug_");
    assert_eq!(
        debug_section_count, 0,
        "Found {} .debug_* sections in binary compiled WITHOUT -g",
        debug_section_count
    );
}

/// Parse `.debug_info` header and verify DWARF version is 4.
#[test]
fn debug_info_version_4() {
    let binary = compile_with_debug(SIMPLE_MAIN, &[]);
    let debug_info = get_section_data(&binary, ".debug_info")
        .expect("Missing .debug_info section");

    let header = parse_debug_info_header(debug_info);
    assert_eq!(
        header.version, 4,
        "Expected DWARF version 4, got version {}",
        header.version
    );
}

// ============================================================================
// Phase 3: Compilation Unit DIE Tests
// ============================================================================

/// Verify `.debug_info` contains a `DW_TAG_compile_unit` DIE with required attributes.
#[test]
fn debug_info_compilation_unit() {
    let binary = compile_with_debug(SIMPLE_MAIN, &[]);

    let debug_info = get_section_data(&binary, ".debug_info")
        .expect("Missing .debug_info section");
    let debug_abbrev = get_section_data(&binary, ".debug_abbrev")
        .expect("Missing .debug_abbrev section");

    let header = parse_debug_info_header(debug_info);
    let abbrev_table = parse_abbrev_table_at(debug_abbrev, header.debug_abbrev_offset);

    // Verify DW_TAG_compile_unit is present in abbreviation table.
    assert!(
        abbrev_has_tag(&abbrev_table, DW_TAG_compile_unit),
        "DW_TAG_compile_unit not found in abbreviation table"
    );

    let cu_entry = abbrev_with_tag(&abbrev_table, DW_TAG_compile_unit)
        .expect("DW_TAG_compile_unit abbreviation entry not found");

    // Verify required attributes are declared in the abbreviation.
    let required_attrs = [
        (DW_AT_name, "DW_AT_name"),
        (DW_AT_language, "DW_AT_language"),
        (DW_AT_comp_dir, "DW_AT_comp_dir"),
        (DW_AT_producer, "DW_AT_producer"),
        (DW_AT_stmt_list, "DW_AT_stmt_list"),
    ];

    for (attr, name) in &required_attrs {
        assert!(
            cu_entry.attrs.iter().any(|&(a, _)| a == *attr),
            "DW_TAG_compile_unit missing required attribute {}",
            name
        );
    }

    // Verify the producer string mentions "bcc" somewhere in debug strings.
    assert!(
        debug_sections_contain_string(&binary, "bcc"),
        "DWARF debug info does not contain 'bcc' producer string"
    );
}

// ============================================================================
// Phase 4: Subprogram DIE Tests
// ============================================================================

/// Verify `DW_TAG_subprogram` DIE with function name and address range attributes.
#[test]
fn debug_info_function() {
    let binary = compile_with_debug(FUNCTION_WITH_PARAMS, &[]);

    let debug_info = get_section_data(&binary, ".debug_info")
        .expect("Missing .debug_info section");
    let debug_abbrev = get_section_data(&binary, ".debug_abbrev")
        .expect("Missing .debug_abbrev section");

    let header = parse_debug_info_header(debug_info);
    let abbrev_table = parse_abbrev_table_at(debug_abbrev, header.debug_abbrev_offset);

    assert!(
        abbrev_has_tag(&abbrev_table, DW_TAG_subprogram),
        "DW_TAG_subprogram not found — functions should have debug info"
    );

    // Verify subprogram has name, low_pc, and high_pc attributes.
    assert!(
        abbrev_tag_has_attr(&abbrev_table, DW_TAG_subprogram, DW_AT_name),
        "DW_TAG_subprogram missing DW_AT_name"
    );
    assert!(
        abbrev_tag_has_attr(&abbrev_table, DW_TAG_subprogram, DW_AT_low_pc),
        "DW_TAG_subprogram missing DW_AT_low_pc"
    );
    assert!(
        abbrev_tag_has_attr(&abbrev_table, DW_TAG_subprogram, DW_AT_high_pc),
        "DW_TAG_subprogram missing DW_AT_high_pc"
    );

    // Verify function names appear in debug strings.
    assert!(
        debug_sections_contain_string(&binary, "add"),
        "Function name 'add' not found in DWARF debug strings"
    );
    assert!(
        debug_sections_contain_string(&binary, "main"),
        "Function name 'main' not found in DWARF debug strings"
    );
}

/// Verify `DW_TAG_formal_parameter` DIEs for function parameters.
#[test]
fn debug_info_function_parameters() {
    let binary = compile_with_debug(FUNCTION_WITH_PARAMS, &[]);

    let debug_abbrev = get_section_data(&binary, ".debug_abbrev")
        .expect("Missing .debug_abbrev section");
    let debug_info = get_section_data(&binary, ".debug_info")
        .expect("Missing .debug_info section");

    let header = parse_debug_info_header(debug_info);
    let abbrev_table = parse_abbrev_table_at(debug_abbrev, header.debug_abbrev_offset);

    assert!(
        abbrev_has_tag(&abbrev_table, DW_TAG_formal_parameter),
        "DW_TAG_formal_parameter not found — function parameters should have debug info"
    );
    assert!(
        abbrev_tag_has_attr(&abbrev_table, DW_TAG_formal_parameter, DW_AT_name),
        "DW_TAG_formal_parameter missing DW_AT_name"
    );
    assert!(
        abbrev_tag_has_attr(&abbrev_table, DW_TAG_formal_parameter, DW_AT_type),
        "DW_TAG_formal_parameter missing DW_AT_type"
    );
}

/// Verify `DW_TAG_variable` DIEs for local variables within function scope.
#[test]
fn debug_info_local_variables() {
    let binary = compile_with_debug(FUNCTION_WITH_LOCALS, &[]);

    let debug_abbrev = get_section_data(&binary, ".debug_abbrev")
        .expect("Missing .debug_abbrev section");
    let debug_info = get_section_data(&binary, ".debug_info")
        .expect("Missing .debug_info section");

    let header = parse_debug_info_header(debug_info);
    let abbrev_table = parse_abbrev_table_at(debug_abbrev, header.debug_abbrev_offset);

    assert!(
        abbrev_has_tag(&abbrev_table, DW_TAG_variable),
        "DW_TAG_variable not found — local variables should have debug info"
    );
    assert!(
        abbrev_tag_has_attr(&abbrev_table, DW_TAG_variable, DW_AT_name),
        "DW_TAG_variable missing DW_AT_name"
    );
    assert!(
        abbrev_tag_has_attr(&abbrev_table, DW_TAG_variable, DW_AT_type),
        "DW_TAG_variable missing DW_AT_type"
    );

    // Verify local variable names appear in debug strings.
    assert!(
        debug_sections_contain_string(&binary, "result"),
        "Local variable 'result' not found in DWARF debug strings"
    );
    assert!(
        debug_sections_contain_string(&binary, "offset"),
        "Local variable 'offset' not found in DWARF debug strings"
    );
}

// ============================================================================
// Phase 5: Type DIE Tests
// ============================================================================

/// Verify `DW_TAG_base_type` DIEs for fundamental C types with correct attributes.
#[test]
fn debug_info_base_types() {
    let binary = compile_with_debug(TYPES_PROGRAM, &[]);

    let debug_abbrev = get_section_data(&binary, ".debug_abbrev")
        .expect("Missing .debug_abbrev section");
    let debug_info = get_section_data(&binary, ".debug_info")
        .expect("Missing .debug_info section");

    let header = parse_debug_info_header(debug_info);
    let abbrev_table = parse_abbrev_table_at(debug_abbrev, header.debug_abbrev_offset);

    // Verify DW_TAG_base_type is present with required attributes.
    assert!(
        abbrev_has_tag(&abbrev_table, DW_TAG_base_type),
        "DW_TAG_base_type not found"
    );
    assert!(
        abbrev_tag_has_attr(&abbrev_table, DW_TAG_base_type, DW_AT_name),
        "DW_TAG_base_type missing DW_AT_name"
    );
    assert!(
        abbrev_tag_has_attr(&abbrev_table, DW_TAG_base_type, DW_AT_byte_size),
        "DW_TAG_base_type missing DW_AT_byte_size"
    );
    assert!(
        abbrev_tag_has_attr(&abbrev_table, DW_TAG_base_type, DW_AT_encoding),
        "DW_TAG_base_type missing DW_AT_encoding"
    );

    // Verify fundamental type names appear in debug strings.
    assert!(
        debug_sections_contain_string(&binary, "int"),
        "Base type 'int' not found in DWARF debug strings"
    );
    assert!(
        debug_sections_contain_string(&binary, "char"),
        "Base type 'char' not found in DWARF debug strings"
    );
}

/// Verify `DW_TAG_pointer_type` DIE for pointer variables.
#[test]
fn debug_info_pointer_type() {
    let binary = compile_with_debug(TYPES_PROGRAM, &[]);

    let debug_abbrev = get_section_data(&binary, ".debug_abbrev")
        .expect("Missing .debug_abbrev section");
    let debug_info = get_section_data(&binary, ".debug_info")
        .expect("Missing .debug_info section");

    let header = parse_debug_info_header(debug_info);
    let abbrev_table = parse_abbrev_table_at(debug_abbrev, header.debug_abbrev_offset);

    assert!(
        abbrev_has_tag(&abbrev_table, DW_TAG_pointer_type),
        "DW_TAG_pointer_type not found"
    );
    assert!(
        abbrev_tag_has_attr(&abbrev_table, DW_TAG_pointer_type, DW_AT_type),
        "DW_TAG_pointer_type missing DW_AT_type"
    );
}

/// Verify `DW_TAG_structure_type` DIE with `DW_TAG_member` children for struct members.
#[test]
fn debug_info_struct_type() {
    let binary = compile_with_debug(TYPES_PROGRAM, &[]);

    let debug_abbrev = get_section_data(&binary, ".debug_abbrev")
        .expect("Missing .debug_abbrev section");
    let debug_info = get_section_data(&binary, ".debug_info")
        .expect("Missing .debug_info section");

    let header = parse_debug_info_header(debug_info);
    let abbrev_table = parse_abbrev_table_at(debug_abbrev, header.debug_abbrev_offset);

    assert!(
        abbrev_has_tag(&abbrev_table, DW_TAG_structure_type),
        "DW_TAG_structure_type not found"
    );
    assert!(
        abbrev_has_tag(&abbrev_table, DW_TAG_member),
        "DW_TAG_member not found — struct members should have debug info"
    );
    assert!(
        abbrev_tag_has_attr(&abbrev_table, DW_TAG_structure_type, DW_AT_name),
        "DW_TAG_structure_type missing DW_AT_name"
    );

    assert!(
        debug_sections_contain_string(&binary, "Point"),
        "Struct name 'Point' not found in DWARF debug strings"
    );
}

/// Verify `DW_TAG_array_type` DIE with `DW_TAG_subrange_type` for bounds.
#[test]
fn debug_info_array_type() {
    let binary = compile_with_debug(TYPES_PROGRAM, &[]);

    let debug_abbrev = get_section_data(&binary, ".debug_abbrev")
        .expect("Missing .debug_abbrev section");
    let debug_info = get_section_data(&binary, ".debug_info")
        .expect("Missing .debug_info section");

    let header = parse_debug_info_header(debug_info);
    let abbrev_table = parse_abbrev_table_at(debug_abbrev, header.debug_abbrev_offset);

    assert!(
        abbrev_has_tag(&abbrev_table, DW_TAG_array_type),
        "DW_TAG_array_type not found"
    );
    assert!(
        abbrev_has_tag(&abbrev_table, DW_TAG_subrange_type),
        "DW_TAG_subrange_type not found — array bounds should be recorded"
    );
}

/// Verify `DW_TAG_typedef` DIE for typedef declarations.
#[test]
fn debug_info_typedef() {
    let binary = compile_with_debug(TYPES_PROGRAM, &[]);

    let debug_abbrev = get_section_data(&binary, ".debug_abbrev")
        .expect("Missing .debug_abbrev section");
    let debug_info = get_section_data(&binary, ".debug_info")
        .expect("Missing .debug_info section");

    let header = parse_debug_info_header(debug_info);
    let abbrev_table = parse_abbrev_table_at(debug_abbrev, header.debug_abbrev_offset);

    assert!(
        abbrev_has_tag(&abbrev_table, DW_TAG_typedef),
        "DW_TAG_typedef not found"
    );
    assert!(
        abbrev_tag_has_attr(&abbrev_table, DW_TAG_typedef, DW_AT_name),
        "DW_TAG_typedef missing DW_AT_name"
    );
    assert!(
        abbrev_tag_has_attr(&abbrev_table, DW_TAG_typedef, DW_AT_type),
        "DW_TAG_typedef missing DW_AT_type"
    );

    assert!(
        debug_sections_contain_string(&binary, "size_type"),
        "Typedef name 'size_type' not found in DWARF debug strings"
    );
}

// ============================================================================
// Phase 6: Line Number Program Tests
// ============================================================================

/// Verify `.debug_line` exists with a valid DWARF v4 header.
#[test]
fn debug_line_program_present() {
    let binary = compile_with_debug(SIMPLE_MAIN, &[]);
    let debug_line = get_section_data(&binary, ".debug_line")
        .expect("Missing .debug_line section");

    let header = parse_line_header(debug_line)
        .expect("Failed to parse .debug_line header");

    assert_eq!(
        header.version, 4,
        "Expected DWARF line program version 4, got {}",
        header.version
    );
    assert!(
        header.minimum_instruction_length >= 1,
        "minimum_instruction_length should be >= 1"
    );
    assert!(header.line_range > 0, "line_range should be > 0");
    assert!(header.opcode_base > 0, "opcode_base should be > 0");
}

/// Verify the line number program has non-trivial content mapping for a multi-line program.
#[test]
fn debug_line_source_mapping() {
    let binary = compile_with_debug(MULTILINE_PROGRAM, &[]);
    let debug_line = get_section_data(&binary, ".debug_line")
        .expect("Missing .debug_line section");

    let header = parse_line_header(debug_line)
        .expect("Failed to parse .debug_line header");

    // The line program should have substantial content for a multi-line, multi-function program.
    let min_expected = header.header_size + 20;
    assert!(
        debug_line.len() >= min_expected,
        ".debug_line too small ({} bytes) for multi-line program",
        debug_line.len()
    );
}

/// Verify the file table in `.debug_line` references a .c source file.
#[test]
fn debug_line_file_table() {
    let binary = compile_with_debug(SIMPLE_MAIN, &[]);
    let debug_line = get_section_data(&binary, ".debug_line")
        .expect("Missing .debug_line section");

    // The file table should contain a .c file extension somewhere in its data.
    let has_c_file = debug_line.windows(2).any(|w| w == b".c");
    assert!(
        has_c_file,
        "No .c source file name found in .debug_line file table"
    );
}

/// Verify line mappings span multiple functions in a multi-function program.
#[test]
fn debug_line_multiple_functions() {
    let binary = compile_with_debug(MULTILINE_PROGRAM, &[]);
    let debug_line = get_section_data(&binary, ".debug_line")
        .expect("Missing .debug_line section");

    let header = parse_line_header(debug_line)
        .expect("Failed to parse .debug_line header");

    // Calculate program body size (excluding header) — should be non-trivial
    // for a program with three functions (square, cube, main).
    let program_body_start = header.header_size + header.header_length as usize;
    let program_body_size = if program_body_start < debug_line.len() {
        debug_line.len() - program_body_start
    } else {
        0
    };
    assert!(
        program_body_size >= 10,
        "Line program body too small ({} bytes) for multi-function program",
        program_body_size
    );
}

// ============================================================================
// Phase 7: Frame Information Tests
// ============================================================================

/// Verify `.debug_frame` section exists with CIE and FDE entries.
#[test]
fn debug_frame_present() {
    let binary = compile_with_debug(FRAME_PROGRAM, &[]);
    let debug_frame = get_section_data(&binary, ".debug_frame")
        .expect("Missing .debug_frame section");

    assert!(!debug_frame.is_empty(), ".debug_frame section is empty");
    assert!(
        debug_frame.len() >= 8,
        ".debug_frame too small for a CIE entry"
    );

    // The first entry in .debug_frame should be a CIE.
    // For 32-bit DWARF, CIE_id = 0xFFFFFFFF.
    let first_length = read_u32_le(debug_frame, 0);
    if first_length != 0xFFFF_FFFF {
        // 32-bit DWARF format — check CIE_id at offset 4.
        let cie_id = read_u32_le(debug_frame, 4);
        assert_eq!(
            cie_id, 0xFFFF_FFFF,
            "Expected CIE_id = 0xFFFFFFFF, got 0x{:08X}",
            cie_id
        );
    }
}

/// Verify CFI entries are present for stack unwinding (at least one CIE + one FDE).
#[test]
fn debug_frame_cfi() {
    let binary = compile_with_debug(FRAME_PROGRAM, &[]);
    let debug_frame = get_section_data(&binary, ".debug_frame")
        .expect("Missing .debug_frame section");

    assert!(
        debug_frame.len() >= 20,
        ".debug_frame too small ({} bytes) for CIE + FDE",
        debug_frame.len()
    );

    // Walk through .debug_frame entries counting CIEs and FDEs.
    let mut pos = 0;
    let mut cie_count = 0;
    let mut fde_count = 0;

    while pos + 4 <= debug_frame.len() {
        let length = read_u32_le(debug_frame, pos);
        if length == 0 {
            break; // Terminator entry.
        }

        let is_64bit = length == 0xFFFF_FFFF;
        let (entry_length, id_offset) = if is_64bit {
            if pos + 12 > debug_frame.len() {
                break;
            }
            (read_u64_le(debug_frame, pos + 4) as usize, pos + 12)
        } else {
            (length as usize, pos + 4)
        };

        if id_offset + 4 > debug_frame.len() {
            break;
        }

        if is_64bit {
            let cie_id = read_u64_le(debug_frame, id_offset);
            if cie_id == 0xFFFF_FFFF_FFFF_FFFF {
                cie_count += 1;
            } else {
                fde_count += 1;
            }
            pos = id_offset + entry_length;
        } else {
            let cie_id = read_u32_le(debug_frame, id_offset);
            if cie_id == 0xFFFF_FFFF {
                cie_count += 1;
            } else {
                fde_count += 1;
            }
            pos = pos + 4 + entry_length;
        }
    }

    assert!(
        cie_count >= 1,
        "No CIE entries found in .debug_frame"
    );
    assert!(
        fde_count >= 1,
        "No FDE entries found in .debug_frame ({} CIEs found)",
        cie_count
    );
}

// ============================================================================
// Phase 8: Multi-Architecture DWARF Tests
// ============================================================================

/// Verify DWARF generation for x86-64 target (64-bit addresses in DIEs).
#[test]
fn dwarf_x86_64() {
    let binary = compile_debug_for_target(SIMPLE_MAIN, common::TARGET_X86_64);

    // Verify ELF class is 64-bit.
    assert_eq!(
        binary[4],
        common::ELFCLASS64,
        "Expected ELF64 for x86-64"
    );

    // Verify DWARF sections are present.
    assert!(
        find_elf_section(&binary, ".debug_info").is_some(),
        "Missing .debug_info in x86-64 binary"
    );
    assert!(
        find_elf_section(&binary, ".debug_abbrev").is_some(),
        "Missing .debug_abbrev in x86-64 binary"
    );
    assert!(
        find_elf_section(&binary, ".debug_line").is_some(),
        "Missing .debug_line in x86-64 binary"
    );

    // Verify DWARF header metadata.
    let debug_info = get_section_data(&binary, ".debug_info")
        .expect("Failed to read .debug_info");
    let header = parse_debug_info_header(debug_info);
    assert_eq!(
        header.version, 4,
        "Expected DWARF v4 for x86-64"
    );
    assert_eq!(
        header.address_size, 8,
        "Expected 8-byte addresses for x86-64, got {}",
        header.address_size
    );
}

/// Verify DWARF generation for i686 target (32-bit addresses in DIEs).
#[test]
fn dwarf_i686() {
    let binary = compile_debug_for_target(SIMPLE_MAIN, common::TARGET_I686);

    assert_eq!(
        binary[4],
        common::ELFCLASS32,
        "Expected ELF32 for i686"
    );

    assert!(
        find_elf_section(&binary, ".debug_info").is_some(),
        "Missing .debug_info in i686 binary"
    );
    assert!(
        find_elf_section(&binary, ".debug_abbrev").is_some(),
        "Missing .debug_abbrev in i686 binary"
    );
    assert!(
        find_elf_section(&binary, ".debug_line").is_some(),
        "Missing .debug_line in i686 binary"
    );

    let debug_info = get_section_data(&binary, ".debug_info")
        .expect("Failed to read .debug_info");
    let header = parse_debug_info_header(debug_info);
    assert_eq!(header.version, 4, "Expected DWARF v4 for i686");
    assert_eq!(
        header.address_size, 4,
        "Expected 4-byte addresses for i686, got {}",
        header.address_size
    );
}

/// Verify DWARF generation for AArch64 target.
#[test]
fn dwarf_aarch64() {
    let binary = compile_debug_for_target(SIMPLE_MAIN, common::TARGET_AARCH64);

    assert_eq!(
        binary[4],
        common::ELFCLASS64,
        "Expected ELF64 for AArch64"
    );

    assert!(
        find_elf_section(&binary, ".debug_info").is_some(),
        "Missing .debug_info in AArch64 binary"
    );
    assert!(
        find_elf_section(&binary, ".debug_abbrev").is_some(),
        "Missing .debug_abbrev in AArch64 binary"
    );
    assert!(
        find_elf_section(&binary, ".debug_line").is_some(),
        "Missing .debug_line in AArch64 binary"
    );

    let debug_info = get_section_data(&binary, ".debug_info")
        .expect("Failed to read .debug_info");
    let header = parse_debug_info_header(debug_info);
    assert_eq!(header.version, 4, "Expected DWARF v4 for AArch64");
    assert_eq!(
        header.address_size, 8,
        "Expected 8-byte addresses for AArch64, got {}",
        header.address_size
    );
}

/// Verify DWARF generation for RISC-V 64 target.
#[test]
fn dwarf_riscv64() {
    let binary = compile_debug_for_target(SIMPLE_MAIN, common::TARGET_RISCV64);

    assert_eq!(
        binary[4],
        common::ELFCLASS64,
        "Expected ELF64 for RISC-V 64"
    );

    assert!(
        find_elf_section(&binary, ".debug_info").is_some(),
        "Missing .debug_info in RISC-V 64 binary"
    );
    assert!(
        find_elf_section(&binary, ".debug_abbrev").is_some(),
        "Missing .debug_abbrev in RISC-V 64 binary"
    );
    assert!(
        find_elf_section(&binary, ".debug_line").is_some(),
        "Missing .debug_line in RISC-V 64 binary"
    );

    let debug_info = get_section_data(&binary, ".debug_info")
        .expect("Failed to read .debug_info");
    let header = parse_debug_info_header(debug_info);
    assert_eq!(header.version, 4, "Expected DWARF v4 for RISC-V 64");
    assert_eq!(
        header.address_size, 8,
        "Expected 8-byte addresses for RISC-V 64, got {}",
        header.address_size
    );
}

// ============================================================================
// Additional Verification Tests
// ============================================================================

/// Verify `-g` flag is accepted by the bcc compiler using Command directly.
#[test]
fn debug_flag_accepted() {
    let bcc = common::get_bcc_binary();
    let source = common::write_temp_source(SIMPLE_MAIN);
    let dir = common::TempDir::new("debug_flag");
    let output = dir.path().join("test_debug");

    let result = Command::new(&bcc)
        .arg("-g")
        .arg("-o")
        .arg(&output)
        .arg(source.path())
        .output()
        .expect("Failed to execute bcc");

    assert!(
        result.status.success(),
        "bcc failed with -g flag:\nstderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(output.exists(), "Output binary not produced with -g");
}

/// Verify DWARF info is generated alongside optimization flags.
#[test]
fn debug_with_optimization_levels() {
    for opt_level in &["-O0", "-O2"] {
        let binary = compile_with_debug(SIMPLE_MAIN, &[opt_level]);

        assert!(
            find_elf_section(&binary, ".debug_info").is_some(),
            "Missing .debug_info with -g {}",
            opt_level
        );
        assert!(
            find_elf_section(&binary, ".debug_line").is_some(),
            "Missing .debug_line with -g {}",
            opt_level
        );
        assert!(
            find_elf_section(&binary, ".debug_abbrev").is_some(),
            "Missing .debug_abbrev with -g {}",
            opt_level
        );

        let debug_info = get_section_data(&binary, ".debug_info")
            .expect("Missing .debug_info data");
        let header = parse_debug_info_header(debug_info);
        assert_eq!(
            header.version, 4,
            "Expected DWARF v4 at {}, got {}",
            opt_level, header.version
        );
    }
}

/// Verify `.debug_str` section contains recognizable type and identifier strings.
#[test]
fn debug_str_section_content() {
    let binary = compile_with_debug(TYPES_PROGRAM, &[]);
    let debug_str = get_section_data(&binary, ".debug_str")
        .expect("Missing .debug_str section");

    assert!(!debug_str.is_empty(), ".debug_str section is empty");

    // Check that .debug_str contains at least some recognizable strings.
    let content = String::from_utf8_lossy(debug_str);
    let has_types = content.contains("int")
        || content.contains("char")
        || content.contains("Point");
    assert!(
        has_types,
        ".debug_str has no recognizable type names. Sample: {:?}",
        &content[..content.len().min(200)]
    );
}

/// Verify `.debug_abbrev` section has valid structure with sequential codes.
#[test]
fn debug_abbrev_structure() {
    let binary = compile_with_debug(SIMPLE_MAIN, &[]);
    let debug_abbrev = get_section_data(&binary, ".debug_abbrev")
        .expect("Missing .debug_abbrev section");

    assert!(!debug_abbrev.is_empty(), ".debug_abbrev is empty");

    let table = parse_abbrev_table(debug_abbrev);
    assert!(!table.is_empty(), "Abbreviation table is empty");

    // First abbreviation code should be 1.
    assert_eq!(
        table[0].code, 1,
        "First abbreviation code should be 1, got {}",
        table[0].code
    );

    // No abbreviation should have code 0 (reserved as terminator).
    for entry in &table {
        assert_ne!(
            entry.code, 0,
            "Abbreviation code 0 is reserved and should not appear as an entry"
        );
    }
}

/// Verify DIE walking finds expected tags in the `.debug_info` section.
#[test]
fn debug_info_die_walk_tags() {
    let binary = compile_with_debug(TYPES_PROGRAM, &[]);
    let debug_info = get_section_data(&binary, ".debug_info")
        .expect("Missing .debug_info section");
    let debug_abbrev = get_section_data(&binary, ".debug_abbrev")
        .expect("Missing .debug_abbrev section");

    assert!(
        debug_info_contains_tag(debug_info, debug_abbrev, DW_TAG_compile_unit),
        "DW_TAG_compile_unit not found during DIE walk"
    );
    assert!(
        debug_info_contains_tag(debug_info, debug_abbrev, DW_TAG_base_type),
        "DW_TAG_base_type not found during DIE walk"
    );
    assert!(
        debug_info_contains_tag(debug_info, debug_abbrev, DW_TAG_variable),
        "DW_TAG_variable not found during DIE walk"
    );
}

/// Verify ELF validity of binary produced with `-g`.
#[test]
fn debug_binary_elf_validity() {
    let result = common::compile_source(SIMPLE_MAIN, &["-g"]);
    assert!(result.success, "Compilation failed: {}", result.stderr);

    let path = result.output_path.as_ref().expect("No output path");
    common::verify_elf_magic(path);
    common::verify_elf_class(path, common::ELFCLASS64);
    common::verify_elf_arch(path, common::EM_X86_64);
}

/// Verify ELF architecture identification in multi-arch debug binaries.
/// This ensures that the correct ELF machine architecture constants are
/// applied for each target when debug info is generated.
#[test]
fn debug_multiarch_elf_arch_verification() {
    // x86-64: verify EM_X86_64 (already done in debug_binary_elf_validity,
    // but we also verify across architectures with explicit CompileResult binding).
    let result_x86: common::CompileResult = common::compile_source(SIMPLE_MAIN, &["-g", "--target", common::TARGET_X86_64]);
    assert!(result_x86.success, "x86-64 compilation failed");
    assert!(result_x86.exit_status.success(), "Expected exit status 0 for x86-64");
    let path_x86 = result_x86.output_path.as_ref().expect("No x86-64 output");
    common::verify_elf_arch(path_x86, common::EM_X86_64);

    // i686: verify EM_386
    let result_i686: common::CompileResult = common::compile_source(SIMPLE_MAIN, &["-g", "--target", common::TARGET_I686]);
    assert!(result_i686.success, "i686 compilation failed: {}", result_i686.stderr);
    assert!(result_i686.exit_status.success(), "Expected exit status 0 for i686");
    let path_i686 = result_i686.output_path.as_ref().expect("No i686 output");
    common::verify_elf_arch(path_i686, common::EM_386);

    // AArch64: verify EM_AARCH64
    let result_aarch64: common::CompileResult = common::compile_source(SIMPLE_MAIN, &["-g", "--target", common::TARGET_AARCH64]);
    assert!(result_aarch64.success, "AArch64 compilation failed: {}", result_aarch64.stderr);
    assert!(result_aarch64.exit_status.success(), "Expected exit status 0 for AArch64");
    let path_aarch64 = result_aarch64.output_path.as_ref().expect("No AArch64 output");
    common::verify_elf_arch(path_aarch64, common::EM_AARCH64);

    // RISC-V 64: verify EM_RISCV
    let result_riscv: common::CompileResult = common::compile_source(SIMPLE_MAIN, &["-g", "--target", common::TARGET_RISCV64]);
    assert!(result_riscv.success, "RISC-V 64 compilation failed: {}", result_riscv.stderr);
    assert!(result_riscv.exit_status.success(), "Expected exit status 0 for RISC-V 64");
    let path_riscv = result_riscv.output_path.as_ref().expect("No RISC-V 64 output");
    common::verify_elf_arch(path_riscv, common::EM_RISCV);
}

/// Verify DWARF debug compilation using direct Command invocation with args()
/// and status() methods, and manual source file creation with fs::write().
#[test]
fn debug_manual_compile_workflow() {
    let bcc = common::get_bcc_binary();
    let dir = common::TempDir::new("debug_manual");
    let src_path = dir.path().join("manual_test.c");
    let out_path = dir.path().join("manual_test");

    // Use fs::write to create the source file manually.
    fs::write(&src_path, SIMPLE_MAIN).expect("Failed to write source file");

    // Use Command.args() and Command.status() methods.
    let status = Command::new(&bcc)
        .args(&["-g", "-o"])
        .arg(&out_path)
        .arg(&src_path)
        .status()
        .expect("Failed to execute bcc");

    assert!(status.success(), "bcc compilation failed with status: {:?}", status);
    assert!(out_path.exists(), "Output binary not created at {:?}", out_path);

    // Verify the produced binary contains debug sections.
    let binary = fs::read(&out_path).expect("Failed to read output binary");
    assert!(
        find_elf_section(&binary, ".debug_info").is_some(),
        "Missing .debug_info in manually compiled binary"
    );
    assert!(
        find_elf_section(&binary, ".debug_line").is_some(),
        "Missing .debug_line in manually compiled binary"
    );
}
