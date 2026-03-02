//! # Section Merging and Layout
//!
//! This module combines identically-named sections from multiple input objects into
//! merged output sections (e.g., all `.text` sections into one `.text`), computes
//! section addresses and file offsets with proper alignment, generates the overall
//! memory layout for `.bss`, `.rodata`, `.data`, and handles section-to-segment
//! mapping for PT_LOAD program headers.
//!
//! ## Key Responsibilities
//! - **Section Merging**: Concatenate sections with the same name from different
//!   input objects, respecting alignment constraints and recording input-to-output
//!   offset mappings for relocation patching.
//! - **Layout Computation**: Assign virtual addresses and file offsets to merged
//!   sections, inserting page-aligned boundaries at segment transitions.
//! - **BSS Handling**: `.bss` sections occupy memory but not file space, and must
//!   appear at the end of the read-write segment.
//! - **Permission Classification**: Map section flags to permission categories
//!   (ReadExecute, ReadOnly, ReadWrite, None) for PT_LOAD segment grouping.
//! - **Section Header String Table**: Build the `.shstrtab` section with
//!   deduplicated, null-terminated section names.

use std::collections::HashMap;

// =============================================================================
// ELF Section Type Constants (sh_type)
// =============================================================================

/// Null section header (inactive).
pub const SHT_NULL: u32 = 0;
/// Section holds program-defined information (code, initialized data).
pub const SHT_PROGBITS: u32 = 1;
/// Section holds a symbol table (.symtab).
pub const SHT_SYMTAB: u32 = 2;
/// Section holds a string table (.strtab, .shstrtab, .dynstr).
pub const SHT_STRTAB: u32 = 3;
/// Section holds relocation entries with explicit addends (.rela.*).
pub const SHT_RELA: u32 = 4;
/// Section holds a hash table (.hash) for symbol lookup.
pub const SHT_HASH: u32 = 5;
/// Section holds dynamic linking information (.dynamic).
pub const SHT_DYNAMIC: u32 = 6;
/// Section holds auxiliary note information (.note.*).
pub const SHT_NOTE: u32 = 7;
/// Section occupies no file space but contributes to memory image (.bss).
/// BSS sections have their data zeroed in memory at load time.
pub const SHT_NOBITS: u32 = 8;
/// Section holds relocation entries without explicit addends (.rel.*).
pub const SHT_REL: u32 = 9;
/// Section holds a dynamic linker symbol table (.dynsym).
pub const SHT_DYNSYM: u32 = 11;

// =============================================================================
// ELF Section Flag Constants (sh_flags)
// =============================================================================

/// Section data should be writable during execution.
pub const SHF_WRITE: u64 = 0x1;
/// Section occupies memory during process execution (allocatable).
pub const SHF_ALLOC: u64 = 0x2;
/// Section contains executable machine instructions.
pub const SHF_EXECINSTR: u64 = 0x4;
/// Section data may be merged to eliminate duplication.
pub const SHF_MERGE: u64 = 0x10;
/// Section consists of null-terminated strings.
pub const SHF_STRINGS: u64 = 0x20;

// =============================================================================
// Input Section Representation
// =============================================================================

/// An input section from a single compiled object file.
///
/// Each input object contributes zero or more sections (`.text`, `.data`, `.bss`,
/// `.rodata`, etc.). During linking, sections with the same name from different
/// objects are merged together. This struct preserves the origin information
/// needed for relocation offset translation after merging.
#[derive(Debug, Clone)]
pub struct InputSection {
    /// Section name (e.g., ".text", ".data", ".rodata", ".bss").
    pub name: String,
    /// Raw section data bytes. Empty for `.bss` sections (SHT_NOBITS).
    pub data: Vec<u8>,
    /// ELF section type. Common values:
    /// - `SHT_PROGBITS` (1): code or initialized data
    /// - `SHT_NOBITS` (8): uninitialized data (.bss)
    pub section_type: u32,
    /// ELF section flags. Bitwise combination of:
    /// - `SHF_ALLOC`: occupies memory during execution
    /// - `SHF_WRITE`: writable data
    /// - `SHF_EXECINSTR`: executable code
    pub flags: u64,
    /// Required alignment in bytes. Must be a power of 2.
    /// Critical for AArch64 and RISC-V which have strict alignment requirements.
    pub alignment: u64,
    /// Size in memory. May differ from `data.len()` for `.bss` sections where
    /// `mem_size > 0` but `data` is empty.
    pub mem_size: u64,
    /// Index of the object file this section came from. Used to correlate
    /// relocations back to their source object during offset translation.
    pub object_index: usize,
    /// Original section index within the source object file.
    pub original_index: usize,
}

// =============================================================================
// Merged Section Representation
// =============================================================================

/// A merged output section containing data from multiple input sections.
///
/// After merging, each output section holds the concatenated data from all
/// input sections of the same name, with alignment padding inserted between
/// inputs as needed. The `input_mappings` field records where each input
/// section was placed, which is essential for relocation adjustment.
#[derive(Debug, Clone)]
pub struct MergedSection {
    /// Section name (e.g., ".text", ".data").
    pub name: String,
    /// Merged section data. For `.bss` sections, this is empty.
    pub data: Vec<u8>,
    /// ELF section type (inherited from input sections).
    pub section_type: u32,
    /// ELF section flags (union of all input section flags).
    pub flags: u64,
    /// Maximum alignment from all input sections contributing to this merge.
    pub alignment: u64,
    /// Total size in memory. For `.bss`: `mem_size > data.len()` because
    /// `.bss` occupies memory but not file space.
    pub mem_size: u64,
    /// Assigned virtual address in the output binary (computed during layout).
    pub virtual_address: u64,
    /// Assigned file offset in the output binary (computed during layout).
    pub file_offset: u64,
    /// Mapping of input sections to their offsets within this merged section.
    /// Used by the relocation processor to translate input-relative offsets
    /// to merged-section-relative offsets.
    pub input_mappings: Vec<InputSectionMapping>,
}

/// Records where an input section was placed within a merged output section.
///
/// During relocation processing, a relocation's offset is relative to the
/// original input section. This mapping provides the translation needed to
/// compute the offset within the merged output section:
/// `merged_offset = output_offset + original_offset_within_input`.
#[derive(Debug, Clone)]
pub struct InputSectionMapping {
    /// Index of the object file this input section came from.
    pub object_index: usize,
    /// Original section index within that object file.
    pub original_index: usize,
    /// Byte offset within the merged section where this input section starts.
    pub output_offset: u64,
    /// Size of this input section in bytes.
    pub size: u64,
}

// =============================================================================
// Layout Result
// =============================================================================

/// Result of section layout computation, providing the total file and memory
/// sizes needed to construct the ELF output.
#[derive(Debug, Clone)]
pub struct LayoutResult {
    /// Total file size in bytes (excludes BSS which has no file backing).
    pub total_file_size: u64,
    /// Total memory size in bytes (includes BSS).
    pub total_mem_size: u64,
}

// =============================================================================
// Section Permission Classification
// =============================================================================

/// Permission category for a section, determining which PT_LOAD segment
/// it belongs to in the ELF output.
///
/// Sections are grouped by permission so that each PT_LOAD segment has
/// uniform memory protection. This grouping is critical for memory-mapped
/// page protection enforced by the Linux kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SectionPermission {
    /// Read + Execute: `.text`, `.init`, `.fini`, `.plt` code sections.
    ReadExecute,
    /// Read-only: `.rodata`, `.eh_frame`, `.interp` data sections.
    ReadOnly,
    /// Read + Write: `.data`, `.bss`, `.got`, `.dynamic` mutable sections.
    ReadWrite,
    /// Not loaded into memory: `.debug_*`, `.symtab`, `.strtab` metadata.
    None,
}

/// Classify a section into its permission category based on ELF flags.
///
/// The classification follows standard ELF conventions:
/// - Non-allocatable sections (no `SHF_ALLOC`) are not loaded into memory.
/// - Executable sections (`SHF_EXECINSTR`) are read-execute.
/// - Writable sections (`SHF_WRITE`) are read-write.
/// - All other allocatable sections are read-only.
///
/// # Arguments
/// * `_name` - Section name (reserved for future fine-grained classification).
/// * `flags` - ELF section flags bitmask.
///
/// # Returns
/// The permission category for PT_LOAD segment grouping.
pub fn classify_section_permission(_name: &str, flags: u64) -> SectionPermission {
    if flags & SHF_ALLOC == 0 {
        SectionPermission::None
    } else if flags & SHF_EXECINSTR != 0 {
        SectionPermission::ReadExecute
    } else if flags & SHF_WRITE != 0 {
        SectionPermission::ReadWrite
    } else {
        SectionPermission::ReadOnly
    }
}

// =============================================================================
// Alignment Utilities
// =============================================================================

/// Align a value up to the specified alignment boundary.
///
/// Returns the smallest value >= `value` that is a multiple of `alignment`.
/// Alignment must be a power of 2 (or 0/1, in which case value is returned
/// unchanged).
///
/// # Examples
/// ```ignore
/// assert_eq!(align_up(0, 16), 0);
/// assert_eq!(align_up(1, 16), 16);
/// assert_eq!(align_up(16, 16), 16);
/// assert_eq!(align_up(17, 16), 32);
/// ```
///
/// # Panics
/// Debug-asserts that `alignment` is a power of 2 when it is > 1.
#[inline]
pub fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 || alignment == 1 {
        return value;
    }
    debug_assert!(
        alignment.is_power_of_two(),
        "alignment must be power of 2, got {}",
        alignment
    );
    (value + alignment - 1) & !(alignment - 1)
}

/// Calculate the number of padding bytes needed to align `value` up to
/// the specified alignment boundary.
///
/// Returns 0 if `value` is already aligned or alignment is 0/1.
///
/// # Examples
/// ```ignore
/// assert_eq!(padding_for_alignment(16, 16), 0);
/// assert_eq!(padding_for_alignment(17, 16), 15);
/// assert_eq!(padding_for_alignment(0, 4096), 0);
/// ```
#[inline]
pub fn padding_for_alignment(value: u64, alignment: u64) -> u64 {
    align_up(value, alignment) - value
}

// =============================================================================
// Section Merging
// =============================================================================

/// Merge identically-named sections from multiple input objects.
///
/// This is the core section merging algorithm used by the integrated linker.
/// All `.text` sections become one merged `.text`, all `.data` become one
/// merged `.data`, and so on.
///
/// Within each merged section, input sections are concatenated with alignment
/// padding between them. The `input_mappings` field records where each input
/// section landed, which is essential for relocation adjustment — relocations
/// reference offsets within the original input section, which must be translated
/// to offsets within the merged section.
///
/// # Arguments
/// * `input_sections` - All input sections from all object files.
/// * `section_order` - Preferred ordering of section names. Sections present
///   in this list appear first, in the specified order. Sections not in the
///   list appear afterward, sorted by name for determinism.
///
/// # Returns
/// A vector of `MergedSection` instances, ordered according to `section_order`
/// with unlisted sections appended alphabetically.
pub fn merge_sections(
    input_sections: Vec<InputSection>,
    section_order: &[&str],
) -> Vec<MergedSection> {
    // Step 1: Group input sections by name, preserving insertion order via Vec.
    // We use a HashMap to map section names to vectors of input sections.
    let mut groups: HashMap<String, Vec<InputSection>> = HashMap::new();
    let mut seen_order: Vec<String> = Vec::new();

    for section in input_sections {
        let name = section.name.clone();
        let entry = groups.entry(name.clone()).or_insert_with(|| {
            seen_order.push(name);
            Vec::new()
        });
        entry.push(section);
    }

    // Step 2: Determine output ordering.
    //
    // The explicit `section_order` list places well-known sections in the
    // standard ELF order: RX (.init, .text, .fini) → R (.rodata, .eh_frame)
    // → RW (.data, .bss) → non-alloc (.comment).
    //
    // When linking large libraries like libc.a, many additional sections
    // appear (e.g. .text.unlikely, .rodata.cst4, .data.rel.ro).  These
    // "remaining" sections must be interleaved into the correct permission
    // groups so that all RX sections are contiguous, all R sections are
    // contiguous, etc.  This is required for non-overlapping PT_LOAD
    // segments.
    //
    // Strategy: identify the last explicit section in each permission group,
    // and insert remaining sections of that group immediately after it.

    // Helper: determine permission key for a section group name.
    let perm_key_for = |name: &str| -> SectionPermission {
        if let Some(inputs) = groups.get(name) {
            if let Some(first) = inputs.first() {
                return classify_section_permission(name, first.flags);
            }
        }
        SectionPermission::None
    };

    // Collect explicitly-ordered sections present in the input.
    let explicit: Vec<String> = section_order
        .iter()
        .filter(|n| groups.contains_key(&n.to_string()))
        .map(|n| n.to_string())
        .collect();

    // Collect remaining sections, grouped by permission and sorted by name.
    let explicit_set: std::collections::HashSet<&str> =
        explicit.iter().map(|s| s.as_str()).collect();
    let mut rem_rx: Vec<String> = Vec::new();
    let mut rem_ro: Vec<String> = Vec::new();
    let mut rem_rw: Vec<String> = Vec::new();
    let mut rem_na: Vec<String> = Vec::new();
    for name in &seen_order {
        if explicit_set.contains(name.as_str()) {
            continue;
        }
        match perm_key_for(name) {
            SectionPermission::ReadExecute => rem_rx.push(name.clone()),
            SectionPermission::ReadOnly    => rem_ro.push(name.clone()),
            SectionPermission::ReadWrite   => rem_rw.push(name.clone()),
            SectionPermission::None        => rem_na.push(name.clone()),
        }
    }
    rem_rx.sort();
    rem_ro.sort();
    rem_rw.sort();
    rem_na.sort();

    // Build final ordering by interleaving remaining sections after the
    // last explicit section of each permission group.
    let mut ordered_names: Vec<String> = Vec::new();
    let mut emitted_rx = false;
    let mut emitted_ro = false;
    let mut emitted_rw = false;

    // Find the last explicit section index for each permission group
    // so we know where to insert remaining sections.
    let last_rx = explicit.iter().rposition(|n| perm_key_for(n) == SectionPermission::ReadExecute);
    let last_ro = explicit.iter().rposition(|n| perm_key_for(n) == SectionPermission::ReadOnly);
    let last_rw = explicit.iter().rposition(|n| perm_key_for(n) == SectionPermission::ReadWrite);

    for (idx, name) in explicit.iter().enumerate() {
        ordered_names.push(name.clone());
        // After the last explicit RX section, insert remaining RX sections.
        if !emitted_rx && last_rx == Some(idx) {
            ordered_names.extend(rem_rx.drain(..));
            emitted_rx = true;
        }
        // After the last explicit R section, insert remaining R sections.
        if !emitted_ro && last_ro == Some(idx) {
            ordered_names.extend(rem_ro.drain(..));
            emitted_ro = true;
        }
        // After the last explicit RW section, insert remaining RW sections.
        if !emitted_rw && last_rw == Some(idx) {
            ordered_names.extend(rem_rw.drain(..));
            emitted_rw = true;
        }
    }

    // If no explicit section existed for a permission group, append at end.
    if !emitted_rx { ordered_names.extend(rem_rx); }
    if !emitted_ro { ordered_names.extend(rem_ro); }
    if !emitted_rw { ordered_names.extend(rem_rw); }
    // Non-alloc sections always go last.
    ordered_names.extend(rem_na);

    // Step 3: For each group, merge the input sections into one output section.
    let mut result = Vec::with_capacity(ordered_names.len());

    for name in ordered_names {
        if let Some(inputs) = groups.remove(&name) {
            let merged = merge_section_group(&name, inputs);
            result.push(merged);
        }
    }

    result
}

/// Merge a group of input sections that share the same name into a single
/// merged output section.
///
/// For SHT_NOBITS (.bss) sections, data is not concatenated; instead, the
/// mem_size fields are summed (with alignment padding). For all other section
/// types, data bytes are concatenated with zero-byte padding for alignment.
fn merge_section_group(name: &str, inputs: Vec<InputSection>) -> MergedSection {
    let mut data = Vec::new();
    let mut input_mappings = Vec::new();
    let mut max_alignment: u64 = 1;
    let mut combined_flags: u64 = 0;
    let mut section_type: u32 = SHT_NULL;
    let mut total_mem_size: u64 = 0;
    let is_nobits = inputs.iter().any(|s| s.section_type == SHT_NOBITS);

    for input in inputs {
        // Track the maximum alignment requirement across all inputs
        if input.alignment > max_alignment {
            max_alignment = input.alignment;
        }

        // Combine flags (union of all input flags)
        combined_flags |= input.flags;

        // Use the section type from the first input with a non-null type,
        // or the first NOBITS section type if any input is NOBITS
        if section_type == SHT_NULL {
            section_type = input.section_type;
        }

        if is_nobits {
            // BSS merging: sizes are summed, no data concatenation.
            // Apply alignment padding to the memory offset
            let aligned_offset = align_up(total_mem_size, input.alignment.max(1));
            let padding = aligned_offset - total_mem_size;
            total_mem_size = aligned_offset;

            input_mappings.push(InputSectionMapping {
                object_index: input.object_index,
                original_index: input.original_index,
                output_offset: total_mem_size,
                size: input.mem_size,
            });

            total_mem_size += input.mem_size;
            // Ensure we account for any padding we added
            let _ = padding;
        } else {
            // PROGBITS merging: concatenate data with alignment padding.
            let current_offset = data.len() as u64;
            let aligned_offset = align_up(current_offset, input.alignment.max(1));
            let padding = (aligned_offset - current_offset) as usize;

            // Insert zero padding bytes for alignment
            data.extend(std::iter::repeat(0u8).take(padding));

            input_mappings.push(InputSectionMapping {
                object_index: input.object_index,
                original_index: input.original_index,
                output_offset: data.len() as u64,
                size: input.data.len() as u64,
            });

            data.extend_from_slice(&input.data);
        }
    }

    // For non-NOBITS sections, mem_size equals data length
    if !is_nobits {
        total_mem_size = data.len() as u64;
    }

    // Ensure section_type is set correctly for NOBITS
    if is_nobits {
        section_type = SHT_NOBITS;
    }

    MergedSection {
        name: name.to_string(),
        data,
        section_type,
        flags: combined_flags,
        alignment: max_alignment,
        mem_size: total_mem_size,
        virtual_address: 0,
        file_offset: 0,
        input_mappings,
    }
}

// =============================================================================
// Section Layout Computation
// =============================================================================

/// Assign virtual addresses and file offsets to all merged sections.
///
/// Starting after the ELF header and program headers (at `header_size`),
/// sections are laid out sequentially with proper alignment. When sections
/// transition between different PT_LOAD segments (different permission
/// categories), a page boundary is inserted to ensure each segment can
/// have its own memory protection attributes.
///
/// # Arguments
/// * `sections` - Mutable slice of merged sections. Each section's
///   `virtual_address` and `file_offset` fields are updated in place.
/// * `base_address` - Base virtual address for the output binary
///   (e.g., 0x400000 for 64-bit, 0x08048000 for 32-bit).
/// * `header_size` - Size of the ELF header plus program headers in bytes.
///   Sections start after this offset.
/// * `_is_64bit` - Whether the output is ELF64 (true) or ELF32 (false).
///   Reserved for future ELF-class-specific layout adjustments.
///
/// # Returns
/// A `LayoutResult` containing the total file size and total memory size.
///
/// # Page Alignment
/// When consecutive sections have different permission categories (e.g.,
/// transitioning from ReadExecute to ReadOnly), both file offset and
/// virtual address are aligned up to the page boundary (4096 bytes).
/// This ensures the kernel can apply per-page memory protection.
pub fn compute_layout(
    sections: &mut [MergedSection],
    base_address: u64,
    header_size: u64,
    _is_64bit: bool,
) -> LayoutResult {
    let page_size: u64 = 0x1000; // 4096 bytes

    let mut file_offset = header_size;
    let mut virtual_address = base_address + header_size;

    // Track the previous section's permission for segment boundary detection
    let mut prev_permission: Option<SectionPermission> = Option::None;

    for section in sections.iter_mut() {
        let current_permission = classify_section_permission(&section.name, section.flags);

        // Insert page alignment at segment transitions (permission changes).
        // Only consider allocatable sections for segment boundary detection.
        if current_permission != SectionPermission::None {
            if let Some(prev) = prev_permission {
                if prev != current_permission {
                    // Segment boundary: align to page size
                    file_offset = align_up(file_offset, page_size);
                    virtual_address = align_up(virtual_address, page_size);
                }
            }
            prev_permission = Some(current_permission);
        }

        // Align to section's own alignment requirement
        let section_align = section.alignment.max(1);
        file_offset = align_up(file_offset, section_align);
        virtual_address = align_up(virtual_address, section_align);

        // Assign addresses to this section
        section.file_offset = file_offset;
        section.virtual_address = virtual_address;

        if section.section_type == SHT_NOBITS {
            // BSS sections: occupy memory but not file space.
            // Only advance virtual address, not file offset.
            virtual_address += section.mem_size;
        } else {
            let section_file_size = section.data.len() as u64;
            file_offset += section_file_size;
            virtual_address += section_file_size;
        }
    }

    LayoutResult {
        total_file_size: file_offset,
        total_mem_size: virtual_address - base_address,
    }
}

// =============================================================================
// Relocation Offset Translation
// =============================================================================

/// Translate a relocation offset from input section coordinates to merged
/// section coordinates.
///
/// During relocation application, a relocation's offset field is relative to
/// the original input section. After section merging, this offset must be
/// translated to the corresponding position within the merged output section.
///
/// # Arguments
/// * `merged` - The merged section containing the input section mapping.
/// * `object_index` - Index of the object file the relocation belongs to.
/// * `original_section_index` - Original section index within that object.
/// * `offset` - Offset within the original input section.
///
/// # Returns
/// `Some(merged_offset)` if the mapping is found, where `merged_offset` is
/// the offset within the merged section. Returns `None` if no mapping matches
/// the given object/section indices.
pub fn translate_offset(
    merged: &MergedSection,
    object_index: usize,
    original_section_index: usize,
    offset: u64,
) -> Option<u64> {
    merged
        .input_mappings
        .iter()
        .find(|m| m.object_index == object_index && m.original_index == original_section_index)
        .map(|m| m.output_offset + offset)
}

// =============================================================================
// Section Header String Table (.shstrtab)
// =============================================================================

/// Builder for the ELF section header string table (`.shstrtab`).
///
/// The section header string table is a contiguous block of null-terminated
/// strings referenced by the `sh_name` field in each section header. This
/// builder provides O(1) deduplication of section names and returns offsets
/// suitable for direct use in ELF section headers.
///
/// Per the ELF specification, the table begins with a null byte at offset 0,
/// which is used for the null section header (SHN_UNDEF).
pub struct ShStrTab {
    /// Raw bytes of the string table, including null terminators.
    data: Vec<u8>,
    /// Map from section name strings to their byte offsets in `data`.
    /// Used for O(1) deduplication on repeated additions.
    offsets: HashMap<String, u32>,
}

impl ShStrTab {
    /// Create a new, empty section header string table.
    ///
    /// The table is initialized with a single null byte at offset 0,
    /// per the ELF specification requirement that offset 0 references
    /// an empty string.
    pub fn new() -> Self {
        Self {
            data: vec![0],
            offsets: HashMap::new(),
        }
    }

    /// Add a section name to the string table and return its byte offset.
    ///
    /// If the name has already been added, the existing offset is returned
    /// without adding a duplicate entry. Names are stored as null-terminated
    /// strings per the ELF string table format.
    ///
    /// # Arguments
    /// * `name` - Section name to add (e.g., ".text", ".data", ".shstrtab").
    ///
    /// # Returns
    /// Byte offset of the name within the string table, suitable for use
    /// in the `sh_name` field of an ELF section header.
    pub fn add(&mut self, name: &str) -> u32 {
        // Check for existing entry to avoid duplication
        if let Some(&offset) = self.offsets.get(name) {
            return offset;
        }

        // Record the current end of data as the new string's offset
        let offset = self.data.len() as u32;

        // Append the name bytes followed by a null terminator
        self.data.extend_from_slice(name.as_bytes());
        self.data.push(0); // null terminator

        // Cache the offset for deduplication
        self.offsets.insert(name.to_string(), offset);

        offset
    }

    /// Return the raw bytes of the string table.
    ///
    /// The returned slice includes the leading null byte and all added
    /// strings with their null terminators. This is the content that
    /// should be written to the `.shstrtab` section in the ELF output.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}

impl std::fmt::Debug for ShStrTab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShStrTab")
            .field("size", &self.data.len())
            .field("entries", &self.offsets.len())
            .finish()
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── align_up tests ──

    #[test]
    fn test_align_up_zero() {
        assert_eq!(align_up(0, 16), 0);
    }

    #[test]
    fn test_align_up_one() {
        assert_eq!(align_up(1, 16), 16);
    }

    #[test]
    fn test_align_up_exact() {
        assert_eq!(align_up(16, 16), 16);
    }

    #[test]
    fn test_align_up_one_over() {
        assert_eq!(align_up(17, 16), 32);
    }

    #[test]
    fn test_align_up_alignment_zero() {
        assert_eq!(align_up(42, 0), 42);
    }

    #[test]
    fn test_align_up_alignment_one() {
        assert_eq!(align_up(42, 1), 42);
    }

    #[test]
    fn test_align_up_large_page() {
        assert_eq!(align_up(0x400001, 0x1000), 0x401000);
    }

    #[test]
    fn test_align_up_already_page_aligned() {
        assert_eq!(align_up(0x401000, 0x1000), 0x401000);
    }

    #[test]
    fn test_align_up_power_of_two_alignments() {
        assert_eq!(align_up(3, 2), 4);
        assert_eq!(align_up(5, 4), 8);
        assert_eq!(align_up(9, 8), 16);
        assert_eq!(align_up(33, 32), 64);
        assert_eq!(align_up(100, 64), 128);
    }

    // ── padding_for_alignment tests ──

    #[test]
    fn test_padding_already_aligned() {
        assert_eq!(padding_for_alignment(16, 16), 0);
        assert_eq!(padding_for_alignment(0, 4096), 0);
        assert_eq!(padding_for_alignment(64, 8), 0);
    }

    #[test]
    fn test_padding_needs_alignment() {
        assert_eq!(padding_for_alignment(17, 16), 15);
        assert_eq!(padding_for_alignment(1, 16), 15);
        assert_eq!(padding_for_alignment(3, 4), 1);
        assert_eq!(padding_for_alignment(5, 8), 3);
    }

    #[test]
    fn test_padding_alignment_zero_or_one() {
        assert_eq!(padding_for_alignment(42, 0), 0);
        assert_eq!(padding_for_alignment(42, 1), 0);
    }

    // ── merge_sections tests ──

    fn make_input_section(
        name: &str,
        data: &[u8],
        section_type: u32,
        flags: u64,
        alignment: u64,
        object_index: usize,
        original_index: usize,
    ) -> InputSection {
        let mem_size = if section_type == SHT_NOBITS {
            data.len() as u64
        } else {
            data.len() as u64
        };
        InputSection {
            name: name.to_string(),
            data: data.to_vec(),
            section_type,
            flags,
            alignment,
            mem_size,
            object_index,
            original_index,
        }
    }

    fn make_bss_section(
        mem_size: u64,
        alignment: u64,
        object_index: usize,
        original_index: usize,
    ) -> InputSection {
        InputSection {
            name: ".bss".to_string(),
            data: Vec::new(),
            section_type: SHT_NOBITS,
            flags: SHF_ALLOC | SHF_WRITE,
            alignment,
            mem_size,
            object_index,
            original_index,
        }
    }

    #[test]
    fn test_merge_two_text_sections() {
        let inputs = vec![
            make_input_section(
                ".text",
                &[0x90, 0x90],
                SHT_PROGBITS,
                SHF_ALLOC | SHF_EXECINSTR,
                1,
                0,
                1,
            ),
            make_input_section(
                ".text",
                &[0xCC, 0xCC, 0xCC],
                SHT_PROGBITS,
                SHF_ALLOC | SHF_EXECINSTR,
                1,
                0,
                2,
            ),
        ];

        let merged = merge_sections(inputs, &[".text"]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].name, ".text");
        // With alignment 1, data is simply concatenated
        assert_eq!(merged[0].data, vec![0x90, 0x90, 0xCC, 0xCC, 0xCC]);
        assert_eq!(merged[0].data.len(), 5);
        assert_eq!(merged[0].mem_size, 5);
        assert_eq!(merged[0].input_mappings.len(), 2);
    }

    #[test]
    fn test_merge_sections_with_different_alignments() {
        let inputs = vec![
            make_input_section(
                ".text",
                &[0x90; 3],
                SHT_PROGBITS,
                SHF_ALLOC | SHF_EXECINSTR,
                4,
                0,
                1,
            ),
            make_input_section(
                ".text",
                &[0xCC; 2],
                SHT_PROGBITS,
                SHF_ALLOC | SHF_EXECINSTR,
                16,
                1,
                1,
            ),
        ];

        let merged = merge_sections(inputs, &[".text"]);
        assert_eq!(merged.len(), 1);
        // Max alignment should be 16
        assert_eq!(merged[0].alignment, 16);
        // First section: 3 bytes at offset 0
        assert_eq!(merged[0].input_mappings[0].output_offset, 0);
        assert_eq!(merged[0].input_mappings[0].size, 3);
        // Second section: aligned to 16 bytes, so offset 16
        assert_eq!(merged[0].input_mappings[1].output_offset, 16);
        assert_eq!(merged[0].input_mappings[1].size, 2);
        // Total data: 3 bytes + 13 padding + 2 bytes = 18 bytes
        assert_eq!(merged[0].data.len(), 18);
    }

    #[test]
    fn test_merge_input_section_mapping_offsets() {
        let inputs = vec![
            make_input_section(
                ".data",
                &[1, 2, 3, 4],
                SHT_PROGBITS,
                SHF_ALLOC | SHF_WRITE,
                4,
                0,
                2,
            ),
            make_input_section(
                ".data",
                &[5, 6],
                SHT_PROGBITS,
                SHF_ALLOC | SHF_WRITE,
                4,
                1,
                3,
            ),
        ];

        let merged = merge_sections(inputs, &[".data"]);
        assert_eq!(merged.len(), 1);

        // First mapping: object 0, section 2, starts at offset 0, size 4
        let m0 = &merged[0].input_mappings[0];
        assert_eq!(m0.object_index, 0);
        assert_eq!(m0.original_index, 2);
        assert_eq!(m0.output_offset, 0);
        assert_eq!(m0.size, 4);

        // Second mapping: object 1, section 3, starts at offset 4, size 2
        let m1 = &merged[0].input_mappings[1];
        assert_eq!(m1.object_index, 1);
        assert_eq!(m1.original_index, 3);
        assert_eq!(m1.output_offset, 4);
        assert_eq!(m1.size, 2);
    }

    #[test]
    fn test_merge_bss_sections() {
        let inputs = vec![
            make_bss_section(100, 8, 0, 5),
            make_bss_section(200, 16, 1, 3),
        ];

        let merged = merge_sections(inputs, &[".bss"]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].name, ".bss");
        assert_eq!(merged[0].section_type, SHT_NOBITS);
        // BSS data should be empty
        assert!(merged[0].data.is_empty());
        // mem_size should account for alignment between sections and sum of sizes
        assert!(merged[0].mem_size >= 300); // at least 100 + 200
                                            // Max alignment
        assert_eq!(merged[0].alignment, 16);
    }

    #[test]
    fn test_merge_preserves_flags_union() {
        let inputs = vec![
            make_input_section(".data", &[1], SHT_PROGBITS, SHF_ALLOC, 1, 0, 1),
            make_input_section(".data", &[2], SHT_PROGBITS, SHF_ALLOC | SHF_WRITE, 1, 1, 1),
        ];

        let merged = merge_sections(inputs, &[".data"]);
        // Flags should be the union: ALLOC | WRITE
        assert_eq!(merged[0].flags, SHF_ALLOC | SHF_WRITE);
    }

    #[test]
    fn test_merge_multiple_section_names() {
        let inputs = vec![
            make_input_section(
                ".text",
                &[0x90],
                SHT_PROGBITS,
                SHF_ALLOC | SHF_EXECINSTR,
                1,
                0,
                1,
            ),
            make_input_section(".data", &[1], SHT_PROGBITS, SHF_ALLOC | SHF_WRITE, 1, 0, 2),
            make_input_section(
                ".text",
                &[0xCC],
                SHT_PROGBITS,
                SHF_ALLOC | SHF_EXECINSTR,
                1,
                1,
                1,
            ),
            make_input_section(".data", &[2], SHT_PROGBITS, SHF_ALLOC | SHF_WRITE, 1, 1, 2),
        ];

        let order = &[".text", ".data"];
        let merged = merge_sections(inputs, order);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].name, ".text");
        assert_eq!(merged[1].name, ".data");
        assert_eq!(merged[0].data, vec![0x90, 0xCC]);
        assert_eq!(merged[1].data, vec![1, 2]);
    }

    #[test]
    fn test_merge_section_order_respected() {
        let inputs = vec![
            make_input_section(".data", &[1], SHT_PROGBITS, SHF_ALLOC | SHF_WRITE, 1, 0, 1),
            make_input_section(
                ".text",
                &[0x90],
                SHT_PROGBITS,
                SHF_ALLOC | SHF_EXECINSTR,
                1,
                0,
                2,
            ),
            make_input_section(".rodata", &[42], SHT_PROGBITS, SHF_ALLOC, 1, 0, 3),
        ];

        // Even though .data appeared first in input, section_order dictates output
        let order = &[".text", ".rodata", ".data"];
        let merged = merge_sections(inputs, order);
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].name, ".text");
        assert_eq!(merged[1].name, ".rodata");
        assert_eq!(merged[2].name, ".data");
    }

    #[test]
    fn test_merge_unknown_sections_sorted_alphabetically() {
        let inputs = vec![
            make_input_section(".zebra", &[1], SHT_PROGBITS, SHF_ALLOC, 1, 0, 1),
            make_input_section(
                ".text",
                &[2],
                SHT_PROGBITS,
                SHF_ALLOC | SHF_EXECINSTR,
                1,
                0,
                2,
            ),
            make_input_section(".apple", &[3], SHT_PROGBITS, SHF_ALLOC, 1, 0, 3),
        ];

        let order = &[".text"];
        let merged = merge_sections(inputs, order);
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].name, ".text"); // from section_order
        assert_eq!(merged[1].name, ".apple"); // alphabetical
        assert_eq!(merged[2].name, ".zebra"); // alphabetical
    }

    // ── section permission classification tests ──

    #[test]
    fn test_classify_text_section() {
        let perm = classify_section_permission(".text", SHF_ALLOC | SHF_EXECINSTR);
        assert_eq!(perm, SectionPermission::ReadExecute);
    }

    #[test]
    fn test_classify_rodata_section() {
        let perm = classify_section_permission(".rodata", SHF_ALLOC);
        assert_eq!(perm, SectionPermission::ReadOnly);
    }

    #[test]
    fn test_classify_data_section() {
        let perm = classify_section_permission(".data", SHF_ALLOC | SHF_WRITE);
        assert_eq!(perm, SectionPermission::ReadWrite);
    }

    #[test]
    fn test_classify_bss_section() {
        let perm = classify_section_permission(".bss", SHF_ALLOC | SHF_WRITE);
        assert_eq!(perm, SectionPermission::ReadWrite);
    }

    #[test]
    fn test_classify_debug_section() {
        let perm = classify_section_permission(".debug_info", 0);
        assert_eq!(perm, SectionPermission::None);
    }

    #[test]
    fn test_classify_symtab_section() {
        let perm = classify_section_permission(".symtab", 0);
        assert_eq!(perm, SectionPermission::None);
    }

    // ── ShStrTab tests ──

    #[test]
    fn test_shstrtab_new() {
        let tab = ShStrTab::new();
        assert_eq!(tab.as_bytes(), &[0]);
    }

    #[test]
    fn test_shstrtab_add_single() {
        let mut tab = ShStrTab::new();
        let offset = tab.add(".text");
        assert_eq!(offset, 1); // after the leading null byte
                               // Data should be: \0 . t e x t \0
        assert_eq!(tab.as_bytes(), &[0, b'.', b't', b'e', b'x', b't', 0]);
    }

    #[test]
    fn test_shstrtab_add_multiple() {
        let mut tab = ShStrTab::new();
        let off1 = tab.add(".text");
        let off2 = tab.add(".data");
        assert_eq!(off1, 1);
        assert_eq!(off2, 7); // 1 (null) + 5 (.text) + 1 (null) = 7
    }

    #[test]
    fn test_shstrtab_deduplication() {
        let mut tab = ShStrTab::new();
        let off1 = tab.add(".text");
        let off2 = tab.add(".text");
        assert_eq!(off1, off2);
        // Should only contain one copy
        assert_eq!(tab.as_bytes().len(), 7); // \0 .text \0
    }

    #[test]
    fn test_shstrtab_empty_string() {
        let mut tab = ShStrTab::new();
        let offset = tab.add("");
        // Empty string added at current position
        assert_eq!(offset, 1);
        // Data: \0 \0 (leading null + empty string's null terminator)
        assert_eq!(tab.as_bytes(), &[0, 0]);
    }

    #[test]
    fn test_shstrtab_multiple_dedup() {
        let mut tab = ShStrTab::new();
        let o1 = tab.add(".text");
        let o2 = tab.add(".data");
        let o3 = tab.add(".text"); // duplicate
        let o4 = tab.add(".bss");
        let o5 = tab.add(".data"); // duplicate

        assert_eq!(o1, o3); // .text deduplicated
        assert_eq!(o2, o5); // .data deduplicated
        assert_ne!(o1, o2);
        assert_ne!(o2, o4);
    }

    // ── layout computation tests ──

    #[test]
    fn test_compute_layout_basic() {
        let mut sections = vec![MergedSection {
            name: ".text".to_string(),
            data: vec![0x90; 100],
            section_type: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_EXECINSTR,
            alignment: 16,
            mem_size: 100,
            virtual_address: 0,
            file_offset: 0,
            input_mappings: vec![],
        }];

        let base = 0x400000;
        let header_size = 64 + 56; // ELF64 header + one phdr
        let result = compute_layout(&mut sections, base, header_size, true);

        // .text aligned to 16 bytes after header
        let expected_offset = align_up(header_size, 16);
        assert_eq!(sections[0].file_offset, expected_offset);
        assert_eq!(sections[0].virtual_address, base + expected_offset);
        assert_eq!(result.total_file_size, expected_offset + 100);
    }

    #[test]
    fn test_compute_layout_bss_no_file_size() {
        let mut sections = vec![
            MergedSection {
                name: ".data".to_string(),
                data: vec![1, 2, 3, 4],
                section_type: SHT_PROGBITS,
                flags: SHF_ALLOC | SHF_WRITE,
                alignment: 4,
                mem_size: 4,
                virtual_address: 0,
                file_offset: 0,
                input_mappings: vec![],
            },
            MergedSection {
                name: ".bss".to_string(),
                data: vec![],
                section_type: SHT_NOBITS,
                flags: SHF_ALLOC | SHF_WRITE,
                alignment: 8,
                mem_size: 256,
                virtual_address: 0,
                file_offset: 0,
                input_mappings: vec![],
            },
        ];

        let base = 0x400000;
        let header_size = 120;
        let result = compute_layout(&mut sections, base, header_size, true);

        // BSS should not increase file size: total_file_size should equal
        // the file offset of the BSS section (since BSS contributes no bytes).
        let bss_file_offset = sections[1].file_offset;
        assert_eq!(result.total_file_size, bss_file_offset);
        // But total memory size includes BSS, so mem_size > file_size
        // (mem_size is relative to base, file_size is absolute file offset)
        // Verify that BSS's contribution to memory exceeds its zero file contribution
        let file_portion = result.total_file_size;
        let mem_portion = result.total_mem_size;
        assert!(
            mem_portion > file_portion,
            "BSS should make mem_size > file_size"
        );

        // Verify BSS virtual address follows data section
        assert!(sections[1].virtual_address >= sections[0].virtual_address + 4);
    }

    #[test]
    fn test_compute_layout_segment_boundary() {
        let mut sections = vec![
            MergedSection {
                name: ".text".to_string(),
                data: vec![0x90; 100],
                section_type: SHT_PROGBITS,
                flags: SHF_ALLOC | SHF_EXECINSTR,
                alignment: 1,
                mem_size: 100,
                virtual_address: 0,
                file_offset: 0,
                input_mappings: vec![],
            },
            MergedSection {
                name: ".rodata".to_string(),
                data: vec![42; 50],
                section_type: SHT_PROGBITS,
                flags: SHF_ALLOC,
                alignment: 1,
                mem_size: 50,
                virtual_address: 0,
                file_offset: 0,
                input_mappings: vec![],
            },
        ];

        let base = 0x400000;
        let header_size = 120;
        compute_layout(&mut sections, base, header_size, true);

        // .text is ReadExecute, .rodata is ReadOnly => segment boundary
        // .rodata's file_offset should be page-aligned
        assert_eq!(sections[1].file_offset % 0x1000, 0);
        assert_eq!(sections[1].virtual_address % 0x1000, 0);
    }

    #[test]
    fn test_compute_layout_same_permission_no_page_boundary() {
        let mut sections = vec![
            MergedSection {
                name: ".text".to_string(),
                data: vec![0x90; 100],
                section_type: SHT_PROGBITS,
                flags: SHF_ALLOC | SHF_EXECINSTR,
                alignment: 1,
                mem_size: 100,
                virtual_address: 0,
                file_offset: 0,
                input_mappings: vec![],
            },
            MergedSection {
                name: ".init".to_string(),
                data: vec![0xCC; 50],
                section_type: SHT_PROGBITS,
                flags: SHF_ALLOC | SHF_EXECINSTR,
                alignment: 1,
                mem_size: 50,
                virtual_address: 0,
                file_offset: 0,
                input_mappings: vec![],
            },
        ];

        let base = 0x400000;
        let header_size = 120;
        compute_layout(&mut sections, base, header_size, true);

        // Both are ReadExecute, so no page boundary between them
        // .init should immediately follow .text (no page alignment needed)
        assert_eq!(sections[1].file_offset, sections[0].file_offset + 100);
    }

    // ── offset translation tests ──

    #[test]
    fn test_translate_offset_found() {
        let merged = MergedSection {
            name: ".text".to_string(),
            data: vec![],
            section_type: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_EXECINSTR,
            alignment: 1,
            mem_size: 0,
            virtual_address: 0,
            file_offset: 0,
            input_mappings: vec![
                InputSectionMapping {
                    object_index: 0,
                    original_index: 1,
                    output_offset: 0,
                    size: 100,
                },
                InputSectionMapping {
                    object_index: 1,
                    original_index: 2,
                    output_offset: 100,
                    size: 50,
                },
            ],
        };

        // Translate offset 10 from object 0, section 1
        let result = translate_offset(&merged, 0, 1, 10);
        assert_eq!(result, Some(10));

        // Translate offset 20 from object 1, section 2
        let result = translate_offset(&merged, 1, 2, 20);
        assert_eq!(result, Some(120)); // 100 + 20
    }

    #[test]
    fn test_translate_offset_not_found() {
        let merged = MergedSection {
            name: ".text".to_string(),
            data: vec![],
            section_type: SHT_PROGBITS,
            flags: 0,
            alignment: 1,
            mem_size: 0,
            virtual_address: 0,
            file_offset: 0,
            input_mappings: vec![InputSectionMapping {
                object_index: 0,
                original_index: 1,
                output_offset: 0,
                size: 100,
            }],
        };

        // Non-existent object/section combination
        let result = translate_offset(&merged, 2, 5, 0);
        assert_eq!(result, Option::None);
    }

    #[test]
    fn test_translate_offset_zero() {
        let merged = MergedSection {
            name: ".data".to_string(),
            data: vec![],
            section_type: SHT_PROGBITS,
            flags: 0,
            alignment: 1,
            mem_size: 0,
            virtual_address: 0,
            file_offset: 0,
            input_mappings: vec![InputSectionMapping {
                object_index: 3,
                original_index: 7,
                output_offset: 500,
                size: 200,
            }],
        };

        let result = translate_offset(&merged, 3, 7, 0);
        assert_eq!(result, Some(500));
    }

    // ── ELF constant value tests ──

    #[test]
    fn test_section_type_constants() {
        assert_eq!(SHT_NULL, 0);
        assert_eq!(SHT_PROGBITS, 1);
        assert_eq!(SHT_SYMTAB, 2);
        assert_eq!(SHT_STRTAB, 3);
        assert_eq!(SHT_RELA, 4);
        assert_eq!(SHT_HASH, 5);
        assert_eq!(SHT_DYNAMIC, 6);
        assert_eq!(SHT_NOTE, 7);
        assert_eq!(SHT_NOBITS, 8);
        assert_eq!(SHT_REL, 9);
        assert_eq!(SHT_DYNSYM, 11);
    }

    #[test]
    fn test_section_flag_constants() {
        assert_eq!(SHF_WRITE, 0x1);
        assert_eq!(SHF_ALLOC, 0x2);
        assert_eq!(SHF_EXECINSTR, 0x4);
        assert_eq!(SHF_MERGE, 0x10);
        assert_eq!(SHF_STRINGS, 0x20);
    }

    // ── Edge case and integration tests ──

    #[test]
    fn test_merge_empty_input() {
        let inputs: Vec<InputSection> = vec![];
        let merged = merge_sections(inputs, &[".text"]);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_merge_single_section() {
        let inputs = vec![make_input_section(
            ".text",
            &[0x90, 0x91, 0x92],
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR,
            4,
            0,
            1,
        )];

        let merged = merge_sections(inputs, &[".text"]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].data, vec![0x90, 0x91, 0x92]);
        assert_eq!(merged[0].alignment, 4);
        assert_eq!(merged[0].input_mappings.len(), 1);
        assert_eq!(merged[0].input_mappings[0].output_offset, 0);
        assert_eq!(merged[0].input_mappings[0].size, 3);
    }

    #[test]
    fn test_layout_with_non_allocatable_sections() {
        let mut sections = vec![
            MergedSection {
                name: ".text".to_string(),
                data: vec![0x90; 100],
                section_type: SHT_PROGBITS,
                flags: SHF_ALLOC | SHF_EXECINSTR,
                alignment: 1,
                mem_size: 100,
                virtual_address: 0,
                file_offset: 0,
                input_mappings: vec![],
            },
            MergedSection {
                name: ".debug_info".to_string(),
                data: vec![0; 200],
                section_type: SHT_PROGBITS,
                flags: 0, // Not allocatable
                alignment: 1,
                mem_size: 200,
                virtual_address: 0,
                file_offset: 0,
                input_mappings: vec![],
            },
        ];

        let base = 0x400000;
        let header_size = 120;
        compute_layout(&mut sections, base, header_size, true);

        // Debug section should still get a file offset for storage
        assert!(sections[1].file_offset > 0);
        // But its permission is None, so no page alignment boundary
        // (non-allocatable to non-allocatable has no segment transition concern)
    }

    #[test]
    fn test_section_permission_hash() {
        // Verify SectionPermission can be used as HashMap key (has Hash trait)
        let mut map: HashMap<SectionPermission, u32> = HashMap::new();
        map.insert(SectionPermission::ReadExecute, 1);
        map.insert(SectionPermission::ReadOnly, 2);
        map.insert(SectionPermission::ReadWrite, 3);
        map.insert(SectionPermission::None, 4);
        assert_eq!(map.len(), 4);
    }
}
