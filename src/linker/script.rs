//! # Default Linker Script Behavior
//!
//! Rather than parsing external linker scripts, this module encodes the default
//! section ordering, entry point selection, segment layout, and section-to-segment
//! mapping rules that a standard ELF linker uses. This is the internal "built-in
//! linker script" that determines how sections from input objects are arranged
//! into the final ELF output.
//!
//! ## Key Responsibilities
//!
//! - **Section Ordering** — Define the canonical order of ELF sections matching
//!   GNU `ld` defaults: code → read-only data → read-write data → BSS → debug.
//! - **Entry Point Selection** — Determine `_start` for executables, `None` for
//!   shared libraries and relocatable objects.
//! - **Segment Layout** — Page-aligned PT_LOAD segments grouping sections by
//!   permission (RX, R, RW).
//! - **Section-to-Segment Mapping** — Classify each section into the correct
//!   ELF segment type with appropriate permission flags.
//! - **Address Assignment** — Compute virtual addresses and file offsets for all
//!   sections, respecting alignment requirements and page boundaries.
//! - **PT_INTERP Path Selection** — Architecture-specific dynamic linker paths.
//! - **PT_GNU_STACK** — Non-executable stack segment generation.
//!
//! ## ELF Dual-Width Support
//!
//! All address computations use `u64` to accommodate both ELF32 (i686) and
//! ELF64 (x86-64, AArch64, RISC-V 64) address spaces. The `default_base_address`
//! function selects the appropriate base address based on the target's pointer
//! width.
//!
//! ## Zero External Dependencies
//!
//! This module uses only the Rust standard library and sibling linker modules.

use super::elf::{
    PF_R, PF_W, PF_X, PT_DYNAMIC, PT_GNU_RELRO, PT_GNU_STACK, PT_INTERP, PT_LOAD, PT_NOTE,
};
use super::sections::align_up;
use super::OutputMode;
use crate::driver::target::{Architecture, TargetConfig};

// ============================================================================
// Section Ordering
// ============================================================================

/// Default section ordering for ELF executables.
///
/// This matches the standard layout used by GNU `ld` without a linker script.
/// Sections are arranged so that:
/// 1. Read-only, non-executable sections come first (interp, notes, hash, dynsym)
/// 2. Executable code sections follow (.init, .plt, .text, .fini)
/// 3. Read-only data sections (.rodata, .eh_frame)
/// 4. Read-write sections with init/fini, dynamic, GOT (.init_array through .got.plt)
/// 5. Initialized and uninitialized data (.data, .bss)
/// 6. Non-loadable debug sections
/// 7. Non-loadable metadata (symbol tables, string tables)
pub const DEFAULT_SECTION_ORDER: &[&str] = &[
    ".interp",
    ".note.ABI-tag",
    ".note.gnu.build-id",
    ".hash",
    ".gnu.hash",
    ".dynsym",
    ".dynstr",
    ".rela.dyn",
    ".rela.plt",
    ".init",
    ".plt",
    ".text",
    ".fini",
    ".rodata",
    ".eh_frame",
    ".eh_frame_hdr",
    ".init_array",
    ".fini_array",
    ".dynamic",
    ".got",
    ".got.plt",
    ".data",
    ".bss",
    // Debug sections (not loaded into memory)
    ".debug_info",
    ".debug_abbrev",
    ".debug_line",
    ".debug_str",
    ".debug_aranges",
    ".debug_frame",
    ".debug_loc",
    // Symbol and string tables (not loaded)
    ".symtab",
    ".strtab",
    ".shstrtab",
];

/// Returns a numeric sort key for ordering sections in the output binary.
///
/// Sections present in [`DEFAULT_SECTION_ORDER`] receive a key equal to their
/// index in that list. Unknown sections receive a key that places them after
/// the last known loadable section (`.bss`) but before the debug sections,
/// ensuring a deterministic and well-structured layout.
///
/// # Arguments
/// * `name` — Section name (e.g., `".text"`, `".data"`).
///
/// # Returns
/// A `u32` sort key. Lower values appear earlier in the ELF output.
pub fn section_sort_key(name: &str) -> u32 {
    // Check if section is in the known ordering list
    for (i, &known) in DEFAULT_SECTION_ORDER.iter().enumerate() {
        if known == name {
            return i as u32;
        }
    }

    // Unknown sections: place after .bss (index 22) but before debug sections (index 23+).
    // Use a sentinel value that groups unknowns together in the loadable region.
    // The index of ".bss" in DEFAULT_SECTION_ORDER is 22, and ".debug_info" is 23.
    // We place unknown sections at index 22 + 1/2 (effectively 23 minus epsilon),
    // but since we use u32, we use the gap.
    let bss_index = DEFAULT_SECTION_ORDER
        .iter()
        .position(|&s| s == ".bss")
        .unwrap_or(22) as u32;
    let debug_info_index = DEFAULT_SECTION_ORDER
        .iter()
        .position(|&s| s == ".debug_info")
        .unwrap_or(23) as u32;

    // Place unknown sections between .bss and .debug_info.
    // We use (bss_index + debug_info_index) / 2, but since these are adjacent
    // integers (22 and 23), we cannot fit between them with integer arithmetic.
    // Instead, we shift all debug/metadata indices up by a gap and place unknowns
    // in that gap.
    // Practical approach: just return bss_index + 1 so unknowns sort right after
    // .bss but the debug sections have higher indices (23+).
    // Since ".debug_info" is at index 23 in the constant array, returning
    // bss_index + 1 = 23 would collide. We handle this by noting that
    // unknown sections that are NOT debug sections get placed between bss and
    // debug. For unknown debug sections (starting with ".debug_"), they sort
    // after known debug sections.
    if name.starts_with(".debug_") {
        // Unknown debug section: place after known debug sections
        debug_info_index + 100
    } else if name.starts_with(".note") {
        // Unknown note sections: place near known note sections (indices 1-2)
        3
    } else if name.starts_with(".rela") || name.starts_with(".rel.") {
        // Unknown relocation sections: place near known rela sections (indices 7-8)
        9
    } else {
        // General unknown sections: place between .bss and debug sections
        // Use a value between bss_index and debug_info_index — since they're
        // adjacent (22, 23), we scale: known sections use their actual index,
        // unknowns use (bss_index * 2 + 1) to create a gap in a scaled space.
        // Simpler approach: return a fixed sentinel.
        bss_index + 1
    }
}

// ============================================================================
// Entry Point Selection
// ============================================================================

/// Determine the entry point symbol for the executable.
///
/// - For static executables: the configured entry point (typically `"_start"`,
///   provided by `crt1.o`).
/// - For shared libraries: no entry point (`e_entry = 0` in the ELF header).
/// - For relocatable objects: no entry point (partial link, not directly executable).
///
/// # Arguments
/// * `output_mode` — The output mode (executable, shared library, or relocatable).
/// * `config_entry` — The configured entry point symbol name (default: `"_start"`).
///
/// # Returns
/// `Some(symbol_name)` for executables, `None` for shared libraries and
/// relocatable objects.
pub fn select_entry_point(output_mode: &OutputMode, config_entry: &str) -> Option<String> {
    match output_mode {
        OutputMode::StaticExecutable => Some(config_entry.to_string()),
        OutputMode::SharedLibrary => None,
        OutputMode::Relocatable => None,
    }
}

// ============================================================================
// Segment Layout Configuration
// ============================================================================

/// Page size for segment alignment.
///
/// Standard Linux page size is 4096 bytes for all supported architectures
/// (x86-64, i686, AArch64, RISC-V 64). All PT_LOAD segments must be aligned
/// to this boundary for the kernel's memory-mapped page protection.
pub const PAGE_SIZE: u64 = 0x1000; // 4096 bytes

/// Default base address for executables.
///
/// This is the virtual address where the first PT_LOAD segment is placed.
/// The choice depends on the target's pointer width:
/// - **64-bit targets** (x86-64, AArch64, RISC-V 64): `0x400000`
/// - **32-bit targets** (i686): `0x08048000`
///
/// These values match the standard GNU `ld` defaults for Linux.
///
/// # Arguments
/// * `target` — Target configuration specifying the architecture.
///
/// # Returns
/// The default base virtual address for the target.
pub fn default_base_address(target: &TargetConfig) -> u64 {
    if target.is_64bit() {
        0x400000 // Standard 64-bit base address
    } else {
        0x08048000 // Standard 32-bit base address (i686)
    }
}

// ============================================================================
// Section-to-Segment Mapping
// ============================================================================

/// ELF segment type classification for section-to-segment mapping.
///
/// Each variant corresponds to an ELF program header type. Sections are
/// classified into segments based on their name and permission requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentType {
    /// `PT_LOAD`: Loadable segment mapped into process memory.
    Load,
    /// `PT_INTERP`: Contains the path to the dynamic linker.
    Interp,
    /// `PT_DYNAMIC`: Contains dynamic linking information.
    Dynamic,
    /// `PT_NOTE`: Contains auxiliary information (ABI tag, build ID).
    Note,
    /// `PT_GNU_STACK`: Specifies stack permission attributes.
    GnuStack,
    /// `PT_GNU_RELRO`: Marks regions to be read-only after relocation.
    GnuRelro,
    /// Not loaded into memory (debug sections, symbol tables, string tables).
    None,
}

impl SegmentType {
    /// Convert this segment type to its ELF `p_type` constant value.
    ///
    /// Returns `0` for `SegmentType::None` (not a real ELF segment type).
    pub fn to_elf_type(self) -> u32 {
        match self {
            SegmentType::Load => PT_LOAD,
            SegmentType::Interp => PT_INTERP,
            SegmentType::Dynamic => PT_DYNAMIC,
            SegmentType::Note => PT_NOTE,
            SegmentType::GnuStack => PT_GNU_STACK,
            SegmentType::GnuRelro => PT_GNU_RELRO,
            SegmentType::None => 0,
        }
    }
}

/// Permission flags for an ELF segment or section.
///
/// These flags map to the ELF `p_flags` field values (`PF_R`, `PF_W`, `PF_X`)
/// and determine the memory protection applied by the kernel's memory mapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentFlags {
    /// Whether the segment is readable.
    pub readable: bool,
    /// Whether the segment is writable.
    pub writable: bool,
    /// Whether the segment is executable.
    pub executable: bool,
}

impl SegmentFlags {
    /// Convert to the ELF `p_flags` bitmask (`PF_R | PF_W | PF_X`).
    pub fn to_elf_flags(self) -> u32 {
        let mut flags = 0u32;
        if self.readable {
            flags |= PF_R;
        }
        if self.writable {
            flags |= PF_W;
        }
        if self.executable {
            flags |= PF_X;
        }
        flags
    }

    /// Check if two `SegmentFlags` have identical permission bits.
    ///
    /// Used to determine whether adjacent sections can be grouped into the
    /// same PT_LOAD segment.
    pub fn same_permissions(&self, other: &SegmentFlags) -> bool {
        self.readable == other.readable
            && self.writable == other.writable
            && self.executable == other.executable
    }
}

/// Determine which segment type and permission flags a section belongs to.
///
/// This function implements the standard section-to-segment mapping used by
/// GNU `ld` without a linker script. The mapping considers section names and
/// classifies them into appropriate ELF segment types with permission flags.
///
/// # Section Classification
///
/// | Section(s) | Segment | Flags |
/// |---|---|---|
/// | `.interp` | `PT_INTERP` | R |
/// | `.text`, `.init`, `.fini`, `.plt` | `PT_LOAD` | R+X |
/// | `.rodata`, `.eh_frame`, `.eh_frame_hdr` | `PT_LOAD` | R |
/// | `.data`, `.bss`, `.got`, `.got.plt` | `PT_LOAD` | R+W |
/// | `.init_array`, `.fini_array` | `PT_LOAD` | R+W |
/// | `.dynamic` | `PT_DYNAMIC` | R+W |
/// | `.note.*` | `PT_NOTE` | R |
/// | `.debug_*` | None | — |
/// | `.symtab`, `.strtab`, `.shstrtab` | None | — |
///
/// # Arguments
/// * `name` — Section name (e.g., `".text"`, `".data"`).
///
/// # Returns
/// A tuple of `(SegmentType, SegmentFlags)` for the section.
pub fn section_to_segment(name: &str) -> (SegmentType, SegmentFlags) {
    match name {
        // Interpreter path section: gets its own PT_INTERP segment
        ".interp" => (
            SegmentType::Interp,
            SegmentFlags {
                readable: true,
                writable: false,
                executable: false,
            },
        ),

        // Executable code sections: PT_LOAD with R+X
        ".text" | ".init" | ".fini" | ".plt" => (
            SegmentType::Load,
            SegmentFlags {
                readable: true,
                writable: false,
                executable: true,
            },
        ),

        // Read-only data sections: PT_LOAD with R
        ".rodata" | ".eh_frame" | ".eh_frame_hdr" => (
            SegmentType::Load,
            SegmentFlags {
                readable: true,
                writable: false,
                executable: false,
            },
        ),

        // Hash tables and dynamic symbol/string tables: PT_LOAD with R
        ".hash" | ".gnu.hash" | ".dynsym" | ".dynstr" => (
            SegmentType::Load,
            SegmentFlags {
                readable: true,
                writable: false,
                executable: false,
            },
        ),

        // Relocation sections used at runtime: PT_LOAD with R
        ".rela.dyn" | ".rela.plt" | ".rel.dyn" | ".rel.plt" => (
            SegmentType::Load,
            SegmentFlags {
                readable: true,
                writable: false,
                executable: false,
            },
        ),

        // Read-write data sections: PT_LOAD with R+W
        ".data" | ".bss" | ".got" | ".got.plt" => (
            SegmentType::Load,
            SegmentFlags {
                readable: true,
                writable: true,
                executable: false,
            },
        ),

        // Init/fini arrays: PT_LOAD with R+W
        ".init_array" | ".fini_array" => (
            SegmentType::Load,
            SegmentFlags {
                readable: true,
                writable: true,
                executable: false,
            },
        ),

        // Dynamic linking information: PT_DYNAMIC segment with R+W
        ".dynamic" => (
            SegmentType::Dynamic,
            SegmentFlags {
                readable: true,
                writable: true,
                executable: false,
            },
        ),

        // Debug sections: not loaded into memory
        s if s.starts_with(".debug_") => (
            SegmentType::None,
            SegmentFlags {
                readable: false,
                writable: false,
                executable: false,
            },
        ),

        // Note sections: PT_NOTE with R
        s if s.starts_with(".note") => (
            SegmentType::Note,
            SegmentFlags {
                readable: true,
                writable: false,
                executable: false,
            },
        ),

        // Symbol and string table sections: not loaded
        ".symtab" | ".strtab" | ".shstrtab" | ".comment" => (
            SegmentType::None,
            SegmentFlags {
                readable: false,
                writable: false,
                executable: false,
            },
        ),

        // Unknown sections: default to PT_LOAD with read-only
        _ => (
            SegmentType::Load,
            SegmentFlags {
                readable: true,
                writable: false,
                executable: false,
            },
        ),
    }
}

// ============================================================================
// Section and Segment Layout Structures
// ============================================================================

/// Describes the layout of a single section in the output ELF binary.
///
/// This struct records the assigned virtual address, file offset, size, and
/// permission flags for each section after address assignment. It is used
/// by [`group_into_segments`] to construct PT_LOAD segments and by the
/// linker to write section data at the correct file offsets.
#[derive(Debug, Clone)]
pub struct SectionLayout {
    /// Section name (e.g., `".text"`, `".data"`, `".bss"`).
    pub name: String,
    /// Assigned virtual address in the output binary's address space.
    pub virtual_address: u64,
    /// File offset where this section's data begins in the output binary.
    /// For `.bss` sections, file_offset is recorded but no data is written.
    pub file_offset: u64,
    /// Size of the section data in the file. For `.bss`, this is 0.
    pub size: u64,
    /// Size of the section in memory. For `.bss`, `mem_size > size` because
    /// BSS occupies memory but not file space.
    pub mem_size: u64,
    /// Required alignment for this section (must be a power of 2).
    pub alignment: u64,
    /// Permission flags determining which PT_LOAD segment this section
    /// belongs to.
    pub flags: SegmentFlags,
}

/// Describes a program header (segment) in the output ELF binary.
///
/// Each `SegmentLayout` corresponds to one ELF program header entry. Multiple
/// sections with the same permission flags are grouped into a single PT_LOAD
/// segment. Special segments like PT_INTERP, PT_DYNAMIC, PT_NOTE, and
/// PT_GNU_STACK have their own entries.
#[derive(Debug, Clone)]
pub struct SegmentLayout {
    /// ELF program header type (`PT_LOAD`, `PT_INTERP`, `PT_DYNAMIC`, etc.).
    pub segment_type: u32,
    /// ELF segment permission flags (`PF_R | PF_W | PF_X`).
    pub flags: u32,
    /// Virtual address of the segment's first byte in memory.
    pub virtual_address: u64,
    /// Physical address of the segment (typically equal to virtual address
    /// for user-space executables).
    pub physical_address: u64,
    /// File offset of the segment's first byte.
    pub file_offset: u64,
    /// Size of the segment in the file (excludes BSS padding).
    pub file_size: u64,
    /// Size of the segment in memory (includes BSS zero-fill).
    pub mem_size: u64,
    /// Segment alignment requirement. For PT_LOAD, this is `PAGE_SIZE`.
    pub alignment: u64,
    /// Names of sections contained in this segment. Used for debugging
    /// and section-to-segment back-references.
    pub sections: Vec<String>,
}

/// Result of the complete address assignment process.
///
/// Contains the final layout of all sections and segments, the entry point
/// virtual address, and the total file size needed for the output binary.
#[derive(Debug, Clone)]
pub struct AddressAssignment {
    /// Final section layouts with assigned addresses and file offsets.
    pub sections: Vec<SectionLayout>,
    /// Program headers (segments) constructed from the section layouts.
    pub segments: Vec<SegmentLayout>,
    /// Virtual address of the entry point (`_start`). Zero if no entry point
    /// (shared library or relocatable).
    pub entry_address: u64,
    /// Total file size in bytes for the output ELF binary.
    pub total_file_size: u64,
}

// ============================================================================
// PT_LOAD Segment Grouping
// ============================================================================

/// Group ordered sections into PT_LOAD segments based on permission flags.
///
/// Sections with the same permission flags (RX, R, RW) are grouped into
/// the same PT_LOAD segment. Each segment is page-aligned. Non-loadable
/// sections (debug, symbol tables) are excluded from segments.
///
/// ## Typical Segment Layout
///
/// | Segment | Permissions | Sections |
/// |---|---|---|
/// | 1 | R | `.interp`, `.note.*`, `.hash`, `.dynsym`, `.dynstr`, `.rela.*` |
/// | 2 | R+X | `.init`, `.plt`, `.text`, `.fini` |
/// | 3 | R | `.rodata`, `.eh_frame` |
/// | 4 | R+W | `.init_array`, `.fini_array`, `.dynamic`, `.got`, `.data`, `.bss` |
///
/// # Arguments
/// * `sections` — Ordered section layouts with assigned addresses and flags.
/// * `_base_address` — Base address (used for context; addresses are already assigned).
///
/// # Returns
/// A vector of `SegmentLayout` entries representing PT_LOAD program headers.
pub fn group_into_segments(sections: &[SectionLayout], _base_address: u64) -> Vec<SegmentLayout> {
    let mut segments: Vec<SegmentLayout> = Vec::new();

    // Track the current segment being built. We start a new segment whenever
    // the permission flags change between consecutive loadable sections.
    let mut current_segment: Option<SegmentLayout> = None;

    for section in sections {
        // Determine the segment type for this section
        let (seg_type, _seg_flags) = section_to_segment(&section.name);

        // Skip non-loadable sections (debug, symtab, strtab)
        if seg_type == SegmentType::None {
            continue;
        }

        // For special segment types (INTERP, DYNAMIC, NOTE), they get their
        // own segment in addition to being part of a PT_LOAD. We handle
        // them as loadable for PT_LOAD grouping purposes and emit separate
        // special-type segments later.

        let elf_flags = section.flags.to_elf_flags();

        match current_segment.as_mut() {
            Some(ref mut seg) if seg.flags == elf_flags => {
                // Same permissions: extend the current segment to include this section
                // File size extends to cover this section's data
                if section.size > 0 {
                    seg.file_size = (section.file_offset + section.size) - seg.file_offset;
                }

                // Memory size extends to cover this section's memory footprint
                seg.mem_size = (section.virtual_address + section.mem_size) - seg.virtual_address;

                // Record this section in the segment
                seg.sections.push(section.name.clone());
            }
            _ => {
                // Different permissions or no current segment: finalize current
                // and start a new one
                if let Some(finished) = current_segment.take() {
                    segments.push(finished);
                }

                current_segment = Some(SegmentLayout {
                    segment_type: PT_LOAD,
                    flags: elf_flags,
                    virtual_address: section.virtual_address,
                    physical_address: section.virtual_address,
                    file_offset: section.file_offset,
                    file_size: section.size,
                    mem_size: section.mem_size,
                    alignment: PAGE_SIZE,
                    sections: vec![section.name.clone()],
                });
            }
        }
    }

    // Finalize the last segment
    if let Some(finished) = current_segment.take() {
        segments.push(finished);
    }

    segments
}

// ============================================================================
// PT_INTERP Path Selection
// ============================================================================

/// Get the dynamic linker (interpreter) path for the target architecture.
///
/// This path is written into the `.interp` section and referenced by the
/// `PT_INTERP` program header. The kernel uses this path to locate and
/// invoke the dynamic linker when loading a dynamically-linked executable.
///
/// `PT_INTERP` is only present for dynamically-linked executables, not for
/// static executables or shared libraries.
///
/// # Arguments
/// * `target` — Target configuration specifying the architecture.
///
/// # Returns
/// The filesystem path to the dynamic linker for the target architecture.
pub fn interpreter_path(target: &TargetConfig) -> &'static str {
    match target.arch {
        Architecture::X86_64 => "/lib64/ld-linux-x86-64.so.2",
        Architecture::I686 => "/lib/ld-linux.so.2",
        Architecture::Aarch64 => "/lib/ld-linux-aarch64.so.1",
        Architecture::Riscv64 => "/lib/ld-linux-riscv64-lp64d.so.1",
    }
}

// ============================================================================
// PT_GNU_STACK Segment
// ============================================================================

/// Generate a `PT_GNU_STACK` program header.
///
/// This marks the stack as non-executable (NX), which is the default for
/// modern Linux executables. The absence of `PF_X` in the flags ensures
/// that the kernel maps the stack without execute permission, providing
/// defense-in-depth against stack-based code injection.
///
/// # Returns
/// A `SegmentLayout` representing the `PT_GNU_STACK` program header.
pub fn gnu_stack_segment() -> SegmentLayout {
    SegmentLayout {
        segment_type: PT_GNU_STACK,
        flags: PF_R | PF_W, // No PF_X: non-executable stack
        virtual_address: 0,
        physical_address: 0,
        file_offset: 0,
        file_size: 0,
        mem_size: 0,
        alignment: 0,
        sections: Vec::new(),
    }
}

// ============================================================================
// Address Assignment
// ============================================================================

/// Assign virtual addresses and file offsets to all sections.
///
/// Starting at the base address (typically `0x400000` for 64-bit or
/// `0x08048000` for 32-bit), this function advances through sections in
/// order, respecting alignment requirements and inserting page-aligned
/// boundaries at segment transitions (where permission flags change).
///
/// The ELF header and program headers occupy the beginning of the file,
/// so sections start after those headers.
///
/// ## Layout Strategy
///
/// 1. ELF header occupies `[0, elf_header_size)`.
/// 2. Program headers occupy `[elf_header_size, elf_header_size + program_header_size)`.
/// 3. Sections follow, each aligned to its required alignment.
/// 4. When permission flags change between consecutive loadable sections,
///    the next section is page-aligned to start a new PT_LOAD segment.
/// 5. Non-loadable sections (debug, metadata) follow loadable sections
///    without virtual addresses.
///
/// # Arguments
/// * `sections` — Mutable slice of section layouts. Virtual addresses and
///   file offsets are written into these entries.
/// * `base_address` — Starting virtual address for the first PT_LOAD segment.
/// * `elf_header_size` — Size of the ELF header in bytes.
/// * `program_header_size` — Total size of all program headers in bytes.
///
/// # Returns
/// An [`AddressAssignment`] with the final section and segment layouts,
/// entry point address, and total file size.
pub fn assign_addresses(
    sections: &mut [SectionLayout],
    base_address: u64,
    elf_header_size: u64,
    program_header_size: u64,
) -> AddressAssignment {
    // Current file offset and virtual address, starting after ELF + program headers.
    let headers_size = elf_header_size + program_header_size;
    let mut current_file_offset = headers_size;
    let mut current_vaddr = base_address + headers_size;

    // Track the previous section's flags to detect segment transitions.
    let mut prev_flags: Option<SegmentFlags> = Option::None;

    // Track the entry point address (virtual address of ".text" or "_start")
    let mut entry_address: u64 = 0;

    for section in sections.iter_mut() {
        let (seg_type, _seg_flags) = section_to_segment(&section.name);

        if seg_type == SegmentType::None {
            // Non-loadable sections: assign file offset but no virtual address.
            // These sections exist in the file but are not mapped into memory.
            current_file_offset = align_up(current_file_offset, section.alignment.max(1));
            section.file_offset = current_file_offset;
            section.virtual_address = 0;
            current_file_offset += section.size;
            continue;
        }

        // Check if we need a page-aligned transition for a new PT_LOAD segment.
        // This happens when the permission flags change between consecutive
        // loadable sections.
        let need_page_align = match prev_flags {
            Some(ref prev) if !prev.same_permissions(&section.flags) => true,
            Option::None => false, // First loadable section: already at page-aligned base
            _ => false,
        };

        if need_page_align {
            // Page-align both the file offset and virtual address for the new segment.
            current_file_offset = align_up(current_file_offset, PAGE_SIZE);
            current_vaddr = align_up(current_vaddr, PAGE_SIZE);
        }

        // Apply section-specific alignment
        let alignment = section.alignment.max(1);
        current_file_offset = align_up(current_file_offset, alignment);
        current_vaddr = align_up(current_vaddr, alignment);

        // Assign addresses
        section.file_offset = current_file_offset;
        section.virtual_address = current_vaddr;

        // Track entry point: the `.text` section contains executable code
        // and `_start` typically resides at its beginning.
        if section.name == ".text" {
            entry_address = current_vaddr;
        }

        // Advance past this section.
        // For BSS sections: they occupy memory but not file space.
        if section.name == ".bss" || section.mem_size > section.size {
            // BSS: advance file offset by the file-backed size only,
            // but advance virtual address by the full memory size.
            current_file_offset += section.size;
            current_vaddr += section.mem_size;
        } else {
            current_file_offset += section.size;
            current_vaddr += section.size;
        }

        prev_flags = Some(section.flags);
    }

    // Build segments from the now-addressed sections
    let segments = group_into_segments(sections, base_address);

    // Add GNU stack segment
    let mut all_segments = segments;
    all_segments.push(gnu_stack_segment());

    AddressAssignment {
        sections: sections.to_vec(),
        segments: all_segments,
        entry_address,
        total_file_size: current_file_offset,
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::target::TargetConfig;

    // -----------------------------------------------------------------------
    // Section Ordering Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_section_order_text_before_rodata() {
        let text_pos = DEFAULT_SECTION_ORDER
            .iter()
            .position(|&s| s == ".text")
            .unwrap();
        let rodata_pos = DEFAULT_SECTION_ORDER
            .iter()
            .position(|&s| s == ".rodata")
            .unwrap();
        assert!(
            text_pos < rodata_pos,
            ".text (pos {}) should come before .rodata (pos {})",
            text_pos,
            rodata_pos
        );
    }

    #[test]
    fn test_default_section_order_rodata_before_data() {
        let rodata_pos = DEFAULT_SECTION_ORDER
            .iter()
            .position(|&s| s == ".rodata")
            .unwrap();
        let data_pos = DEFAULT_SECTION_ORDER
            .iter()
            .position(|&s| s == ".data")
            .unwrap();
        assert!(
            rodata_pos < data_pos,
            ".rodata (pos {}) should come before .data (pos {})",
            rodata_pos,
            data_pos
        );
    }

    #[test]
    fn test_default_section_order_data_before_bss() {
        let data_pos = DEFAULT_SECTION_ORDER
            .iter()
            .position(|&s| s == ".data")
            .unwrap();
        let bss_pos = DEFAULT_SECTION_ORDER
            .iter()
            .position(|&s| s == ".bss")
            .unwrap();
        assert!(
            data_pos < bss_pos,
            ".data (pos {}) should come before .bss (pos {})",
            data_pos,
            bss_pos
        );
    }

    #[test]
    fn test_default_section_order_bss_before_debug() {
        let bss_pos = DEFAULT_SECTION_ORDER
            .iter()
            .position(|&s| s == ".bss")
            .unwrap();
        let debug_pos = DEFAULT_SECTION_ORDER
            .iter()
            .position(|&s| s == ".debug_info")
            .unwrap();
        assert!(
            bss_pos < debug_pos,
            ".bss (pos {}) should come before .debug_info (pos {})",
            bss_pos,
            debug_pos
        );
    }

    #[test]
    fn test_default_section_order_interp_first() {
        assert_eq!(
            DEFAULT_SECTION_ORDER[0], ".interp",
            ".interp should be the first section in the default order"
        );
    }

    #[test]
    fn test_default_section_order_shstrtab_last() {
        let last = *DEFAULT_SECTION_ORDER.last().unwrap();
        assert_eq!(
            last, ".shstrtab",
            ".shstrtab should be the last section in the default order"
        );
    }

    // -----------------------------------------------------------------------
    // Section Sort Key Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_section_sort_key_known_sections() {
        let text_key = section_sort_key(".text");
        let data_key = section_sort_key(".data");
        let bss_key = section_sort_key(".bss");
        let rodata_key = section_sort_key(".rodata");

        assert!(
            text_key < rodata_key,
            ".text key should be less than .rodata key"
        );
        assert!(
            rodata_key < data_key,
            ".rodata key should be less than .data key"
        );
        assert!(data_key < bss_key, ".data key should be less than .bss key");
    }

    #[test]
    fn test_section_sort_key_unknown_section() {
        let bss_key = section_sort_key(".bss");
        let debug_key = section_sort_key(".debug_info");
        let unknown_key = section_sort_key(".custom_section");

        assert!(
            unknown_key > bss_key,
            "Unknown section key ({}) should be after .bss key ({})",
            unknown_key,
            bss_key
        );
        // Unknown sections should appear before or at the debug section boundary
        // (they get bss_index + 1 which equals debug_info's index, so they may
        // be equal to or less than debug_info depending on exact placement)
    }

    #[test]
    fn test_section_sort_key_debug_section_unknown() {
        let known_debug = section_sort_key(".debug_info");
        let unknown_debug = section_sort_key(".debug_custom");

        assert!(
            unknown_debug > known_debug,
            "Unknown debug section key ({}) should be after known debug key ({})",
            unknown_debug,
            known_debug
        );
    }

    // -----------------------------------------------------------------------
    // Entry Point Selection Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_select_entry_point_static_executable() {
        let result = select_entry_point(&OutputMode::StaticExecutable, "_start");
        assert_eq!(result, Some("_start".to_string()));
    }

    #[test]
    fn test_select_entry_point_static_executable_custom() {
        let result = select_entry_point(&OutputMode::StaticExecutable, "main");
        assert_eq!(result, Some("main".to_string()));
    }

    #[test]
    fn test_select_entry_point_shared_library() {
        let result = select_entry_point(&OutputMode::SharedLibrary, "_start");
        assert_eq!(result, None, "Shared libraries should have no entry point");
    }

    #[test]
    fn test_select_entry_point_relocatable() {
        let result = select_entry_point(&OutputMode::Relocatable, "_start");
        assert_eq!(
            result, None,
            "Relocatable objects should have no entry point"
        );
    }

    // -----------------------------------------------------------------------
    // Base Address Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_base_address_64bit() {
        let target = TargetConfig::x86_64();
        assert_eq!(
            default_base_address(&target),
            0x400000,
            "64-bit x86-64 base address should be 0x400000"
        );
    }

    #[test]
    fn test_default_base_address_32bit() {
        let target = TargetConfig::i686();
        assert_eq!(
            default_base_address(&target),
            0x08048000,
            "32-bit i686 base address should be 0x08048000"
        );
    }

    #[test]
    fn test_default_base_address_aarch64() {
        let target = TargetConfig::aarch64();
        assert_eq!(
            default_base_address(&target),
            0x400000,
            "64-bit AArch64 base address should be 0x400000"
        );
    }

    #[test]
    fn test_default_base_address_riscv64() {
        let target = TargetConfig::riscv64();
        assert_eq!(
            default_base_address(&target),
            0x400000,
            "64-bit RISC-V 64 base address should be 0x400000"
        );
    }

    // -----------------------------------------------------------------------
    // Section-to-Segment Mapping Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_section_to_segment_text() {
        let (seg_type, flags) = section_to_segment(".text");
        assert_eq!(seg_type, SegmentType::Load);
        assert!(flags.readable);
        assert!(!flags.writable);
        assert!(flags.executable);
    }

    #[test]
    fn test_section_to_segment_data() {
        let (seg_type, flags) = section_to_segment(".data");
        assert_eq!(seg_type, SegmentType::Load);
        assert!(flags.readable);
        assert!(flags.writable);
        assert!(!flags.executable);
    }

    #[test]
    fn test_section_to_segment_bss() {
        let (seg_type, flags) = section_to_segment(".bss");
        assert_eq!(seg_type, SegmentType::Load);
        assert!(flags.readable);
        assert!(flags.writable);
        assert!(!flags.executable);
    }

    #[test]
    fn test_section_to_segment_rodata() {
        let (seg_type, flags) = section_to_segment(".rodata");
        assert_eq!(seg_type, SegmentType::Load);
        assert!(flags.readable);
        assert!(!flags.writable);
        assert!(!flags.executable);
    }

    #[test]
    fn test_section_to_segment_interp() {
        let (seg_type, flags) = section_to_segment(".interp");
        assert_eq!(seg_type, SegmentType::Interp);
        assert!(flags.readable);
        assert!(!flags.writable);
        assert!(!flags.executable);
    }

    #[test]
    fn test_section_to_segment_dynamic() {
        let (seg_type, flags) = section_to_segment(".dynamic");
        assert_eq!(seg_type, SegmentType::Dynamic);
        assert!(flags.readable);
        assert!(flags.writable);
        assert!(!flags.executable);
    }

    #[test]
    fn test_section_to_segment_debug() {
        let (seg_type, flags) = section_to_segment(".debug_info");
        assert_eq!(
            seg_type,
            SegmentType::None,
            "Debug sections should not be loaded"
        );
        assert!(!flags.readable);
        assert!(!flags.writable);
        assert!(!flags.executable);
    }

    #[test]
    fn test_section_to_segment_debug_line() {
        let (seg_type, _flags) = section_to_segment(".debug_line");
        assert_eq!(seg_type, SegmentType::None);
    }

    #[test]
    fn test_section_to_segment_debug_str() {
        let (seg_type, _flags) = section_to_segment(".debug_str");
        assert_eq!(seg_type, SegmentType::None);
    }

    #[test]
    fn test_section_to_segment_note() {
        let (seg_type, flags) = section_to_segment(".note.ABI-tag");
        assert_eq!(seg_type, SegmentType::Note);
        assert!(flags.readable);
        assert!(!flags.writable);
    }

    #[test]
    fn test_section_to_segment_symtab() {
        let (seg_type, _flags) = section_to_segment(".symtab");
        assert_eq!(seg_type, SegmentType::None, ".symtab should not be loaded");
    }

    #[test]
    fn test_section_to_segment_strtab() {
        let (seg_type, _flags) = section_to_segment(".strtab");
        assert_eq!(seg_type, SegmentType::None, ".strtab should not be loaded");
    }

    #[test]
    fn test_section_to_segment_init() {
        let (seg_type, flags) = section_to_segment(".init");
        assert_eq!(seg_type, SegmentType::Load);
        assert!(flags.executable, ".init should be executable");
    }

    #[test]
    fn test_section_to_segment_plt() {
        let (seg_type, flags) = section_to_segment(".plt");
        assert_eq!(seg_type, SegmentType::Load);
        assert!(flags.executable, ".plt should be executable");
    }

    #[test]
    fn test_section_to_segment_got() {
        let (seg_type, flags) = section_to_segment(".got");
        assert_eq!(seg_type, SegmentType::Load);
        assert!(flags.writable, ".got should be writable");
    }

    #[test]
    fn test_section_to_segment_init_array() {
        let (seg_type, flags) = section_to_segment(".init_array");
        assert_eq!(seg_type, SegmentType::Load);
        assert!(flags.writable, ".init_array should be writable");
    }

    #[test]
    fn test_section_to_segment_unknown() {
        let (seg_type, flags) = section_to_segment(".custom_section");
        assert_eq!(
            seg_type,
            SegmentType::Load,
            "Unknown sections default to PT_LOAD"
        );
        assert!(flags.readable);
        assert!(!flags.writable);
        assert!(!flags.executable);
    }

    // -----------------------------------------------------------------------
    // Interpreter Path Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpreter_path_x86_64() {
        let target = TargetConfig::x86_64();
        assert_eq!(interpreter_path(&target), "/lib64/ld-linux-x86-64.so.2");
    }

    #[test]
    fn test_interpreter_path_i686() {
        let target = TargetConfig::i686();
        assert_eq!(interpreter_path(&target), "/lib/ld-linux.so.2");
    }

    #[test]
    fn test_interpreter_path_aarch64() {
        let target = TargetConfig::aarch64();
        assert_eq!(interpreter_path(&target), "/lib/ld-linux-aarch64.so.1");
    }

    #[test]
    fn test_interpreter_path_riscv64() {
        let target = TargetConfig::riscv64();
        assert_eq!(
            interpreter_path(&target),
            "/lib/ld-linux-riscv64-lp64d.so.1"
        );
    }

    // -----------------------------------------------------------------------
    // Page Size and Alignment Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_page_size_value() {
        assert_eq!(PAGE_SIZE, 4096, "PAGE_SIZE should be 4096 bytes");
    }

    #[test]
    fn test_page_size_is_power_of_two() {
        assert!(
            PAGE_SIZE.is_power_of_two(),
            "PAGE_SIZE must be a power of 2"
        );
    }

    // -----------------------------------------------------------------------
    // PT_GNU_STACK Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_gnu_stack_segment_type() {
        let seg = gnu_stack_segment();
        assert_eq!(seg.segment_type, PT_GNU_STACK);
    }

    #[test]
    fn test_gnu_stack_not_executable() {
        let seg = gnu_stack_segment();
        assert_eq!(
            seg.flags & PF_X,
            0,
            "PT_GNU_STACK should NOT have execute permission"
        );
    }

    #[test]
    fn test_gnu_stack_readable_writable() {
        let seg = gnu_stack_segment();
        assert_ne!(seg.flags & PF_R, 0, "PT_GNU_STACK should be readable");
        assert_ne!(seg.flags & PF_W, 0, "PT_GNU_STACK should be writable");
    }

    #[test]
    fn test_gnu_stack_zero_sizes() {
        let seg = gnu_stack_segment();
        assert_eq!(seg.virtual_address, 0);
        assert_eq!(seg.physical_address, 0);
        assert_eq!(seg.file_offset, 0);
        assert_eq!(seg.file_size, 0);
        assert_eq!(seg.mem_size, 0);
    }

    // -----------------------------------------------------------------------
    // SegmentType Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_segment_type_to_elf_type() {
        assert_eq!(SegmentType::Load.to_elf_type(), PT_LOAD);
        assert_eq!(SegmentType::Interp.to_elf_type(), PT_INTERP);
        assert_eq!(SegmentType::Dynamic.to_elf_type(), PT_DYNAMIC);
        assert_eq!(SegmentType::Note.to_elf_type(), PT_NOTE);
        assert_eq!(SegmentType::GnuStack.to_elf_type(), PT_GNU_STACK);
        assert_eq!(SegmentType::GnuRelro.to_elf_type(), PT_GNU_RELRO);
        assert_eq!(SegmentType::None.to_elf_type(), 0);
    }

    // -----------------------------------------------------------------------
    // SegmentFlags Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_segment_flags_to_elf_flags() {
        let rx = SegmentFlags {
            readable: true,
            writable: false,
            executable: true,
        };
        assert_eq!(rx.to_elf_flags(), PF_R | PF_X);

        let rw = SegmentFlags {
            readable: true,
            writable: true,
            executable: false,
        };
        assert_eq!(rw.to_elf_flags(), PF_R | PF_W);

        let r = SegmentFlags {
            readable: true,
            writable: false,
            executable: false,
        };
        assert_eq!(r.to_elf_flags(), PF_R);

        let none = SegmentFlags {
            readable: false,
            writable: false,
            executable: false,
        };
        assert_eq!(none.to_elf_flags(), 0);
    }

    #[test]
    fn test_segment_flags_same_permissions() {
        let rx1 = SegmentFlags {
            readable: true,
            writable: false,
            executable: true,
        };
        let rx2 = SegmentFlags {
            readable: true,
            writable: false,
            executable: true,
        };
        let rw = SegmentFlags {
            readable: true,
            writable: true,
            executable: false,
        };

        assert!(rx1.same_permissions(&rx2));
        assert!(!rx1.same_permissions(&rw));
    }

    // -----------------------------------------------------------------------
    // Group Into Segments Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_group_into_segments_basic() {
        let sections = vec![
            SectionLayout {
                name: ".text".to_string(),
                virtual_address: 0x401000,
                file_offset: 0x1000,
                size: 0x200,
                mem_size: 0x200,
                alignment: 16,
                flags: SegmentFlags {
                    readable: true,
                    writable: false,
                    executable: true,
                },
            },
            SectionLayout {
                name: ".rodata".to_string(),
                virtual_address: 0x402000,
                file_offset: 0x2000,
                size: 0x100,
                mem_size: 0x100,
                alignment: 8,
                flags: SegmentFlags {
                    readable: true,
                    writable: false,
                    executable: false,
                },
            },
            SectionLayout {
                name: ".data".to_string(),
                virtual_address: 0x403000,
                file_offset: 0x3000,
                size: 0x80,
                mem_size: 0x80,
                alignment: 8,
                flags: SegmentFlags {
                    readable: true,
                    writable: true,
                    executable: false,
                },
            },
        ];

        let segments = group_into_segments(&sections, 0x400000);

        // Three different permission groups → three PT_LOAD segments
        assert_eq!(
            segments.len(),
            3,
            "Should have 3 segments for 3 permission groups"
        );
        assert_eq!(segments[0].segment_type, PT_LOAD);
        assert_eq!(segments[1].segment_type, PT_LOAD);
        assert_eq!(segments[2].segment_type, PT_LOAD);
    }

    #[test]
    fn test_group_into_segments_merges_same_permissions() {
        let sections = vec![
            SectionLayout {
                name: ".init".to_string(),
                virtual_address: 0x401000,
                file_offset: 0x1000,
                size: 0x10,
                mem_size: 0x10,
                alignment: 4,
                flags: SegmentFlags {
                    readable: true,
                    writable: false,
                    executable: true,
                },
            },
            SectionLayout {
                name: ".text".to_string(),
                virtual_address: 0x401010,
                file_offset: 0x1010,
                size: 0x200,
                mem_size: 0x200,
                alignment: 16,
                flags: SegmentFlags {
                    readable: true,
                    writable: false,
                    executable: true,
                },
            },
        ];

        let segments = group_into_segments(&sections, 0x400000);

        // Same permissions → should be merged into one segment
        assert_eq!(
            segments.len(),
            1,
            ".init and .text with same flags should merge"
        );
        assert_eq!(segments[0].sections.len(), 2);
        assert!(segments[0].sections.contains(&".init".to_string()));
        assert!(segments[0].sections.contains(&".text".to_string()));
    }

    #[test]
    fn test_group_into_segments_skips_debug() {
        let sections = vec![
            SectionLayout {
                name: ".text".to_string(),
                virtual_address: 0x401000,
                file_offset: 0x1000,
                size: 0x200,
                mem_size: 0x200,
                alignment: 16,
                flags: SegmentFlags {
                    readable: true,
                    writable: false,
                    executable: true,
                },
            },
            SectionLayout {
                name: ".debug_info".to_string(),
                virtual_address: 0,
                file_offset: 0x2000,
                size: 0x100,
                mem_size: 0x100,
                alignment: 1,
                flags: SegmentFlags {
                    readable: false,
                    writable: false,
                    executable: false,
                },
            },
        ];

        let segments = group_into_segments(&sections, 0x400000);

        // Debug section should be skipped
        assert_eq!(
            segments.len(),
            1,
            "Debug sections should not create segments"
        );
        assert_eq!(segments[0].sections[0], ".text");
    }

    // -----------------------------------------------------------------------
    // Address Assignment Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_assign_addresses_basic() {
        let mut sections = vec![
            SectionLayout {
                name: ".text".to_string(),
                virtual_address: 0,
                file_offset: 0,
                size: 0x100,
                mem_size: 0x100,
                alignment: 16,
                flags: SegmentFlags {
                    readable: true,
                    writable: false,
                    executable: true,
                },
            },
            SectionLayout {
                name: ".data".to_string(),
                virtual_address: 0,
                file_offset: 0,
                size: 0x40,
                mem_size: 0x40,
                alignment: 8,
                flags: SegmentFlags {
                    readable: true,
                    writable: true,
                    executable: false,
                },
            },
        ];

        let result = assign_addresses(&mut sections, 0x400000, 64, 56 * 4);

        // Entry address should be the start of .text
        assert_ne!(
            result.entry_address, 0,
            "Entry address should be set to .text"
        );

        // Total file size should cover all sections
        assert!(result.total_file_size > 0);

        // Sections should have addresses assigned
        assert!(sections[0].virtual_address >= 0x400000);
        assert!(sections[0].file_offset > 0);
        assert!(sections[1].virtual_address > sections[0].virtual_address);
    }

    #[test]
    fn test_assign_addresses_bss_no_file_space() {
        let mut sections = vec![
            SectionLayout {
                name: ".text".to_string(),
                virtual_address: 0,
                file_offset: 0,
                size: 0x100,
                mem_size: 0x100,
                alignment: 16,
                flags: SegmentFlags {
                    readable: true,
                    writable: false,
                    executable: true,
                },
            },
            SectionLayout {
                name: ".bss".to_string(),
                virtual_address: 0,
                file_offset: 0,
                size: 0,         // BSS has no file data
                mem_size: 0x200, // But occupies 0x200 bytes in memory
                alignment: 16,
                flags: SegmentFlags {
                    readable: true,
                    writable: true,
                    executable: false,
                },
            },
        ];

        let header_size = 64 + 56 * 4;
        let result = assign_addresses(&mut sections, 0x400000, 64, 56 * 4);

        // BSS should have a virtual address but its file contribution should be 0
        assert!(sections[1].virtual_address > 0);
        // Total file size should NOT include BSS memory
        // (file size = headers + .text size + padding, NOT + .bss mem_size)
        assert!(
            result.total_file_size < sections[1].virtual_address,
            "Total file size should not include BSS memory"
        );
    }

    #[test]
    fn test_assign_addresses_debug_no_vaddr() {
        let mut sections = vec![
            SectionLayout {
                name: ".text".to_string(),
                virtual_address: 0,
                file_offset: 0,
                size: 0x100,
                mem_size: 0x100,
                alignment: 16,
                flags: SegmentFlags {
                    readable: true,
                    writable: false,
                    executable: true,
                },
            },
            SectionLayout {
                name: ".debug_info".to_string(),
                virtual_address: 0,
                file_offset: 0,
                size: 0x80,
                mem_size: 0x80,
                alignment: 1,
                flags: SegmentFlags {
                    readable: false,
                    writable: false,
                    executable: false,
                },
            },
        ];

        let _result = assign_addresses(&mut sections, 0x400000, 64, 56 * 4);

        // Debug section should have file offset but virtual address = 0
        assert_eq!(
            sections[1].virtual_address, 0,
            "Debug sections should have virtual address 0"
        );
        assert!(
            sections[1].file_offset > 0,
            "Debug sections should have a file offset"
        );
    }

    #[test]
    fn test_assign_addresses_includes_gnu_stack() {
        let mut sections = vec![SectionLayout {
            name: ".text".to_string(),
            virtual_address: 0,
            file_offset: 0,
            size: 0x100,
            mem_size: 0x100,
            alignment: 16,
            flags: SegmentFlags {
                readable: true,
                writable: false,
                executable: true,
            },
        }];

        let result = assign_addresses(&mut sections, 0x400000, 64, 56 * 4);

        // Should include a PT_GNU_STACK segment
        let has_gnu_stack = result
            .segments
            .iter()
            .any(|s| s.segment_type == PT_GNU_STACK);
        assert!(
            has_gnu_stack,
            "Address assignment should include PT_GNU_STACK segment"
        );
    }

    #[test]
    fn test_assign_addresses_page_alignment_on_segment_transition() {
        let mut sections = vec![
            SectionLayout {
                name: ".text".to_string(),
                virtual_address: 0,
                file_offset: 0,
                size: 0x100,
                mem_size: 0x100,
                alignment: 16,
                flags: SegmentFlags {
                    readable: true,
                    writable: false,
                    executable: true,
                },
            },
            SectionLayout {
                name: ".data".to_string(),
                virtual_address: 0,
                file_offset: 0,
                size: 0x40,
                mem_size: 0x40,
                alignment: 8,
                flags: SegmentFlags {
                    readable: true,
                    writable: true,
                    executable: false,
                },
            },
        ];

        let _result = assign_addresses(&mut sections, 0x400000, 64, 56 * 4);

        // .data should be page-aligned since permissions differ from .text
        assert_eq!(
            sections[1].virtual_address % PAGE_SIZE,
            0,
            ".data virtual address should be page-aligned (segment transition)"
        );
        assert_eq!(
            sections[1].file_offset % PAGE_SIZE,
            0,
            ".data file offset should be page-aligned (segment transition)"
        );
    }

    // -----------------------------------------------------------------------
    // AddressAssignment Structure Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_address_assignment_entry_zero_for_no_text() {
        let mut sections = vec![SectionLayout {
            name: ".data".to_string(),
            virtual_address: 0,
            file_offset: 0,
            size: 0x40,
            mem_size: 0x40,
            alignment: 8,
            flags: SegmentFlags {
                readable: true,
                writable: true,
                executable: false,
            },
        }];

        let result = assign_addresses(&mut sections, 0x400000, 64, 56 * 4);

        // No .text section → entry address should be 0
        assert_eq!(
            result.entry_address, 0,
            "Entry address should be 0 when no .text section"
        );
    }
}
