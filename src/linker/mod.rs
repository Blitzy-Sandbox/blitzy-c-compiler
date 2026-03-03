//! # Integrated ELF Linker
//!
//! This module implements a complete ELF linker that reads relocatable objects
//! and `ar` static archives for system CRT and library linkage, without
//! reliance on external `ld`, `lld`, or `gold`.
//!
//! ## Supported Output Modes
//! - Static executables (default)
//! - Shared libraries (`-shared` + `-fPIC`)
//! - Relocatable objects (`-c`)
//!
//! ## Supported Formats
//! - ELF64 for x86-64, AArch64, RISC-V 64
//! - ELF32 for i686
//!
//! ## Pipeline
//! Object reading → Symbol collection → Symbol resolution →
//! Section merging → Relocation application → ELF output

// ============================================================================
// Submodule declarations
// ============================================================================

pub mod elf;
pub mod archive;
pub mod relocations;
pub mod sections;
pub mod symbols;
pub mod dynamic;
pub mod script;

// ============================================================================
// Standard library imports
// ============================================================================

use std::path::PathBuf;

// ============================================================================
// Internal crate imports
// ============================================================================

use crate::codegen::{self, ObjectCode, Relocation, Section as CodegenSection, Symbol as CodegenSymbol};
use crate::driver::target::{Architecture, TargetConfig};
use crate::common::diagnostics::DiagnosticEmitter;

// ============================================================================
// OutputMode — the three linking modes
// ============================================================================

/// The output mode for the linker, determining what kind of ELF binary to produce.
///
/// This enum controls the overall linking strategy: whether to produce a relocatable
/// object (partial link), a fully resolved static executable, or a shared library
/// with dynamic linking support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    /// Relocatable object file (`-c` flag). Combines input objects into a single
    /// `.o` file without symbol resolution or relocation application.
    Relocatable,
    /// Static executable (default mode). Fully linked binary with all symbols
    /// resolved and relocations applied. Linked against CRT startup objects
    /// and static libraries.
    StaticExecutable,
    /// Shared library (`-shared` + `-fPIC`). Position-independent shared object
    /// with `.dynamic`, `.dynsym`, `.dynstr`, PLT, and GOT sections.
    SharedLibrary,
}

// ============================================================================
// LinkerError — unified error type for all linker failures
// ============================================================================

/// Unified error type encompassing all possible linker failures.
///
/// Each variant wraps enough context for GCC-compatible diagnostic output
/// on stderr. The linker returns `LinkerError` from `link()` and every
/// internal helper function, enabling idiomatic `?` propagation.
#[derive(Debug)]
pub enum LinkerError {
    /// A symbol was referenced but never defined in any input object or library.
    UndefinedSymbol(String),
    /// Two or more strong (global) definitions of the same symbol were found.
    DuplicateSymbol(String),
    /// An I/O error occurred while reading input files or writing output.
    IoError(std::io::Error),
    /// An input file was not a valid ELF object (corrupt header, wrong class, etc.).
    InvalidElfObject(String),
    /// An input archive file was malformed (bad magic, truncated headers, etc.).
    InvalidArchive(String),
    /// A relocation type was encountered that is not supported for the target.
    UnsupportedRelocation(String),
    /// An internal linker logic error that should not occur during normal operation.
    InternalError(String),
}

impl std::fmt::Display for LinkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinkerError::UndefinedSymbol(name) => {
                write!(f, "undefined reference to `{}`", name)
            }
            LinkerError::DuplicateSymbol(name) => {
                write!(f, "multiple definition of `{}`", name)
            }
            LinkerError::IoError(err) => {
                write!(f, "I/O error: {}", err)
            }
            LinkerError::InvalidElfObject(msg) => {
                write!(f, "invalid ELF object: {}", msg)
            }
            LinkerError::InvalidArchive(msg) => {
                write!(f, "invalid archive: {}", msg)
            }
            LinkerError::UnsupportedRelocation(msg) => {
                write!(f, "unsupported relocation: {}", msg)
            }
            LinkerError::InternalError(msg) => {
                write!(f, "internal linker error: {}", msg)
            }
        }
    }
}

impl std::error::Error for LinkerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LinkerError::IoError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for LinkerError {
    fn from(err: std::io::Error) -> Self {
        LinkerError::IoError(err)
    }
}

impl From<elf::ElfError> for LinkerError {
    fn from(err: elf::ElfError) -> Self {
        LinkerError::InvalidElfObject(format!("{}", err))
    }
}

impl From<archive::ArchiveError> for LinkerError {
    fn from(err: archive::ArchiveError) -> Self {
        LinkerError::InvalidArchive(format!("{}", err))
    }
}

impl From<symbols::SymbolError> for LinkerError {
    fn from(err: symbols::SymbolError) -> Self {
        match err {
            symbols::SymbolError::Undefined(name) => LinkerError::UndefinedSymbol(name),
            symbols::SymbolError::Duplicate(name) => LinkerError::DuplicateSymbol(name),
            symbols::SymbolError::ReadError(msg) => LinkerError::InvalidElfObject(msg),
        }
    }
}

impl From<relocations::RelocationError> for LinkerError {
    fn from(err: relocations::RelocationError) -> Self {
        LinkerError::UnsupportedRelocation(format!("{}", err))
    }
}

// ============================================================================
// LinkerConfig — all linker configuration parameters
// ============================================================================

/// Configuration for the linking step, constructed from CLI arguments.
///
/// Contains the output mode, paths, library specifications, target architecture
/// configuration, and entry point. The driver populates this struct from parsed
/// CLI flags before calling `link()`.
#[derive(Debug)]
pub struct LinkerConfig {
    /// Output mode: relocatable, static executable, or shared library.
    pub output_mode: OutputMode,
    /// Path to the output file.
    pub output_path: PathBuf,
    /// Library search paths from `-L` flags, searched in order.
    pub library_paths: Vec<PathBuf>,
    /// Library names from `-l` flags (e.g. `"c"` for `-lc`).
    pub libraries: Vec<String>,
    /// Whether to force static linking (`-static` flag).
    pub force_static: bool,
    /// Target configuration carrying architecture, ELF class, pointer width,
    /// ELF machine type, and CRT/library search paths.
    pub target: TargetConfig,
    /// Entry point symbol name. Defaults to `"_start"`.
    pub entry_point: String,
}

impl LinkerConfig {
    /// Create a `LinkerConfig` with sensible defaults for the given target.
    pub fn new(target: TargetConfig) -> Self {
        LinkerConfig {
            output_mode: OutputMode::StaticExecutable,
            output_path: PathBuf::from("a.out"),
            library_paths: Vec::new(),
            libraries: Vec::new(),
            force_static: false,
            target,
            entry_point: String::from("_start"),
        }
    }
}

// ============================================================================
// LinkerInput — compiled objects and optional debug info
// ============================================================================

/// Input to the linker: compiled object code and optional debug information.
///
/// The driver constructs this after code generation (and optionally DWARF
/// generation) and passes it to `link()`.
#[derive(Debug)]
pub struct LinkerInput {
    /// Compiled object code from the code generation phase. Each `ObjectCode`
    /// contains machine code sections, symbol definitions, and relocations
    /// for one compilation unit.
    pub objects: Vec<ObjectCode>,
    /// Optional DWARF debug info sections, present when `-g` was specified.
    pub debug_sections: Option<DebugSections>,
}

// ============================================================================
// DebugSections — DWARF v4 debug section data
// ============================================================================

/// DWARF v4 debug section data to be included in the output ELF binary.
///
/// When the `-g` flag is specified, the debug info generator produces these
/// byte vectors representing the standard DWARF sections. The linker includes
/// them as non-loadable sections in the output and applies their relocations.
#[derive(Debug)]
pub struct DebugSections {
    /// `.debug_info` — compilation unit DIEs, subprogram DIEs, variable DIEs.
    pub debug_info: Vec<u8>,
    /// `.debug_abbrev` — abbreviation table for DIE encoding.
    pub debug_abbrev: Vec<u8>,
    /// `.debug_line` — line number program mapping addresses to source lines.
    pub debug_line: Vec<u8>,
    /// `.debug_str` — string table for DWARF attributes.
    pub debug_str: Vec<u8>,
    /// `.debug_aranges` — address range lookup table.
    pub debug_aranges: Vec<u8>,
    /// `.debug_frame` — call frame information for stack unwinding.
    pub debug_frame: Vec<u8>,
    /// `.debug_loc` — location list entries for variable tracking.
    pub debug_loc: Vec<u8>,
    /// Relocations that must be applied to the debug sections.
    pub relocations: Vec<Relocation>,
}

// ============================================================================
// CrtObjects — system CRT startup files
// ============================================================================

/// Container for the system C runtime startup objects.
///
/// For static executables, the linker must find and link `crt1.o`, `crti.o`,
/// and `crtn.o` from the system sysroot. These provide `_start`, init/fini
/// array processing, and other startup/shutdown infrastructure.
struct CrtObjects {
    /// `crt1.o` bytes — provides `_start` entry point (calls `__libc_start_main`).
    crt1: Option<Vec<u8>>,
    /// `crti.o` bytes — provides `.init` and `.fini` section prologue.
    crti: Option<Vec<u8>>,
    /// `crtn.o` bytes — provides `.init` and `.fini` section epilogue.
    crtn: Option<Vec<u8>>,
}

// ============================================================================
// LibraryFile — a resolved library from -l flags
// ============================================================================

/// A library file located on disk and loaded into memory.
///
/// Created by `resolve_libraries()` when processing `-l` flags. The `data`
/// field holds the complete file contents (typically an `ar` archive).
struct LibraryFile {
    /// The library name as specified with `-l` (e.g. `"c"` for `-lc`).
    name: String,
    /// Full filesystem path to the resolved library file.
    path: PathBuf,
    /// Complete file contents read into memory.
    data: Vec<u8>,
}

// ============================================================================
// CRT object discovery
// ============================================================================

/// Locate system CRT startup objects for the target architecture.
///
/// Searches paths in this order:
/// 1. User-specified library paths from `-L` flags
/// 2. The target's `crt_search_paths` (architecture-specific system paths)
/// 3. Standard fallback paths based on architecture
///
/// For relocatable output (`-c`), CRT objects are not needed and this returns
/// an empty `CrtObjects`.
fn find_crt_objects(config: &LinkerConfig) -> Result<CrtObjects, LinkerError> {
    // CRT objects are only needed for executables
    if config.output_mode == OutputMode::Relocatable
        || config.output_mode == OutputMode::SharedLibrary
    {
        return Ok(CrtObjects {
            crt1: None,
            crti: None,
            crtn: None,
        });
    }

    // Build the search path list: user -L paths first, then target-specific paths,
    // then architecture-specific fallback paths.
    let search_paths = build_crt_search_paths(config);

    let crt1 = find_file_in_paths("crt1.o", &search_paths);
    let crti = find_file_in_paths("crti.o", &search_paths);
    let crtn = find_file_in_paths("crtn.o", &search_paths);

    Ok(CrtObjects { crt1, crti, crtn })
}

/// Build the ordered list of directories to search for CRT objects.
fn build_crt_search_paths(config: &LinkerConfig) -> Vec<PathBuf> {
    let mut paths = Vec::new();

    // 1. User-specified -L paths
    for p in &config.library_paths {
        paths.push(p.clone());
    }

    // 2. Target crt_search_paths from TargetConfig
    for p in &config.target.crt_search_paths {
        paths.push(PathBuf::from(p));
    }

    // 3. Architecture-specific standard system paths as fallback
    match config.target.arch {
        Architecture::X86_64 => {
            paths.push(PathBuf::from("/usr/lib/x86_64-linux-gnu"));
            paths.push(PathBuf::from("/usr/lib64"));
            paths.push(PathBuf::from("/usr/lib"));
        }
        Architecture::I686 => {
            paths.push(PathBuf::from("/usr/lib/i386-linux-gnu"));
            paths.push(PathBuf::from("/usr/lib32"));
            paths.push(PathBuf::from("/usr/lib"));
        }
        Architecture::Aarch64 => {
            paths.push(PathBuf::from("/usr/aarch64-linux-gnu/lib"));
            paths.push(PathBuf::from("/usr/lib/aarch64-linux-gnu"));
            paths.push(PathBuf::from("/usr/lib"));
        }
        Architecture::Riscv64 => {
            paths.push(PathBuf::from("/usr/riscv64-linux-gnu/lib"));
            paths.push(PathBuf::from("/usr/lib/riscv64-linux-gnu"));
            paths.push(PathBuf::from("/usr/lib"));
        }
    }

    paths
}

/// Search for a file by name in a list of directories, returning its contents
/// if found.
fn find_file_in_paths(filename: &str, paths: &[PathBuf]) -> Option<Vec<u8>> {
    for dir in paths {
        let full_path = dir.join(filename);
        if full_path.exists() {
            if let Ok(data) = std::fs::read(&full_path) {
                return Some(data);
            }
        }
    }
    None
}

// ============================================================================
// Library resolution
// ============================================================================

/// Resolve `-l` flags to actual library files on disk.
///
/// For each library name specified with `-l`, searches for `lib<name>.a` in
/// the library search paths. If `-static` is not set, also searches for
/// `lib<name>.so` (static archives are preferred when both exist).
///
/// Search order:
/// 1. User-specified `-L` paths
/// 2. Target `lib_search_paths` from `TargetConfig`
/// 3. Architecture-specific standard library paths
fn resolve_libraries(config: &LinkerConfig) -> Result<Vec<LibraryFile>, LinkerError> {
    let search_paths = build_lib_search_paths(config);
    let mut resolved = Vec::new();

    for lib_name in &config.libraries {
        let static_name = format!("lib{}.a", lib_name);
        let shared_name = format!("lib{}.so", lib_name);

        // Search for static library first (always preferred for -static,
        // and preferred by default in our linker).
        let found = if config.force_static {
            find_library_file(&static_name, &search_paths)
        } else {
            // Try static first, then shared
            find_library_file(&static_name, &search_paths)
                .or_else(|| find_library_file(&shared_name, &search_paths))
        };

        if let Some((path, data)) = found {
            resolved.push(LibraryFile {
                name: lib_name.clone(),
                path,
                data,
            });
        }
        // Libraries that cannot be found are silently skipped — the linker
        // will report undefined symbol errors if the missing library was
        // actually needed.
    }

    Ok(resolved)
}

/// Build the ordered list of directories to search for libraries.
fn build_lib_search_paths(config: &LinkerConfig) -> Vec<PathBuf> {
    let mut paths = Vec::new();

    // 1. User-specified -L paths
    for p in &config.library_paths {
        paths.push(p.clone());
    }

    // 2. Target lib_search_paths from TargetConfig
    for p in &config.target.lib_search_paths {
        paths.push(PathBuf::from(p));
    }

    // 3. Architecture-specific standard library paths
    match config.target.arch {
        Architecture::X86_64 => {
            paths.push(PathBuf::from("/usr/lib/x86_64-linux-gnu"));
            paths.push(PathBuf::from("/usr/lib64"));
            paths.push(PathBuf::from("/usr/lib"));
        }
        Architecture::I686 => {
            paths.push(PathBuf::from("/usr/lib/i386-linux-gnu"));
            paths.push(PathBuf::from("/usr/lib32"));
            paths.push(PathBuf::from("/usr/lib"));
        }
        Architecture::Aarch64 => {
            paths.push(PathBuf::from("/usr/aarch64-linux-gnu/lib"));
            paths.push(PathBuf::from("/usr/lib/aarch64-linux-gnu"));
            paths.push(PathBuf::from("/usr/lib"));
        }
        Architecture::Riscv64 => {
            paths.push(PathBuf::from("/usr/riscv64-linux-gnu/lib"));
            paths.push(PathBuf::from("/usr/lib/riscv64-linux-gnu"));
            paths.push(PathBuf::from("/usr/lib"));
        }
    }

    paths
}

/// Search for a library file in the given search paths.
///
/// Returns the path and file contents if found, or `None` if the file does
/// not exist in any search directory.
fn find_library_file(filename: &str, paths: &[PathBuf]) -> Option<(PathBuf, Vec<u8>)> {
    for dir in paths {
        let full_path = dir.join(filename);
        if full_path.exists() {
            if let Ok(data) = std::fs::read(&full_path) {
                return Some((full_path, data));
            }
        }
    }
    None
}

// ============================================================================
// Conversion helpers: codegen types → linker-internal types
// ============================================================================

/// Convert a codegen `SectionType` to the ELF `sh_type` constant.
fn section_type_to_elf(st: &codegen::SectionType) -> u32 {
    match st {
        codegen::SectionType::Text => elf::SHT_PROGBITS,
        codegen::SectionType::Data => elf::SHT_PROGBITS,
        codegen::SectionType::Rodata => elf::SHT_PROGBITS,
        codegen::SectionType::Bss => elf::SHT_NOBITS,
        codegen::SectionType::Custom(v) => *v,
    }
}

/// Convert codegen `SectionFlags` to the ELF `sh_flags` bitmask.
fn section_flags_to_elf(flags: &codegen::SectionFlags) -> u64 {
    let mut elf_flags: u64 = 0;
    if flags.allocatable {
        elf_flags |= elf::SHF_ALLOC;
    }
    if flags.writable {
        elf_flags |= elf::SHF_WRITE;
    }
    if flags.executable {
        elf_flags |= elf::SHF_EXECINSTR;
    }
    elf_flags
}

/// Convert a codegen `Section` into a linker `InputSection`.
fn codegen_section_to_input(
    section: &CodegenSection,
    object_index: usize,
    section_index: usize,
) -> sections::InputSection {
    let elf_type = section_type_to_elf(&section.section_type);
    let elf_flags = section_flags_to_elf(&section.flags);
    let mem_size = if elf_type == elf::SHT_NOBITS {
        section.data.len() as u64
    } else {
        section.data.len() as u64
    };

    sections::InputSection {
        name: section.name.clone(),
        data: section.data.clone(),
        section_type: elf_type,
        flags: elf_flags,
        alignment: section.alignment as u64,
        mem_size,
        object_index,
        original_index: section_index,
    }
}

/// Convert a codegen `Symbol` to a linker `symbols::Symbol`.
///
/// The codegen `Symbol.section_index` is a 0-based index into
/// `ObjectCode.sections`. This maps directly to the `original_index` used
/// when building `InputSection` entries, so the linker's section mapping
/// lookup `(source_object, section_index)` will find the correct section.
///
/// The `is_definition` flag is propagated directly — it is the authoritative
/// source for distinguishing defined symbols from undefined references,
/// independent of section_index value.
fn codegen_symbol_to_linker(
    sym: &CodegenSymbol,
    object_index: usize,
) -> symbols::Symbol {
    let binding = match sym.binding {
        codegen::SymbolBinding::Local => symbols::SymbolBinding::Local,
        codegen::SymbolBinding::Global => symbols::SymbolBinding::Global,
        codegen::SymbolBinding::Weak => symbols::SymbolBinding::Weak,
    };
    let symbol_type = match sym.symbol_type {
        codegen::SymbolType::Function => symbols::SymbolType::Function,
        codegen::SymbolType::Object => symbols::SymbolType::Object,
        codegen::SymbolType::NoType => symbols::SymbolType::NoType,
        codegen::SymbolType::Section => symbols::SymbolType::Section,
    };
    let visibility = match sym.visibility {
        codegen::SymbolVisibility::Default => symbols::SymbolVisibility::Default,
        codegen::SymbolVisibility::Hidden => symbols::SymbolVisibility::Hidden,
        codegen::SymbolVisibility::Protected => symbols::SymbolVisibility::Protected,
    };

    symbols::Symbol {
        name: sym.name.clone(),
        binding,
        symbol_type,
        visibility,
        // Keep the 0-based ObjectCode section index here. The linker works
        // with 0-based merged-section indices internally. The +1 mapping to
        // ELF section header indices (where 0 = null section) happens during
        // ELF symtab generation in generate_symtab().
        section_index: sym.section_index as u16,
        value: sym.offset,
        size: sym.size,
        source_object: object_index,
        is_definition: sym.is_definition,
    }
}

/// Convert a codegen `RelocationType` to the ELF relocation type constant.
fn reloc_type_to_elf(rt: &codegen::RelocationType) -> u32 {
    match rt {
        codegen::RelocationType::X86_64_64 => relocations::R_X86_64_64,
        codegen::RelocationType::X86_64_PC32 => relocations::R_X86_64_PC32,
        codegen::RelocationType::X86_64_PLT32 => relocations::R_X86_64_PLT32,
        codegen::RelocationType::X86_64_GOT64 => relocations::R_X86_64_GOT32,
        codegen::RelocationType::X86_64_GOTPCREL => relocations::R_X86_64_GOTPCREL,
        codegen::RelocationType::I386_32 => relocations::R_386_32,
        codegen::RelocationType::I386_PC32 => relocations::R_386_PC32,
        codegen::RelocationType::I386_GOT32 => relocations::R_386_GOT32,
        codegen::RelocationType::I386_PLT32 => relocations::R_386_PLT32,
        codegen::RelocationType::Aarch64_ABS64 => relocations::R_AARCH64_ABS64,
        codegen::RelocationType::Aarch64_ABS32 => relocations::R_AARCH64_ABS32,
        codegen::RelocationType::Aarch64_CALL26 => relocations::R_AARCH64_CALL26,
        codegen::RelocationType::Aarch64_JUMP26 => relocations::R_AARCH64_JUMP26,
        codegen::RelocationType::Aarch64_ADR_PREL_PG_HI21 => {
            relocations::R_AARCH64_ADR_PREL_PG_HI21
        }
        codegen::RelocationType::Aarch64_ADD_ABS_LO12_NC => {
            relocations::R_AARCH64_ADD_ABS_LO12_NC
        }
        codegen::RelocationType::Riscv_64 => relocations::R_RISCV_64,
        codegen::RelocationType::Riscv_32 => relocations::R_RISCV_32,
        codegen::RelocationType::Riscv_Branch => relocations::R_RISCV_BRANCH,
        codegen::RelocationType::Riscv_Jal => relocations::R_RISCV_JAL,
        codegen::RelocationType::Riscv_Call => relocations::R_RISCV_CALL,
        codegen::RelocationType::Riscv_Pcrel_Hi20 => relocations::R_RISCV_PCREL_HI20,
        codegen::RelocationType::Riscv_Pcrel_Lo12_I => relocations::R_RISCV_PCREL_LO12_I,
        codegen::RelocationType::Riscv_Hi20 => relocations::R_RISCV_HI20,
        codegen::RelocationType::Riscv_Lo12_I => relocations::R_RISCV_LO12_I,
    }
}

// ============================================================================
// Primary link() function — the linker orchestrator
// ============================================================================

/// Link compiled objects into a final ELF binary.
///
/// This is the main entry point for the integrated linker, called by the
/// driver after code generation (and optionally debug info generation).
///
/// # Pipeline
///
/// 1. **Validate inputs** — ensure all objects target the same architecture.
/// 2. **Locate CRT objects** — find `crt1.o`, `crti.o`, `crtn.o` for executables.
/// 3. **Resolve libraries** — find `lib*.a` files for each `-l` flag.
/// 4. **Collect sections** — extract sections from all input objects.
/// 5. **Collect symbols** — extract symbols from all input objects and CRT files.
/// 6. **Process archives** — pull in library members that define needed symbols.
/// 7. **Resolve symbols** — ensure every reference has exactly one definition.
/// 8. **Merge sections** — combine same-name sections from all objects.
/// 9. **Compute layout** — assign virtual addresses and file offsets.
/// 10. **Assign symbol addresses** — update resolved symbols with final addresses.
/// 11. **Apply relocations** — patch machine code with computed addresses.
/// 12. **Generate dynamic sections** — for shared library output.
/// 13. **Include debug sections** — if `-g` was specified.
/// 14. **Emit ELF** — write the final ELF32/ELF64 binary.
///
/// # Errors
///
/// Returns `LinkerError` if any step fails (undefined symbols, I/O errors,
/// unsupported relocations, etc.). All errors are reported in GCC-compatible
/// format via the diagnostic system.
pub fn link(input: LinkerInput, config: &LinkerConfig) -> Result<Vec<u8>, LinkerError> {
    let mut diag = DiagnosticEmitter::new();
    let is_64bit = config.target.is_64bit();
    let machine = config.target.elf_machine;

    // Dispatch to mode-specific linking pipeline.
    match config.output_mode {
        OutputMode::Relocatable => link_relocatable(input, config, is_64bit, machine),
        OutputMode::StaticExecutable => {
            link_static_executable(input, config, &mut diag, is_64bit, machine)
        }
        OutputMode::SharedLibrary => {
            link_shared_library(input, config, &mut diag, is_64bit, machine)
        }
    }
}

// ============================================================================
// Relocatable linking — partial link producing a single .o file
// ============================================================================

/// Link in relocatable mode: merge all input objects into a single `.o` file
/// without symbol resolution or relocation application.
fn link_relocatable(
    input: LinkerInput,
    _config: &LinkerConfig,
    is_64bit: bool,
    machine: u16,
) -> Result<Vec<u8>, LinkerError> {
    let mut all_sections = Vec::new();

    // Collect sections from all input objects.
    for (obj_idx, obj) in input.objects.iter().enumerate() {
        for (sec_idx, sec) in obj.sections.iter().enumerate() {
            all_sections.push(codegen_section_to_input(sec, obj_idx, sec_idx));
        }
    }

    // Merge sections (without layout assignment, since this is relocatable).
    let section_order = default_section_order();
    let merged = sections::merge_sections(all_sections, &section_order);

    // Build a mapping from (object_index, original_section_index) →
    // merged ELF section index (1-based, since section 0 is the null header).
    let mut section_index_map: std::collections::HashMap<(usize, usize), u16> =
        std::collections::HashMap::new();
    for (merged_idx, sec) in merged.iter().enumerate() {
        let elf_section_idx = (merged_idx + 1) as u16; // +1 for null section
        for mapping in &sec.input_mappings {
            section_index_map.insert(
                (mapping.object_index, mapping.original_index),
                elf_section_idx,
            );
        }
    }

    // Build symbol table from all input objects, remapping section indices
    // from codegen 0-based to ELF 1-based via the merged section mapping.
    let mut resolver = symbols::SymbolResolver::new();
    for (obj_idx, obj) in input.objects.iter().enumerate() {
        let syms: Vec<symbols::Symbol> = obj
            .symbols
            .iter()
            .map(|s| {
                let mut sym = codegen_symbol_to_linker(s, obj_idx);
                if sym.is_definition && sym.section_index < elf::SHN_ABS {
                    if let Some(&elf_idx) =
                        section_index_map.get(&(obj_idx, sym.section_index as usize))
                    {
                        sym.section_index = elf_idx;
                    }
                }
                sym
            })
            .collect();
        resolver.add_object_symbols(obj_idx, &syms);
    }
    let (symtab_data, strtab_data) = resolver.generate_symtab(is_64bit);

    // Build the ELF output using the writer.
    let mut writer = elf::ElfWriter::new(is_64bit, machine);
    writer.set_type(elf::ET_REL);
    writer.set_entry(0);

    // Add merged sections.
    for sec in &merged {
        // For SHT_NOBITS (.bss) sections, the merged data is empty but
        // mem_size tracks the actual virtual size. Set data to a zero-filled
        // vector so the ELF writer's data.len() reports the correct sh_size.
        let section_data = if sec.section_type == elf::SHT_NOBITS && sec.data.is_empty() && sec.mem_size > 0 {
            vec![0u8; sec.mem_size as usize]
        } else {
            sec.data.clone()
        };
        writer.add_section(elf::OutputSection {
            name: sec.name.clone(),
            data: section_data,
            header: elf::OutputSectionHeader {
                sh_type: sec.section_type,
                sh_flags: sec.flags,
                sh_addr: 0,
                sh_addralign: sec.alignment,
                sh_entsize: 0,
                sh_link: 0,
                sh_info: 0,
            },
            file_offset: None,
        });
    }

    // ---------------------------------------------------------------
    // Build relocation sections (.rela.*) for the relocatable output.
    // ---------------------------------------------------------------
    // Build a symbol name → symtab index map. The symtab order is:
    //   [0] = null symbol
    //   [1..=local_count] = local symbols
    //   [local_count+1..] = global symbols
    // We need to map symbol names to their indices for relocation entries.
    let sym_name_to_idx: std::collections::HashMap<String, u32> = {
        let mut map = std::collections::HashMap::new();
        let mut idx: u32 = 1; // 0 is null symbol
        // CRITICAL: Must mirror generate_symtab order exactly.
        // generate_symtab emits: [null, locals (insertion order), globals (alphabetically sorted)]
        // First pass: locals in insertion order
        for (_obj_idx, obj) in input.objects.iter().enumerate() {
            for s in &obj.symbols {
                if s.binding == codegen::SymbolBinding::Local && !s.name.is_empty() {
                    map.entry(s.name.clone()).or_insert(idx);
                    idx += 1;
                }
            }
        }
        // Second pass: collect all global symbol names, then sort alphabetically
        // to match generate_symtab's alphabetical ordering of globals.
        let mut global_names: Vec<String> = Vec::new();
        for obj in input.objects.iter() {
            for s in &obj.symbols {
                if s.binding != codegen::SymbolBinding::Local && !s.name.is_empty() {
                    if !global_names.contains(&s.name) {
                        global_names.push(s.name.clone());
                    }
                }
            }
        }
        global_names.sort();
        for name in global_names {
            map.entry(name).or_insert_with(|| {
                let i = idx;
                idx += 1;
                i
            });
        }
        map
    };

    // Build a (object_index, codegen_section_index) → merged_offset mapping
    // so we can adjust relocation offsets within merged sections.
    let mut offset_map: std::collections::HashMap<(usize, usize), u64> =
        std::collections::HashMap::new();
    for sec in &merged {
        for mapping in &sec.input_mappings {
            offset_map.insert(
                (mapping.object_index, mapping.original_index),
                mapping.output_offset,
            );
        }
    }

    // Group relocations by their target merged section name.
    let mut relocs_by_section: std::collections::HashMap<String, Vec<u8>> =
        std::collections::HashMap::new();
    let mut rela_section_targets: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new(); // rela name → target section ELF index

    for (obj_idx, obj) in input.objects.iter().enumerate() {
        for reloc in &obj.relocations {
            // Find which merged section this relocation targets.
            if let Some(&elf_sec_idx) = section_index_map.get(&(obj_idx, reloc.section_index)) {
                let merged_idx = (elf_sec_idx as usize) - 1; // back to 0-based merged index
                if merged_idx < merged.len() {
                    let sec_name = &merged[merged_idx].name;
                    let rela_name = format!(".rela{}", sec_name);

                    // Adjust offset: add the input section's offset within the merged section.
                    let base_offset = offset_map
                        .get(&(obj_idx, reloc.section_index))
                        .copied()
                        .unwrap_or(0);
                    let adjusted_offset = reloc.offset + base_offset;

                    // Look up symbol index.
                    let sym_idx = sym_name_to_idx.get(&reloc.symbol).copied().unwrap_or(0);

                    let elf_reloc_type = reloc_type_to_elf(&reloc.reloc_type);

                    let entry = relocs_by_section.entry(rela_name.clone()).or_insert_with(Vec::new);
                    rela_section_targets.entry(rela_name).or_insert(elf_sec_idx as usize);

                    if is_64bit {
                        elf::write_u64_le(entry, adjusted_offset);
                        elf::write_u64_le(entry, elf::elf64_r_info(sym_idx, elf_reloc_type));
                        elf::write_i64_le(entry, reloc.addend);
                    } else {
                        elf::write_u32_le(entry, adjusted_offset as u32);
                        elf::write_u32_le(entry, elf::elf32_r_info(sym_idx, elf_reloc_type));
                    }
                }
            }
        }
    }

    // Count how many rela sections we'll add (needed for strtab/symtab index).
    let rela_section_count = relocs_by_section.len();

    // Section ordering: content sections (already added), then rela sections,
    // then strtab, then symtab.
    let symtab_entsize = if is_64bit { 24u64 } else { 16u64 };
    let rela_entsize = if is_64bit { 24u64 } else { 8u64 };
    // strtab index = 1 (null) + merged_content + rela_count
    let strtab_section_idx = 1 + merged.len() + rela_section_count;
    // symtab index = strtab + 1
    let symtab_section_idx = strtab_section_idx + 1;

    // Add relocation sections (.rela.*) BEFORE strtab/symtab.
    let mut rela_names_sorted: Vec<String> = relocs_by_section.keys().cloned().collect();
    rela_names_sorted.sort();
    for rela_name in &rela_names_sorted {
        let rela_data = relocs_by_section.get(rela_name).unwrap();
        let target_elf_idx = rela_section_targets.get(rela_name).copied().unwrap_or(1);
        writer.add_section(elf::OutputSection {
            name: rela_name.clone(),
            data: rela_data.clone(),
            header: elf::OutputSectionHeader {
                sh_type: elf::SHT_RELA,
                sh_flags: 0,
                sh_addr: 0,
                sh_addralign: if is_64bit { 8 } else { 4 },
                sh_entsize: rela_entsize,
                sh_link: symtab_section_idx as u32,  // points to .symtab
                sh_info: target_elf_idx as u32,       // section being relocated
            },
            file_offset: None,
        });
    }

    // Add strtab.
    writer.add_section(elf::OutputSection {
        name: String::from(".strtab"),
        data: strtab_data,
        header: elf::OutputSectionHeader {
            sh_type: elf::SHT_STRTAB,
            sh_flags: 0,
            sh_addr: 0,
            sh_addralign: 1,
            sh_entsize: 0,
            sh_link: 0,
            sh_info: 0,
        },
        file_offset: None,
    });

    // sh_link = strtab section index; sh_info = first non-local symbol index
    let first_global = 1 + resolver.local_symbol_count();
    writer.add_section(elf::OutputSection {
        name: String::from(".symtab"),
        data: symtab_data,
        header: elf::OutputSectionHeader {
            sh_type: elf::SHT_SYMTAB,
            sh_flags: 0,
            sh_addr: 0,
            sh_addralign: if is_64bit { 8 } else { 4 },
            sh_entsize: symtab_entsize,
            sh_link: strtab_section_idx as u32,
            sh_info: first_global as u32,
        },
        file_offset: None,
    });

    // Add debug sections if present.
    if let Some(ref dbg) = input.debug_sections {
        add_debug_sections_to_writer(&mut writer, dbg);
    }

    Ok(writer.write())
}

// ============================================================================
// Built-in startup stub
// ============================================================================

/// Generate a minimal `_start` function and a `_start` symbol for the target
/// architecture.  This replaces system CRT objects (`crt1.o`, `crti.o`,
/// Machine code for a minimal builtin printf/puts implementation (x86-64).
/// Compiled from freestanding C using write(2) syscall directly — zero libc
/// dependency. Handles: %d, %i, %u, %s, %c, %x, %p, %ld, %lu, %%, and literal text.
#[rustfmt::skip]
const BUILTIN_PRINTF_X86_64: [u8; 1516] = [
    0xf3,0x0f,0x1e,0xfa,0x41,0x57,0x41,0x56,0x41,0x55,0x41,0x54,0x55,0x53,0x48,0x89,0x74,0x24,0xd0,0x48,
    0x89,0x54,0x24,0xd8,0x48,0x89,0x4c,0x24,0xe0,0x4c,0x89,0x44,0x24,0xe8,0x4c,0x89,0x4c,0x24,0xf0,0xc7,
    0x44,0x24,0xb0,0x08,0x00,0x00,0x00,0x48,0x8d,0x44,0x24,0x38,0x48,0x89,0x44,0x24,0xb8,0x48,0x8d,0x44,
    0x24,0xc8,0x48,0x89,0x44,0x24,0xc0,0x0f,0xb6,0x07,0x84,0xc0,0x0f,0x84,0x36,0x05,0x00,0x00,0x49,0x89,
    0xfa,0x41,0xb9,0x00,0x00,0x00,0x00,0xbb,0x01,0x00,0x00,0x00,0x41,0xbc,0x01,0x00,0x00,0x00,0x41,0xbd,
    0x17,0x00,0x00,0x00,0xbd,0x00,0x00,0x00,0x00,0xe9,0x32,0x01,0x00,0x00,0x41,0x0f,0xb6,0x02,0x84,0xc0,
    0x74,0x33,0x4d,0x89,0xd0,0x3c,0x25,0x74,0x2c,0x49,0x83,0xc0,0x01,0x41,0x0f,0xb6,0x00,0x84,0xc0,0x74,
    0x04,0x3c,0x25,0x75,0xf0,0x4c,0x89,0xc2,0x4c,0x29,0xd2,0x41,0x89,0xd6,0x85,0xd2,0x0f,0x8f,0xd0,0x04,
    0x00,0x00,0x45,0x01,0xf1,0x4d,0x89,0xc2,0xe9,0xeb,0x00,0x00,0x00,0x4d,0x89,0xd0,0x41,0xbe,0x00,0x00,
    0x00,0x00,0xeb,0xea,0x49,0x83,0xc2,0x02,0x41,0x89,0xd8,0xe9,0xfa,0x00,0x00,0x00,0x8b,0x44,0x24,0xb0,
    0x83,0xf8,0x2f,0x77,0x40,0x89,0xc2,0x48,0x03,0x54,0x24,0xc0,0x83,0xc0,0x08,0x89,0x44,0x24,0xb0,0x48,
    0x8b,0x02,0x48,0x85,0xc0,0x78,0x3a,0x48,0x89,0xc1,0xc6,0x44,0x24,0xaf,0x00,0x0f,0x85,0x59,0x01,0x00,
    0x00,0xc6,0x44,0x24,0xae,0x30,0x41,0x89,0xd8,0x48,0x8d,0x74,0x24,0xae,0x49,0x63,0xd0,0x89,0xd8,0x4c,
    0x89,0xe7,0x0f,0x05,0xe9,0x84,0x00,0x00,0x00,0x48,0x8b,0x54,0x24,0xb8,0x48,0x8d,0x42,0x08,0x48,0x89,
    0x44,0x24,0xb8,0xeb,0xbe,0x48,0xf7,0xd8,0x48,0x89,0xc1,0xc6,0x44,0x24,0xaf,0x00,0x41,0x89,0xde,0x48,
    0x8d,0x7c,0x24,0xae,0xbe,0x17,0x00,0x00,0x00,0x49,0xbb,0xcd,0xcc,0xcc,0xcc,0xcc,0xcc,0xcc,0xcc,0x41,
    0x89,0xf0,0x83,0xee,0x01,0x48,0x89,0xc8,0x49,0xf7,0xe3,0x48,0xc1,0xea,0x03,0x4c,0x8d,0x3c,0x92,0x4d,
    0x01,0xff,0x48,0x89,0xc8,0x4c,0x29,0xf8,0x83,0xc0,0x30,0x88,0x07,0x48,0x89,0xc8,0x48,0x89,0xd1,0x48,
    0x83,0xef,0x01,0x48,0x83,0xf8,0x09,0x77,0xce,0x45,0x85,0xf6,0x74,0x0c,0x41,0x8d,0x70,0xfe,0x48,0x63,
    0xc6,0xc6,0x44,0x04,0x98,0x2d,0x45,0x89,0xe8,0x41,0x29,0xf0,0x45,0x85,0xc0,0x0f,0x8f,0xc4,0x00,0x00,
    0x00,0x45,0x01,0xc1,0x49,0x83,0xc2,0x01,0x41,0x0f,0xb6,0x02,0x84,0xc0,0x0f,0x84,0xe6,0x03,0x00,0x00,
    0x3c,0x25,0x0f,0x85,0xc6,0xfe,0xff,0xff,0x41,0x80,0x7a,0x01,0x6c,0x0f,0x84,0x01,0xff,0xff,0xff,0x49,
    0x83,0xc2,0x01,0x41,0x89,0xe8,0x41,0x0f,0xb6,0x02,0x3c,0x64,0x0f,0x84,0xfa,0xfe,0xff,0xff,0x3c,0x69,
    0x0f,0x84,0xf2,0xfe,0xff,0xff,0x3c,0x75,0x0f,0x84,0x88,0x00,0x00,0x00,0x3c,0x73,0x0f,0x84,0x38,0x01,
    0x00,0x00,0x3c,0x63,0x0f,0x84,0xa3,0x01,0x00,0x00,0x3c,0x25,0x0f,0x84,0xe5,0x01,0x00,0x00,0x3c,0x78,
    0x0f,0x84,0xff,0x01,0x00,0x00,0x3c,0x70,0x0f,0x84,0x95,0x02,0x00,0x00,0xc6,0x44,0x24,0x97,0x25,0xba,
    0x01,0x00,0x00,0x00,0x48,0x8d,0x74,0x24,0x97,0xb8,0x01,0x00,0x00,0x00,0x48,0x89,0xd7,0x0f,0x05,0x45,
    0x85,0xc0,0x0f,0x85,0x3d,0x03,0x00,0x00,0x41,0x0f,0xb6,0x02,0x88,0x44,0x24,0x97,0xba,0x01,0x00,0x00,
    0x00,0x48,0x8d,0x74,0x24,0x97,0xb8,0x01,0x00,0x00,0x00,0x48,0x89,0xd7,0x0f,0x05,0x47,0x8d,0x4c,0x08,
    0x02,0xe9,0x4a,0xff,0xff,0xff,0x41,0xbe,0x00,0x00,0x00,0x00,0xe9,0xd6,0xfe,0xff,0xff,0x48,0x63,0xf6,
    0x48,0x8d,0x74,0x34,0x98,0xe9,0x9c,0xfe,0xff,0xff,0x8b,0x44,0x24,0xb0,0x83,0xf8,0x2f,0x77,0x3c,0x89,
    0xc2,0x48,0x03,0x54,0x24,0xc0,0x83,0xc0,0x08,0x89,0x44,0x24,0xb0,0x48,0x8b,0x0a,0xc6,0x44,0x24,0xaf,
    0x00,0x48,0x85,0xc9,0x75,0x31,0xc6,0x44,0x24,0xae,0x30,0x41,0xb8,0x01,0x00,0x00,0x00,0x48,0x8d,0x74,
    0x24,0xae,0x49,0x63,0xd0,0xb8,0x01,0x00,0x00,0x00,0xbf,0x01,0x00,0x00,0x00,0x0f,0x05,0xeb,0x5e,0x48,
    0x8b,0x54,0x24,0xb8,0x48,0x8d,0x42,0x08,0x48,0x89,0x44,0x24,0xb8,0xeb,0xc2,0xbe,0x16,0x00,0x00,0x00,
    0x48,0xbf,0xcd,0xcc,0xcc,0xcc,0xcc,0xcc,0xcc,0xcc,0x48,0x89,0xc8,0x48,0xf7,0xe7,0x48,0xc1,0xea,0x03,
    0x4c,0x8d,0x04,0x92,0x4d,0x01,0xc0,0x48,0x89,0xc8,0x4c,0x29,0xc0,0x83,0xc0,0x30,0x88,0x44,0x34,0x98,
    0x48,0x89,0xc8,0x48,0x89,0xd1,0x48,0x89,0xf2,0x48,0x83,0xee,0x01,0x48,0x83,0xf8,0x09,0x77,0xcf,0x41,
    0xb8,0x17,0x00,0x00,0x00,0x41,0x29,0xd0,0x45,0x85,0xc0,0x7f,0x08,0x45,0x01,0xc1,0xe9,0x87,0xfe,0xff,
    0xff,0x48,0x63,0xd2,0x48,0x8d,0x74,0x14,0x98,0xe9,0x7c,0xff,0xff,0xff,0x8b,0x44,0x24,0xb0,0x83,0xf8,
    0x2f,0x77,0x41,0x89,0xc2,0x48,0x03,0x54,0x24,0xc0,0x83,0xc0,0x08,0x89,0x44,0x24,0xb0,0x48,0x8b,0x32,
    0x48,0x85,0xf6,0x0f,0x84,0x57,0xfe,0xff,0xff,0x80,0x3e,0x00,0x74,0x43,0xb8,0x01,0x00,0x00,0x00,0x48,
    0x89,0xc2,0x48,0x83,0xc0,0x01,0x80,0x7c,0x06,0xff,0x00,0x75,0xf2,0x41,0x89,0xd0,0x85,0xd2,0x7f,0x18,
    0x45,0x01,0xc1,0xe9,0x30,0xfe,0xff,0xff,0x48,0x8b,0x54,0x24,0xb8,0x48,0x8d,0x42,0x08,0x48,0x89,0x44,
    0x24,0xb8,0xeb,0xbd,0x48,0x63,0xd2,0xb8,0x01,0x00,0x00,0x00,0xbf,0x01,0x00,0x00,0x00,0x0f,0x05,0xeb,
    0xd7,0x41,0xb8,0x00,0x00,0x00,0x00,0xeb,0xcf,0x8b,0x44,0x24,0xb0,0x83,0xf8,0x2f,0x77,0x31,0x89,0xc2,
    0x48,0x03,0x54,0x24,0xc0,0x83,0xc0,0x08,0x89,0x44,0x24,0xb0,0x8b,0x02,0x88,0x44,0x24,0x97,0xba,0x01,
    0x00,0x00,0x00,0x48,0x8d,0x74,0x24,0x97,0xb8,0x01,0x00,0x00,0x00,0x48,0x89,0xd7,0x0f,0x05,0x41,0x83,
    0xc1,0x01,0xe9,0xcd,0xfd,0xff,0xff,0x48,0x8b,0x54,0x24,0xb8,0x48,0x8d,0x42,0x08,0x48,0x89,0x44,0x24,
    0xb8,0xeb,0xcd,0xc6,0x44,0x24,0x97,0x25,0xba,0x01,0x00,0x00,0x00,0x48,0x8d,0x74,0x24,0x97,0xb8,0x01,
    0x00,0x00,0x00,0x48,0x89,0xd7,0x0f,0x05,0x41,0x83,0xc1,0x01,0xe9,0x9b,0xfd,0xff,0xff,0x8b,0x44,0x24,
    0xb0,0x83,0xf8,0x2f,0x77,0x3c,0x89,0xc2,0x48,0x03,0x54,0x24,0xc0,0x83,0xc0,0x08,0x89,0x44,0x24,0xb0,
    0x48,0x8b,0x02,0xc6,0x44,0x24,0xaf,0x00,0x48,0x85,0xc0,0x75,0x31,0xc6,0x44,0x24,0xae,0x30,0x41,0xb8,
    0x01,0x00,0x00,0x00,0x48,0x8d,0x74,0x24,0xae,0x49,0x63,0xd0,0xb8,0x01,0x00,0x00,0x00,0xbf,0x01,0x00,
    0x00,0x00,0x0f,0x05,0xeb,0x47,0x48,0x8b,0x54,0x24,0xb8,0x48,0x8d,0x42,0x08,0x48,0x89,0x44,0x24,0xb8,
    0xeb,0xc2,0xbe,0x16,0x00,0x00,0x00,0xeb,0x03,0x48,0x89,0xd6,0x89,0xc1,0x83,0xe1,0x0f,0x8d,0x79,0x30,
    0x8d,0x51,0x57,0x83,0xf9,0x09,0x0f,0x4e,0xd7,0x88,0x54,0x34,0x98,0x48,0x8d,0x56,0xff,0x48,0xc1,0xe8,
    0x04,0x75,0xde,0x41,0xb8,0x17,0x00,0x00,0x00,0x41,0x29,0xf0,0x45,0x85,0xc0,0x7f,0x08,0x45,0x01,0xc1,
    0xe9,0x07,0xfd,0xff,0xff,0x48,0x63,0xf6,0x48,0x8d,0x74,0x34,0x98,0xeb,0x96,0x8b,0x44,0x24,0xb0,0x83,
    0xf8,0x2f,0x77,0x67,0x89,0xc2,0x48,0x03,0x54,0x24,0xc0,0x83,0xc0,0x08,0x89,0x44,0x24,0xb0,0x4c,0x8b,
    0x02,0xc6,0x44,0x24,0x97,0x30,0x41,0xbe,0x01,0x00,0x00,0x00,0xba,0x01,0x00,0x00,0x00,0x48,0x8d,0x74,
    0x24,0x97,0x44,0x89,0xf0,0x48,0x89,0xd7,0x0f,0x05,0xc6,0x44,0x24,0x97,0x78,0x44,0x89,0xf0,0x0f,0x05,
    0x41,0x83,0xc1,0x02,0xc6,0x44,0x24,0xaf,0x00,0x4d,0x85,0xc0,0x75,0x31,0xc6,0x44,0x24,0xae,0x30,0x41,
    0xb8,0x01,0x00,0x00,0x00,0x48,0x8d,0x74,0x24,0xae,0x49,0x63,0xd0,0xb8,0x01,0x00,0x00,0x00,0xbf,0x01,
    0x00,0x00,0x00,0x0f,0x05,0xeb,0x48,0x48,0x8b,0x54,0x24,0xb8,0x48,0x8d,0x42,0x08,0x48,0x89,0x44,0x24,
    0xb8,0xeb,0x97,0xb9,0x16,0x00,0x00,0x00,0xeb,0x03,0x48,0x89,0xc1,0x44,0x89,0xc2,0x83,0xe2,0x0f,0x8d,
    0x72,0x30,0x8d,0x42,0x57,0x83,0xfa,0x09,0x0f,0x4e,0xc6,0x88,0x44,0x0c,0x98,0x48,0x8d,0x41,0xff,0x49,
    0xc1,0xe8,0x04,0x75,0xdd,0x41,0xb8,0x17,0x00,0x00,0x00,0x41,0x29,0xc8,0x45,0x85,0xc0,0x7f,0x08,0x45,
    0x01,0xc1,0xe9,0x3d,0xfc,0xff,0xff,0x48,0x63,0xc9,0x48,0x8d,0x74,0x0c,0x98,0xeb,0x95,0xc6,0x44,0x24,
    0x97,0x6c,0xb8,0x01,0x00,0x00,0x00,0x0f,0x05,0xe9,0xb2,0xfc,0xff,0xff,0x48,0x63,0xd2,0x89,0xd8,0x4c,
    0x89,0xe7,0x4c,0x89,0xd6,0x0f,0x05,0xe9,0x1e,0xfb,0xff,0xff,0x41,0xb9,0x00,0x00,0x00,0x00,0x44,0x89,
    0xc8,0x5b,0x5d,0x41,0x5c,0x41,0x5d,0x41,0x5e,0x41,0x5f,0xc3,0xf3,0x0f,0x1e,0xfa,0x48,0x89,0xfe,0x80,
    0x3f,0x00,0x74,0x40,0xba,0x01,0x00,0x00,0x00,0x48,0x89,0xd0,0x48,0x83,0xc2,0x01,0x80,0x7c,0x16,0xff,
    0x00,0x75,0xf2,0x41,0x89,0xc0,0x41,0xb9,0x01,0x00,0x00,0x00,0xbf,0x01,0x00,0x00,0x00,0x49,0x63,0xd0,
    0x44,0x89,0xc8,0x0f,0x05,0xc6,0x44,0x24,0xff,0x0a,0x48,0x8d,0x74,0x24,0xff,0x44,0x89,0xc8,0x48,0x89,
    0xfa,0x0f,0x05,0x41,0x8d,0x40,0x01,0xc3,0x41,0xb8,0x00,0x00,0x00,0x00,0xeb,0xce,
];
const BUILTIN_PUTS_OFFSET_X86_64: u64 = 0x598;

/// Generate a builtin printf/puts implementation for the given architecture.
/// Returns (code_bytes, relocations, symbols) like generate_builtin_start.
fn generate_builtin_printf(
    arch: crate::driver::target::Architecture,
) -> Option<(Vec<u8>, Vec<codegen::Relocation>, Vec<codegen::Symbol>)> {
    use crate::driver::target::Architecture;
    match arch {
        Architecture::X86_64 => {
            let code = BUILTIN_PRINTF_X86_64.to_vec();
            let total_size = code.len() as u64;
            let syms = vec![
                codegen::Symbol {
                    name: String::from("printf"),
                    offset: 0,
                    size: BUILTIN_PUTS_OFFSET_X86_64,
                    binding: codegen::SymbolBinding::Global,
                    symbol_type: codegen::SymbolType::Function,
                    section_index: 0,
                    visibility: codegen::SymbolVisibility::Default,
                    is_definition: true,
                },
                codegen::Symbol {
                    name: String::from("puts"),
                    offset: BUILTIN_PUTS_OFFSET_X86_64,
                    size: total_size - BUILTIN_PUTS_OFFSET_X86_64,
                    binding: codegen::SymbolBinding::Global,
                    symbol_type: codegen::SymbolType::Function,
                    section_index: 0,
                    visibility: codegen::SymbolVisibility::Default,
                    is_definition: true,
                },
            ];
            Some((code, Vec::new(), syms))
        }
        _ => None,
    }
}

/// `crtn.o`) with a compact stub that calls `main` and exits via syscall,
/// enabling standalone executables that need no libc.
///
/// Returns `(text_bytes, relocations, symbols)`.
fn generate_builtin_start(
    arch: crate::driver::target::Architecture,
) -> (
    Vec<u8>,
    Vec<codegen::Relocation>,
    Vec<codegen::Symbol>,
) {
    use crate::driver::target::Architecture;

    match arch {
        Architecture::X86_64 => {
            // x86_64 _start stub:
            //   xor  ebp, ebp          ; 31 ed
            //   mov  rdi, [rsp]        ; 48 8b 3c 24        (argc)
            //   lea  rsi, [rsp+8]      ; 48 8d 74 24 08     (argv)
            //   and  rsp, -16          ; 48 83 e4 f0         (align)
            //   call main              ; e8 XX XX XX XX      (rel32 reloc)
            //   mov  edi, eax          ; 89 c7               (exit code)
            //   mov  eax, 60           ; b8 3c 00 00 00      (sys_exit)
            //   syscall                ; 0f 05
            //   hlt                    ; f4
            let code: Vec<u8> = vec![
                0x31, 0xED,                         // xor ebp, ebp
                0x48, 0x8B, 0x3C, 0x24,             // mov rdi, [rsp]
                0x48, 0x8D, 0x74, 0x24, 0x08,       // lea rsi, [rsp+8]
                0x48, 0x83, 0xE4, 0xF0,             // and rsp, -16
                0xE8, 0x00, 0x00, 0x00, 0x00,       // call rel32 (patched)
                0x89, 0xC7,                         // mov edi, eax
                0xB8, 0x3C, 0x00, 0x00, 0x00,       // mov eax, 60
                0x0F, 0x05,                         // syscall
                0xF4,                               // hlt
            ];
            let relocs = vec![codegen::Relocation {
                offset: 16,  // offset of the rel32 in `call main`
                symbol: String::from("main"),
                reloc_type: codegen::RelocationType::X86_64_PLT32,
                addend: -4,
                section_index: 0,
            }];
            let syms = vec![codegen::Symbol {
                name: String::from("_start"),
                offset: 0,
                size: code.len() as u64,
                binding: codegen::SymbolBinding::Global,
                symbol_type: codegen::SymbolType::Function,
                section_index: 0,
                visibility: codegen::SymbolVisibility::Default,
                is_definition: true,
            }];
            (code, relocs, syms)
        }
        Architecture::I686 => {
            // i686 _start stub:
            //   xor  ebp, ebp          ; 31 ed
            //   mov  ecx, [esp]        ; 8b 0c 24            (argc)
            //   lea  edx, [esp+4]      ; 8d 54 24 04         (argv)
            //   and  esp, -16          ; 83 e4 f0             (align)
            //   push edx               ; 52                   (push argv)
            //   push ecx               ; 51                   (push argc)
            //   call main              ; e8 XX XX XX XX
            //   mov  ebx, eax          ; 89 c3               (exit code)
            //   mov  eax, 1            ; b8 01 00 00 00       (sys_exit)
            //   int  0x80              ; cd 80
            //   hlt                    ; f4
            let code: Vec<u8> = vec![
                0x31, 0xED,                         // xor ebp, ebp
                0x8B, 0x0C, 0x24,                   // mov ecx, [esp]
                0x8D, 0x54, 0x24, 0x04,             // lea edx, [esp+4]
                0x83, 0xE4, 0xF0,                   // and esp, -16
                0x52,                               // push edx
                0x51,                               // push ecx
                0xE8, 0x00, 0x00, 0x00, 0x00,       // call rel32
                0x89, 0xC3,                         // mov ebx, eax
                0xB8, 0x01, 0x00, 0x00, 0x00,       // mov eax, 1
                0xCD, 0x80,                         // int 0x80
                0xF4,                               // hlt
            ];
            let relocs = vec![codegen::Relocation {
                offset: 15,
                symbol: String::from("main"),
                reloc_type: codegen::RelocationType::I386_PC32,
                addend: -4,
                section_index: 0,
            }];
            let syms = vec![codegen::Symbol {
                name: String::from("_start"),
                offset: 0,
                size: code.len() as u64,
                binding: codegen::SymbolBinding::Global,
                symbol_type: codegen::SymbolType::Function,
                section_index: 0,
                visibility: codegen::SymbolVisibility::Default,
                is_definition: true,
            }];
            (code, relocs, syms)
        }
        Architecture::Aarch64 => {
            // AArch64 _start stub:
            //   ldr  x0, [sp]          ; argc
            //   add  x1, sp, #8        ; argv
            //   and  sp, sp, #-16      ; align
            //   bl   main              ; needs relocation
            //   mov  x8, #93           ; sys_exit
            //   svc  #0                ; syscall
            let code: Vec<u8> = vec![
                0xE0, 0x03, 0x40, 0xF9, // ldr  x0, [sp]
                0xE1, 0x23, 0x00, 0x91, // add  x1, sp, #8
                0xFF, 0x43, 0xC0, 0x92, // and  sp, sp, #~0xf
                0x00, 0x00, 0x00, 0x94, // bl   <main> (R_AARCH64_CALL26)
                0xA8, 0x0B, 0x80, 0xD2, // mov  x8, #93
                0x01, 0x00, 0x00, 0xD4, // svc  #0
            ];
            let relocs = vec![codegen::Relocation {
                offset: 12,
                symbol: String::from("main"),
                reloc_type: codegen::RelocationType::Aarch64_CALL26,
                addend: 0,
                section_index: 0,
            }];
            let syms = vec![codegen::Symbol {
                name: String::from("_start"),
                offset: 0,
                size: code.len() as u64,
                binding: codegen::SymbolBinding::Global,
                symbol_type: codegen::SymbolType::Function,
                section_index: 0,
                visibility: codegen::SymbolVisibility::Default,
                is_definition: true,
            }];
            (code, relocs, syms)
        }
        Architecture::Riscv64 => {
            // RISC-V 64 _start stub:
            //   ld   a0, 0(sp)         ; argc
            //   addi a1, sp, 8         ; argv
            //   andi sp, sp, -16       ; align
            //   call main              ; auipc+jalr (2 instructions)
            //   li   a7, 93            ; sys_exit
            //   ecall
            let code: Vec<u8> = vec![
                0x03, 0x35, 0x01, 0x00, // ld  a0, 0(sp)
                0x93, 0x05, 0x81, 0x00, // addi a1, sp, 8
                0x13, 0x71, 0x01, 0xFF, // andi sp, sp, -16
                0x97, 0x00, 0x00, 0x00, // auipc ra, 0 (R_RISCV_CALL_PLT)
                0xE7, 0x80, 0x00, 0x00, // jalr  ra, ra, 0
                0x93, 0x08, 0xD0, 0x05, // li   a7, 93
                0x73, 0x00, 0x00, 0x00, // ecall
            ];
            let relocs = vec![codegen::Relocation {
                offset: 12,
                symbol: String::from("main"),
                reloc_type: codegen::RelocationType::Riscv_Call,
                addend: 0,
                section_index: 0,
            }];
            let syms = vec![codegen::Symbol {
                name: String::from("_start"),
                offset: 0,
                size: code.len() as u64,
                binding: codegen::SymbolBinding::Global,
                symbol_type: codegen::SymbolType::Function,
                section_index: 0,
                visibility: codegen::SymbolVisibility::Default,
                is_definition: true,
            }];
            (code, relocs, syms)
        }
    }
}

// ============================================================================
// Static executable linking — full link with CRT and symbol resolution
// ============================================================================

/// Link a static executable: resolve all symbols, apply relocations, and
/// emit a fully linked ELF binary.
///
/// If the user specified `-lc` (or any `-l` flags), we use system CRT
/// objects (`crt1.o`, `crti.o`, `crtn.o`) and link the requested
/// libraries.  Otherwise we emit a **built-in `_start` stub** that calls
/// `main` and exits via a syscall, producing compact standalone binaries
/// that require no libc.
fn link_static_executable(
    input: LinkerInput,
    config: &LinkerConfig,
    diag: &mut DiagnosticEmitter,
    is_64bit: bool,
    machine: u16,
) -> Result<Vec<u8>, LinkerError> {
    // Determine whether to link system CRT objects (crt1.o, crti.o, crtn.o).
    //
    // Two separate decisions are made:
    //  1. **Full system CRT** (crt1.o + crti.o + crtn.o + libc): used when
    //     the user explicitly requests libraries via `-l` flags.
    //  2. **Partial CRT** (crti.o + crtn.o only): used when the program
    //     defines `main()` but no libraries are requested.  This gives us
    //     the `.init` / `.fini` sections that standard toolchains expect,
    //     while still using our lightweight built-in `_start` stub (which
    //     doesn't depend on libc's `__libc_start_main`).
    let has_main = input.objects.iter().any(|obj| {
        obj.symbols.iter().any(|s| {
            s.name == "main"
                && s.binding == codegen::SymbolBinding::Global
        })
    });
    let use_full_system_crt = !config.libraries.is_empty();
    let use_partial_crt = !use_full_system_crt && has_main;

    // Step 1: Find CRT objects for this architecture.
    let crt = if use_full_system_crt || use_partial_crt {
        match find_crt_objects(config) {
            Ok(mut c) => {
                // Partial CRT: drop crt1.o (it needs __libc_start_main from
                // libc) — our builtin _start will call main() directly.
                if use_partial_crt {
                    c.crt1 = None;
                }
                c
            }
            Err(_) => CrtObjects { crt1: None, crti: None, crtn: None },
        }
    } else {
        CrtObjects { crt1: None, crti: None, crtn: None }
    };

    // Step 2: Resolve libraries specified via -l flags.
    let libraries = if use_full_system_crt {
        resolve_libraries(config)?
    } else {
        Vec::new()
    };

    // Step 2b: Generate built-in _start stub if not using full system CRT
    // (i.e. no libc, so we need our own _start that calls main() directly).
    let builtin_start = if !use_full_system_crt {
        Some(generate_builtin_start(config.target.arch))
    } else {
        None
    };

    // Step 3: Parse CRT objects into ELF objects for section/symbol extraction.
    let mut crt_elf_objects: Vec<elf::ElfObject> = Vec::new();
    if let Some(ref data) = crt.crti {
        match elf::ElfObject::parse(data) {
            Ok(obj) => crt_elf_objects.push(obj),
            Err(e) => {
                diag.error_no_loc(&format!("failed to parse crti.o: {}", e));
            }
        }
    }
    if let Some(ref data) = crt.crt1 {
        match elf::ElfObject::parse(data) {
            Ok(obj) => crt_elf_objects.push(obj),
            Err(e) => {
                diag.error_no_loc(&format!("failed to parse crt1.o: {}", e));
            }
        }
    }

    // Step 4: Collect all sections from CRT and input objects.
    let mut all_sections: Vec<sections::InputSection> = Vec::new();
    let mut object_count: usize = 0;

    // CRT sections come first (crti, crt1 provide _start and init/fini).
    for crt_obj in &crt_elf_objects {
        for (sec_idx, sec) in crt_obj.sections.iter().enumerate() {
            // Skip non-content sections (NULL, STRTAB, SYMTAB, RELA, REL).
            if sec.section_type == elf::SHT_NULL
                || sec.section_type == elf::SHT_STRTAB
                || sec.section_type == elf::SHT_SYMTAB
                || sec.section_type == elf::SHT_RELA
                || sec.section_type == elf::SHT_REL
            {
                continue;
            }
            all_sections.push(sections::InputSection {
                name: sec.name.clone(),
                data: sec.data.clone(),
                section_type: sec.section_type,
                flags: sec.flags,
                alignment: sec.alignment,
                mem_size: sec.data.len() as u64,
                object_index: object_count,
                original_index: sec_idx,
            });
        }
        object_count += 1;
    }

    // Built-in _start stub (when not using system CRT).
    let builtin_start_obj_idx = if let Some(ref start) = builtin_start {
        let idx = object_count;
        let (ref text, ref _relocs, ref _syms) = *start;
        all_sections.push(sections::InputSection {
            name: String::from(".text"),
            data: text.clone(),
            section_type: elf::SHT_PROGBITS,
            flags: elf::SHF_ALLOC | elf::SHF_EXECINSTR,
            alignment: 16,
            mem_size: text.len() as u64,
            object_index: idx,
            original_index: 0,
        });
        object_count += 1;
        Some(idx)
    } else {
        None
    };

    // User input objects.
    let user_object_base = object_count;
    for (obj_idx, obj) in input.objects.iter().enumerate() {
        let global_obj_idx = user_object_base + obj_idx;
        for (sec_idx, sec) in obj.sections.iter().enumerate() {
            all_sections.push(codegen_section_to_input(sec, global_obj_idx, sec_idx));
        }
        object_count += 1;
    }

    // crtn sections come last (provides .init/.fini epilogue).
    if let Some(ref data) = crt.crtn {
        if let Ok(crtn_obj) = elf::ElfObject::parse(data) {
            for (sec_idx, sec) in crtn_obj.sections.iter().enumerate() {
                if sec.section_type == elf::SHT_NULL
                    || sec.section_type == elf::SHT_STRTAB
                    || sec.section_type == elf::SHT_SYMTAB
                    || sec.section_type == elf::SHT_RELA
                    || sec.section_type == elf::SHT_REL
                {
                    continue;
                }
                all_sections.push(sections::InputSection {
                    name: sec.name.clone(),
                    data: sec.data.clone(),
                    section_type: sec.section_type,
                    flags: sec.flags,
                    alignment: sec.alignment,
                    mem_size: sec.data.len() as u64,
                    object_index: object_count,
                    original_index: sec_idx,
                });
            }
            object_count += 1;
        }
    }

    // Step 5: Collect all symbols from CRT objects and user objects.
    let mut resolver = symbols::SymbolResolver::new();

    // CRT symbols.
    let mut crt_obj_idx = 0;
    for crt_obj in &crt_elf_objects {
        let syms: Vec<symbols::Symbol> = crt_obj
            .symbols
            .iter()
            .map(|ps| symbols::Symbol::from_parsed(ps, crt_obj_idx))
            .collect();
        resolver.add_object_symbols(crt_obj_idx, &syms);
        crt_obj_idx += 1;
    }

    // Built-in _start stub symbols (when not using system CRT).
    if let (Some(idx), Some(ref start)) = (builtin_start_obj_idx, &builtin_start) {
        let (ref _text, ref _relocs, ref syms_codegen) = *start;
        let syms: Vec<symbols::Symbol> = syms_codegen
            .iter()
            .map(|s| codegen_symbol_to_linker(s, idx))
            .collect();
        resolver.add_object_symbols(idx, &syms);
    }

    // User object symbols.
    for (obj_idx, obj) in input.objects.iter().enumerate() {
        let global_obj_idx = user_object_base + obj_idx;
        let syms: Vec<symbols::Symbol> = obj
            .symbols
            .iter()
            .map(|s| codegen_symbol_to_linker(s, global_obj_idx))
            .collect();
        resolver.add_object_symbols(global_obj_idx, &syms);
    }

    // crtn symbols (if crtn was parsed).
    if let Some(ref data) = crt.crtn {
        if let Ok(crtn_obj) = elf::ElfObject::parse(data) {
            let crtn_idx = object_count - 1;
            let syms: Vec<symbols::Symbol> = crtn_obj
                .symbols
                .iter()
                .map(|ps| symbols::Symbol::from_parsed(ps, crtn_idx))
                .collect();
            resolver.add_object_symbols(crtn_idx, &syms);
        }
    }

    // Step 5b: Inject builtin printf/puts BEFORE archive processing.
    // This ensures our minimal write(2)-based printf/puts are defined
    // before the linker tries to extract glibc's printf from libc.a,
    // which would pull in hundreds of transitive dependencies.
    let builtin_printf_obj_idx: Option<usize> = {
        let undef = resolver.undefined_symbols();
        let needs_printf = undef.iter().any(|s| *s == "printf" || *s == "puts");
        if needs_printf {
            if let Some((text, _relocs, syms)) = generate_builtin_printf(config.target.arch) {
                let idx = object_count;
                all_sections.push(sections::InputSection {
                    name: String::from(".text"),
                    data: text,
                    section_type: elf::SHT_PROGBITS,
                    flags: elf::SHF_ALLOC | elf::SHF_EXECINSTR,
                    alignment: 16,
                    mem_size: BUILTIN_PRINTF_X86_64.len() as u64,
                    object_index: idx,
                    original_index: 0,
                });
                let linker_syms: Vec<symbols::Symbol> = syms
                    .iter()
                    .map(|s| codegen_symbol_to_linker(s, idx))
                    .collect();
                resolver.add_object_symbols(idx, &linker_syms);
                object_count += 1;
                Some(idx)
            } else {
                None
            }
        } else {
            None
        }
    };

    // Step 6: Process library archives to resolve remaining undefined symbols.
    let mut archives: Vec<archive::Archive> = Vec::new();
    for lib in &libraries {
        match archive::Archive::parse(lib.data.clone()) {
            Ok(ar) => archives.push(ar),
            Err(e) => {
                diag.error_no_loc(&format!(
                    "failed to parse library {}: {}",
                    lib.path.display(),
                    e
                ));
            }
        }
    }

    let extracted = resolver.process_archives(&archives)?;
    // Save base object index for archive members (needed for relocation
    // processing later).
    let archive_member_base = object_count;
    // Add sections and symbols from extracted archive members.
    for member in &extracted {
        if let Ok(member_obj) = elf::ElfObject::parse(&member.data) {
            for (sec_idx, sec) in member_obj.sections.iter().enumerate() {
                if sec.section_type == elf::SHT_NULL
                    || sec.section_type == elf::SHT_STRTAB
                    || sec.section_type == elf::SHT_SYMTAB
                    || sec.section_type == elf::SHT_RELA
                    || sec.section_type == elf::SHT_REL
                {
                    continue;
                }
                all_sections.push(sections::InputSection {
                    name: sec.name.clone(),
                    data: sec.data.clone(),
                    section_type: sec.section_type,
                    flags: sec.flags,
                    alignment: sec.alignment,
                    mem_size: sec.data.len() as u64,
                    object_index: object_count,
                    original_index: sec_idx,
                });
            }
            object_count += 1;
        }
    }

    // Step 7: Resolve all symbols — every reference must have a definition.
    if let Err(e) = resolver.resolve() {
        let err = LinkerError::from(e);
        diag.error_no_loc(&format!("{}", err));
        return Err(err);
    }

    // Step 8: Merge sections from all input objects.
    let section_order = default_section_order();
    let mut merged = sections::merge_sections(all_sections, &section_order);

    // Step 9: Compute section addresses and file offsets.
    let base_address = script::default_base_address(&config.target);
    let class = if is_64bit { elf::ELFCLASS64 } else { elf::ELFCLASS32 };
    let ehdr_sz = elf::ehdr_size(class) as u64;
    let phdr_sz = elf::phdr_size(class) as u64;

    // Count actual program headers by scanning which permission categories
    // have allocatable sections. This MUST match what build_program_headers()
    // will produce, otherwise the header size estimate causes section file
    // offsets to diverge from the ELF writer's actual placement, creating
    // LOAD segment mismatches that crash i686 and RISC-V binaries.
    let mut has_rx = false;
    let mut has_ro = false;
    let mut has_rw = false;
    for sec in &merged {
        if sec.flags & elf::SHF_ALLOC == 0 { continue; }
        let is_exec = sec.flags & elf::SHF_EXECINSTR != 0;
        let is_write = sec.flags & elf::SHF_WRITE != 0;
        if is_exec { has_rx = true; }
        else if is_write { has_rw = true; }
        else { has_ro = true; }
    }
    let phdr_count: u64 = (has_rx as u64) + (has_ro as u64) + (has_rw as u64) + 1; // +1 for GNU_STACK
    let header_size = ehdr_sz + phdr_count * phdr_sz;

    let _layout = sections::compute_layout(&mut merged, base_address, header_size, is_64bit);

    // Step 10: Build section address and mapping tables used for both
    // symbol address assignment and relocation processing.
    let section_addresses: Vec<u64> = merged.iter().map(|s| s.virtual_address).collect();
    let section_mappings: Vec<sections::InputSectionMapping> = merged
        .iter()
        .flat_map(|s| s.input_mappings.iter().cloned())
        .collect();

    // Build mapping from (object_index, original_section_index) to
    // (merged_section_index, output_offset_within_merged).
    //
    // This remapping is essential because:
    // 1. CRT objects prepend sections (e.g., .init, .fini) that shift merged
    //    indices relative to codegen's 0-based section array indices.
    // 2. Section merging concatenates multiple objects' identically-named
    //    sections, so relocation offsets must be adjusted by the input
    //    section's starting position within the merged section.
    let mut reloc_section_map: std::collections::HashMap<(usize, usize), (usize, u64)> =
        std::collections::HashMap::new();
    let mut merged_index_map: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for (merged_idx, merged_sec) in merged.iter().enumerate() {
        for mapping in &merged_sec.input_mappings {
            reloc_section_map.insert(
                (mapping.object_index, mapping.original_index),
                (merged_idx, mapping.output_offset),
            );
            merged_index_map.insert(
                (mapping.object_index, mapping.original_index),
                merged_idx,
            );
        }
    }

    // Assign final addresses to resolved symbols using the merged section
    // index mapping for correct section-to-address translation.
    resolver.assign_addresses(&section_addresses, &section_mappings, &merged_index_map);

    // Step 11: Apply relocations — patch machine code with final addresses.
    let resolved_symbols = resolver.all_resolved();
    let mut resolved_vec: Vec<symbols::ResolvedSymbol> =
        resolved_symbols.into_iter().cloned().collect();

    // Add LOCAL symbols (e.g., string constants like .str.0, static functions)
    // to the resolved vector so that relocations referencing them can be resolved.
    // Local symbols are not processed by resolve() but still need addresses.
    //
    // CRITICAL: We must NOT skip duplicates for local symbols.  Multiple
    // objects can define identically-named local (static) symbols, and each
    // object's relocations must resolve to its OWN definition.  We tag each
    // local entry with `source_object` so the relocation lookup can prefer
    // the correct per-object definition.
    for (obj_idx, obj) in input.objects.iter().enumerate() {
        let global_obj_idx = user_object_base + obj_idx;
        for sym in &obj.symbols {
            if sym.binding == codegen::SymbolBinding::Local && !sym.name.is_empty() {
                // Compute final address for this local symbol
                let sec_idx = sym.section_index;
                if let Some(&(merged_idx, output_offset)) =
                    reloc_section_map.get(&(global_obj_idx, sec_idx))
                {
                    let addr = if merged_idx < section_addresses.len() {
                        section_addresses[merged_idx] + output_offset + sym.offset
                    } else {
                        output_offset + sym.offset
                    };
                    resolved_vec.push(symbols::ResolvedSymbol {
                        name: sym.name.clone(),
                        address: addr,
                        size: sym.size,
                        binding: symbols::SymbolBinding::Local,
                        symbol_type: symbols::SymbolType::Object,
                        visibility: symbols::SymbolVisibility::Default,
                        output_section_index: merged_idx,
                        is_definition: true,
                        source_object: Some(global_obj_idx),
                    });
                }
            }
        }
    }

    let mut section_data: Vec<Vec<u8>> = merged.iter().map(|s| s.data.clone()).collect();

    // Build relocation entries from ALL input objects (CRT + codegen + crtn),
    // remapping section indices and adjusting offsets to account for section
    // merging.
    let mut reloc_entries: Vec<relocations::RelocationEntry> = Vec::new();

    // Process CRT object relocations (crti.o, crt1.o).
    // CRT objects were assigned object indices 0..crt_elf_objects.len()-1.
    for (crt_idx, crt_obj) in crt_elf_objects.iter().enumerate() {
        for reloc in &crt_obj.relocations {
            // Look up the symbol by index in the CRT object's symbol table.
            let sym_name = if (reloc.symbol_index as usize) < crt_obj.symbols.len() {
                &crt_obj.symbols[reloc.symbol_index as usize].name
            } else {
                continue;
            };
            let sym_idx = resolved_vec
                .iter()
                .position(|rs| rs.name == *sym_name)
                .unwrap_or(0) as u32;

            if let Some(&(merged_idx, output_offset)) =
                reloc_section_map.get(&(crt_idx, reloc.section_index))
            {
                reloc_entries.push(relocations::RelocationEntry {
                    offset: reloc.offset + output_offset,
                    reloc_type: reloc.reloc_type,
                    symbol_index: sym_idx,
                    addend: reloc.addend,
                    section_index: merged_idx,
                });
            }
        }
    }

    // Process built-in _start stub relocations (when not using system CRT).
    if let (Some(idx), Some(ref start)) = (builtin_start_obj_idx, &builtin_start) {
        let (ref _text, ref relocs, ref _syms) = *start;
        for reloc in relocs {
            let sym_idx = resolved_vec
                .iter()
                .position(|rs| rs.name == reloc.symbol)
                .unwrap_or(0) as u32;

            if let Some(&(merged_idx, output_offset)) =
                reloc_section_map.get(&(idx, reloc.section_index))
            {
                reloc_entries.push(relocations::RelocationEntry {
                    offset: reloc.offset as u64 + output_offset,
                    reloc_type: reloc_type_to_elf(&reloc.reloc_type),
                    symbol_index: sym_idx,
                    addend: reloc.addend,
                    section_index: merged_idx,
                });
            }
        }
    }

    // Process codegen object relocations.
    //
    // For each relocation, we first look for a LOCAL symbol from the SAME
    // source object (matching source_object == global_obj_idx).  This
    // ensures `static` functions / variables resolve to the defining
    // object's own copy, not an identically-named local from another
    // translation unit.  If no same-object local is found, fall back to
    // global name lookup (the previous behaviour).
    for (obj_idx, obj) in input.objects.iter().enumerate() {
        let global_obj_idx = user_object_base + obj_idx;
        for reloc in &obj.relocations {
            // Prefer same-object local symbol, then any symbol by name.
            let sym_idx = resolved_vec
                .iter()
                .position(|rs| {
                    rs.name == reloc.symbol
                        && rs.source_object == Some(global_obj_idx)
                })
                .or_else(|| {
                    resolved_vec
                        .iter()
                        .position(|rs| rs.name == reloc.symbol)
                })
                .unwrap_or(0) as u32;

            // Map the codegen section index to the merged section index and
            // adjust the offset by the input section's position within the
            // merged output section.
            if let Some(&(merged_idx, output_offset)) =
                reloc_section_map.get(&(global_obj_idx, reloc.section_index))
            {
                reloc_entries.push(relocations::RelocationEntry {
                    offset: reloc.offset + output_offset,
                    reloc_type: reloc_type_to_elf(&reloc.reloc_type),
                    symbol_index: sym_idx,
                    addend: reloc.addend,
                    section_index: merged_idx,
                });
            }
        }
    }

    // Process crtn.o relocations (if present).
    if let Some(ref crtn_data) = crt.crtn {
        if let Ok(crtn_obj) = elf::ElfObject::parse(crtn_data) {
            let crtn_idx = object_count - 1;
            for reloc in &crtn_obj.relocations {
                let sym_name = if (reloc.symbol_index as usize) < crtn_obj.symbols.len() {
                    &crtn_obj.symbols[reloc.symbol_index as usize].name
                } else {
                    continue;
                };
                let sym_idx = resolved_vec
                    .iter()
                    .position(|rs| rs.name == *sym_name)
                    .unwrap_or(0) as u32;

                if let Some(&(merged_idx, output_offset)) =
                    reloc_section_map.get(&(crtn_idx, reloc.section_index))
                {
                    reloc_entries.push(relocations::RelocationEntry {
                        offset: reloc.offset + output_offset,
                        reloc_type: reloc.reloc_type,
                        symbol_index: sym_idx,
                        addend: reloc.addend,
                        section_index: merged_idx,
                    });
                }
            }
        }
    }

    let reloc_ctx = relocations::RelocationContext {
        symbols: &resolved_vec,
        section_addresses: &section_addresses,
        got_address: 0,
        plt_address: 0,
        arch: config.target.arch,
        is_pic: false,
    };

    relocations::apply_relocations(&mut section_data, &reloc_entries, &reloc_ctx)?;

    // Update merged section data with relocated bytes.
    for (i, data) in section_data.into_iter().enumerate() {
        if i < merged.len() {
            merged[i].data = data;
        }
    }

    // Step 12: Build symbol table for the output.
    let (symtab_data, strtab_data) = resolver.generate_symtab(is_64bit);

    // Step 13: Find entry point address.
    let entry_addr = resolver
        .entry_point_address(&config.entry_point)
        .unwrap_or(base_address);

    // Step 14: Emit the final ELF binary.
    build_elf_output(
        &merged,
        &symtab_data,
        &strtab_data,
        input.debug_sections.as_ref(),
        is_64bit,
        machine,
        elf::ET_EXEC,
        entry_addr,
        base_address,
    )
}

// ============================================================================
// Shared library linking
// ============================================================================

/// Link a shared library: generate dynamic sections, apply relocations for
/// position-independent code, and emit an ET_DYN ELF binary.
fn link_shared_library(
    input: LinkerInput,
    config: &LinkerConfig,
    diag: &mut DiagnosticEmitter,
    is_64bit: bool,
    machine: u16,
) -> Result<Vec<u8>, LinkerError> {
    // Step 1: Resolve libraries specified via -l flags.
    let libraries = resolve_libraries(config)?;

    // Step 2: Collect all sections from input objects.
    let mut all_sections: Vec<sections::InputSection> = Vec::new();
    let mut object_count: usize = 0;

    for (obj_idx, obj) in input.objects.iter().enumerate() {
        for (sec_idx, sec) in obj.sections.iter().enumerate() {
            all_sections.push(codegen_section_to_input(sec, obj_idx, sec_idx));
        }
        object_count += 1;
    }

    // Step 3: Collect symbols from all input objects.
    let mut resolver = symbols::SymbolResolver::new();
    for (obj_idx, obj) in input.objects.iter().enumerate() {
        let syms: Vec<symbols::Symbol> = obj
            .symbols
            .iter()
            .map(|s| codegen_symbol_to_linker(s, obj_idx))
            .collect();
        resolver.add_object_symbols(obj_idx, &syms);
    }

    // Step 4: Process library archives.
    let mut archives: Vec<archive::Archive> = Vec::new();
    for lib in &libraries {
        match archive::Archive::parse(lib.data.clone()) {
            Ok(ar) => archives.push(ar),
            Err(e) => {
                diag.error_no_loc(&format!(
                    "failed to parse library {}: {}",
                    lib.path.display(),
                    e
                ));
            }
        }
    }

    let extracted = resolver.process_archives(&archives)?;
    let archive_member_base = object_count;
    for member in &extracted {
        if let Ok(member_obj) = elf::ElfObject::parse(&member.data) {
            for (sec_idx, sec) in member_obj.sections.iter().enumerate() {
                if sec.section_type == elf::SHT_NULL
                    || sec.section_type == elf::SHT_STRTAB
                    || sec.section_type == elf::SHT_SYMTAB
                    || sec.section_type == elf::SHT_RELA
                    || sec.section_type == elf::SHT_REL
                {
                    continue;
                }
                all_sections.push(sections::InputSection {
                    name: sec.name.clone(),
                    data: sec.data.clone(),
                    section_type: sec.section_type,
                    flags: sec.flags,
                    alignment: sec.alignment,
                    mem_size: sec.data.len() as u64,
                    object_index: object_count,
                    original_index: sec_idx,
                });
            }
            object_count += 1;
        }
    }

    // Step 5: Resolve symbols. Shared libraries are more permissive — undefined
    // symbols may be resolved at runtime by the dynamic linker, so we allow
    // resolution to partially fail.
    let _ = resolver.resolve();

    // Step 6: Merge sections.
    let section_order = default_section_order();
    let mut merged = sections::merge_sections(all_sections, &section_order);

    // Step 7: Compute layout. Shared libraries use base address 0 (PIE).
    let base_address: u64 = 0;
    let class = if is_64bit { elf::ELFCLASS64 } else { elf::ELFCLASS32 };
    let ehdr_sz = elf::ehdr_size(class) as u64;
    let phdr_sz = elf::phdr_size(class) as u64;
    // Count actual segments for shared libraries (same logic as executable + PT_DYNAMIC).
    let mut has_rx_so = false;
    let mut has_ro_so = false;
    let mut has_rw_so = false;
    for sec in &merged {
        if sec.flags & elf::SHF_ALLOC == 0 { continue; }
        let is_exec = sec.flags & elf::SHF_EXECINSTR != 0;
        let is_write = sec.flags & elf::SHF_WRITE != 0;
        if is_exec { has_rx_so = true; }
        else if is_write { has_rw_so = true; }
        else { has_ro_so = true; }
    }
    let phdr_count_so: u64 = (has_rx_so as u64) + (has_ro_so as u64) + (has_rw_so as u64) + 2; // +1 GNU_STACK, +1 PT_DYNAMIC
    let header_size = ehdr_sz + phdr_count_so * phdr_sz;

    let _layout = sections::compute_layout(&mut merged, base_address, header_size, is_64bit);

    // Step 8: Build section address and mapping tables, then assign addresses.
    let section_addresses: Vec<u64> = merged.iter().map(|s| s.virtual_address).collect();
    let section_mappings: Vec<sections::InputSectionMapping> = merged
        .iter()
        .flat_map(|s| s.input_mappings.iter().cloned())
        .collect();

    // Build mapping from (object_index, original_section_index) to
    // (merged_section_index, output_offset_within_merged) for relocation
    // section index remapping and offset adjustment after section merging.
    let mut reloc_section_map: std::collections::HashMap<(usize, usize), (usize, u64)> =
        std::collections::HashMap::new();
    let mut merged_index_map: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for (merged_idx, merged_sec) in merged.iter().enumerate() {
        for mapping in &merged_sec.input_mappings {
            reloc_section_map.insert(
                (mapping.object_index, mapping.original_index),
                (merged_idx, mapping.output_offset),
            );
            merged_index_map.insert(
                (mapping.object_index, mapping.original_index),
                merged_idx,
            );
        }
    }

    resolver.assign_addresses(&section_addresses, &section_mappings, &merged_index_map);

    // Step 9: Apply relocations with PIC mode enabled.
    let resolved_symbols = resolver.all_resolved();
    let resolved_vec: Vec<symbols::ResolvedSymbol> =
        resolved_symbols.into_iter().cloned().collect();
    let mut section_data: Vec<Vec<u8>> = merged.iter().map(|s| s.data.clone()).collect();

    let mut reloc_entries: Vec<relocations::RelocationEntry> = Vec::new();
    for (obj_idx, obj) in input.objects.iter().enumerate() {
        for reloc in &obj.relocations {
            let sym_idx = resolved_vec
                .iter()
                .position(|rs| rs.name == reloc.symbol)
                .unwrap_or(0) as u32;

            // Map the codegen section index to the merged section index and
            // adjust the offset by the input section's position in the
            // merged output section.
            if let Some(&(merged_idx, output_offset)) =
                reloc_section_map.get(&(obj_idx, reloc.section_index))
            {
                reloc_entries.push(relocations::RelocationEntry {
                    offset: reloc.offset + output_offset,
                    reloc_type: reloc_type_to_elf(&reloc.reloc_type),
                    symbol_index: sym_idx,
                    addend: reloc.addend,
                    section_index: merged_idx,
                });
            }
        }
    }

    // Process extracted archive member relocations (e.g. libc.a).
    // Each extracted member was assigned an object index starting at
    // archive_member_base.  We re-parse the member data to retrieve its
    // relocation and symbol tables, then build relocation entries using
    // the same section-remapping approach as for CRT objects.
    for (member_idx, member) in extracted.iter().enumerate() {
        if let Ok(member_obj) = elf::ElfObject::parse(&member.data) {
            let member_global_idx = archive_member_base + member_idx;
            for reloc in &member_obj.relocations {
                let sym_name = if (reloc.symbol_index as usize) < member_obj.symbols.len() {
                    &member_obj.symbols[reloc.symbol_index as usize].name
                } else {
                    continue;
                };
                let sym_idx = resolved_vec
                    .iter()
                    .position(|rs| rs.name == *sym_name)
                    .unwrap_or(0) as u32;

                if let Some(&(merged_idx, output_offset)) =
                    reloc_section_map.get(&(member_global_idx, reloc.section_index))
                {
                    reloc_entries.push(relocations::RelocationEntry {
                        offset: reloc.offset + output_offset,
                        reloc_type: reloc.reloc_type,
                        symbol_index: sym_idx,
                        addend: reloc.addend,
                        section_index: merged_idx,
                    });
                }
            }
        }
    }

    let reloc_ctx = relocations::RelocationContext {
        symbols: &resolved_vec,
        section_addresses: &section_addresses,
        got_address: 0,
        plt_address: 0,
        arch: config.target.arch,
        is_pic: true, // Shared libraries always use PIC.
    };

    relocations::apply_relocations(&mut section_data, &reloc_entries, &reloc_ctx)?;

    for (i, data) in section_data.into_iter().enumerate() {
        if i < merged.len() {
            merged[i].data = data;
        }
    }

    // Step 10: Build symbol table for output.
    let (symtab_data, strtab_data) = resolver.generate_symtab(is_64bit);

    // Step 10b: Generate dynamic sections for the shared library.
    // Collect exported (global) symbols for the .dynsym table.
    let all_resolved = resolver.all_resolved();
    let exported_for_dyn: Vec<_> = all_resolved
        .iter()
        .filter(|s| {
            matches!(
                s.binding,
                symbols::SymbolBinding::Global | symbols::SymbolBinding::Weak
            )
        })
        .cloned()
        .cloned()
        .collect();

    // Derive the soname from the config output path (if it looks like a .so).
    let soname_str = config
        .output_path
        .file_name()
        .and_then(|n| n.to_str())
        .map(String::from);

    let needed_libs: Vec<String> = Vec::new(); // No DT_NEEDED for now

    if let Ok(dyn_out) = dynamic::generate_dynamic_sections(
        &exported_for_dyn,
        &needed_libs,
        soname_str.as_deref(),
        &config.target,
    ) {
        // Add dynamic sections to the merged list so they appear in the ELF.
        // These sections need SHF_ALLOC so the kernel maps them into memory.
        let dynstr_sec = sections::MergedSection {
            name: String::from(".dynstr"),
            data: dyn_out.dynstr.clone(),
            section_type: elf::SHT_STRTAB,
            flags: elf::SHF_ALLOC,
            alignment: 1,
            virtual_address: 0,
            file_offset: 0,
            mem_size: dyn_out.dynstr.len() as u64,
            input_mappings: Vec::new(),
        };
        let dynsym_entsize: u64 = if is_64bit { 24 } else { 16 };
        let dynsym_sec = sections::MergedSection {
            name: String::from(".dynsym"),
            data: dyn_out.dynsym.clone(),
            section_type: elf::SHT_DYNSYM,
            flags: elf::SHF_ALLOC,
            alignment: if is_64bit { 8 } else { 4 },
            virtual_address: 0,
            file_offset: 0,
            mem_size: dyn_out.dynsym.len() as u64,
            input_mappings: Vec::new(),
        };
        let hash_sec = sections::MergedSection {
            name: String::from(".hash"),
            data: dyn_out.hash.clone(),
            section_type: elf::SHT_HASH,
            flags: elf::SHF_ALLOC,
            alignment: 4,
            virtual_address: 0,
            file_offset: 0,
            mem_size: dyn_out.hash.len() as u64,
            input_mappings: Vec::new(),
        };
        let dynamic_sec = sections::MergedSection {
            name: String::from(".dynamic"),
            data: dyn_out.dynamic.clone(),
            section_type: elf::SHT_DYNAMIC,
            flags: elf::SHF_ALLOC | elf::SHF_WRITE,
            alignment: if is_64bit { 8 } else { 4 },
            virtual_address: 0,
            file_offset: 0,
            mem_size: dyn_out.dynamic.len() as u64,
            input_mappings: Vec::new(),
        };

        merged.push(dynstr_sec);
        merged.push(dynsym_sec);
        merged.push(hash_sec);
        merged.push(dynamic_sec);

        // Recompute layout so the new sections get valid virtual addresses
        // and file offsets.
        let _relayout = sections::compute_layout(
            &mut merged,
            base_address,
            header_size,
            is_64bit,
        );
    }

    // Step 11: Emit the final ELF shared object.
    build_elf_output(
        &merged,
        &symtab_data,
        &strtab_data,
        input.debug_sections.as_ref(),
        is_64bit,
        machine,
        elf::ET_DYN,
        0, // Shared libraries have no fixed entry point.
        base_address,
    )
}

// ============================================================================
// ELF output builder — shared between executable and shared library modes
// ============================================================================

/// Build the final ELF binary from merged sections and metadata.
///
/// This is the common output stage used by both `link_static_executable` and
/// `link_shared_library`. It constructs the ELF writer, adds all sections,
/// builds program headers, and serializes the binary.
fn build_elf_output(
    merged: &[sections::MergedSection],
    symtab_data: &[u8],
    strtab_data: &[u8],
    debug_sections: Option<&DebugSections>,
    is_64bit: bool,
    machine: u16,
    elf_type: u16,
    entry: u64,
    _base_address: u64,
) -> Result<Vec<u8>, LinkerError> {
    let mut writer = elf::ElfWriter::new(is_64bit, machine);
    writer.set_type(elf_type);
    writer.set_entry(entry);

    // Add merged output sections (code, data, bss, rodata, etc.).
    for sec in merged {
        // Set sh_entsize for structured sections that require it.
        let entsize = match sec.section_type {
            elf::SHT_DYNSYM => {
                if is_64bit { 24 } else { 16 }
            }
            elf::SHT_DYNAMIC => {
                if is_64bit { 16 } else { 8 }
            }
            elf::SHT_HASH => 4,
            _ => 0,
        };
        // For SHT_NOBITS (.bss) sections, the merged data is empty but
        // mem_size tracks the actual virtual size. Set data to a zero-filled
        // vector so the ELF writer's data.len() reports the correct sh_size.
        let section_data = if sec.section_type == elf::SHT_NOBITS && sec.data.is_empty() && sec.mem_size > 0 {
            vec![0u8; sec.mem_size as usize]
        } else {
            sec.data.clone()
        };
        writer.add_section(elf::OutputSection {
            name: sec.name.clone(),
            data: section_data,
            header: elf::OutputSectionHeader {
                sh_type: sec.section_type,
                sh_flags: sec.flags,
                sh_addr: sec.virtual_address,
                sh_addralign: sec.alignment,
                sh_entsize: entsize,
                sh_link: 0,
                sh_info: 0,
            },
            file_offset: Some(sec.file_offset),
        });
    }

    // Add string table section.
    writer.add_section(elf::OutputSection {
        name: String::from(".strtab"),
        data: strtab_data.to_vec(),
        header: elf::OutputSectionHeader {
            sh_type: elf::SHT_STRTAB,
            sh_flags: 0,
            sh_addr: 0,
            sh_addralign: 1,
            sh_entsize: 0,
            sh_link: 0,
            sh_info: 0,
        },
        file_offset: None,
    });

    // Add symbol table section.
    // sh_link must point to the associated string table section (.strtab).
    // The ELF section header table is: [0]=NULL, [1..N]=merged sections,
    // [N+1]=.strtab, [N+2]=.symtab. So .strtab index = merged.len() + 1.
    let strtab_section_index = (merged.len() + 1) as u32;
    let symtab_entsize = if is_64bit { 24u64 } else { 16u64 };
    writer.add_section(elf::OutputSection {
        name: String::from(".symtab"),
        data: symtab_data.to_vec(),
        header: elf::OutputSectionHeader {
            sh_type: elf::SHT_SYMTAB,
            sh_flags: 0,
            sh_addr: 0,
            sh_addralign: if is_64bit { 8 } else { 4 },
            sh_entsize: symtab_entsize,
            sh_link: strtab_section_index,
            sh_info: 0,
        },
        file_offset: None,
    });

    // Add DWARF debug sections if present.
    if let Some(dbg) = debug_sections {
        add_debug_sections_to_writer(&mut writer, dbg);
    }

    // Build program headers for executable and shared library modes.
    if elf_type == elf::ET_EXEC || elf_type == elf::ET_DYN {
        build_program_headers(&mut writer, merged, is_64bit);
    }

    Ok(writer.write())
}

/// Add DWARF debug sections to the ELF writer.
///
/// Each non-empty debug section is added as a `SHT_PROGBITS` section with
/// appropriate flags. `.debug_str` uses `SHF_MERGE | SHF_STRINGS` flags
/// per the DWARF v4 specification.
fn add_debug_sections_to_writer(writer: &mut elf::ElfWriter, dbg: &DebugSections) {
    let make_debug_section = |name: &str, data: &[u8]| elf::OutputSection {
        name: String::from(name),
        data: data.to_vec(),
        header: elf::OutputSectionHeader {
            sh_type: elf::SHT_PROGBITS,
            sh_flags: 0,
            sh_addr: 0,
            sh_addralign: 1,
            sh_entsize: 0,
            sh_link: 0,
            sh_info: 0,
        },
        file_offset: None,
    };

    if !dbg.debug_info.is_empty() {
        writer.add_section(make_debug_section(".debug_info", &dbg.debug_info));
    }
    if !dbg.debug_abbrev.is_empty() {
        writer.add_section(make_debug_section(".debug_abbrev", &dbg.debug_abbrev));
    }
    if !dbg.debug_line.is_empty() {
        writer.add_section(make_debug_section(".debug_line", &dbg.debug_line));
    }
    if !dbg.debug_str.is_empty() {
        // .debug_str uses SHF_MERGE | SHF_STRINGS per DWARF v4.
        writer.add_section(elf::OutputSection {
            name: String::from(".debug_str"),
            data: dbg.debug_str.clone(),
            header: elf::OutputSectionHeader {
                sh_type: elf::SHT_PROGBITS,
                sh_flags: sections::SHF_MERGE | sections::SHF_STRINGS,
                sh_addr: 0,
                sh_addralign: 1,
                sh_entsize: 1,
                sh_link: 0,
                sh_info: 0,
            },
            file_offset: None,
        });
    }
    if !dbg.debug_aranges.is_empty() {
        writer.add_section(make_debug_section(".debug_aranges", &dbg.debug_aranges));
    }
    if !dbg.debug_frame.is_empty() {
        writer.add_section(make_debug_section(".debug_frame", &dbg.debug_frame));
    }
    if !dbg.debug_loc.is_empty() {
        writer.add_section(make_debug_section(".debug_loc", &dbg.debug_loc));
    }
}

/// Build PT_LOAD program headers based on section permissions.
///
/// Groups allocatable sections into segments based on their permission flags:
/// - **R+X** for executable code (`.text`, `.init`, `.fini`, `.plt`)
/// - **R** for read-only data (`.rodata`, `.eh_frame`)
/// - **R+W** for writable data (`.data`, `.bss`, `.got`)
///
/// Also emits a `PT_GNU_STACK` header indicating a non-executable stack.
fn build_program_headers(
    writer: &mut elf::ElfWriter,
    merged: &[sections::MergedSection],
    _is_64bit: bool,
) {
    let page_size: u64 = 0x1000;

    // Track segment bounds for each permission category.
    struct SegmentInfo {
        flags: u32,
        min_vaddr: u64,
        max_vaddr: u64,
        min_offset: u64,
        file_size: u64,
        mem_size: u64,
    }

    let mut rx_seg: Option<SegmentInfo> = None; // Executable code
    let mut ro_seg: Option<SegmentInfo> = None; // Read-only data
    let mut rw_seg: Option<SegmentInfo> = None; // Read-write data

    for sec in merged {
        // Only allocatable sections contribute to segments.
        if sec.flags & elf::SHF_ALLOC == 0 {
            continue;
        }

        let is_exec = sec.flags & elf::SHF_EXECINSTR != 0;
        let is_write = sec.flags & elf::SHF_WRITE != 0;
        let sec_end_vaddr = sec.virtual_address + sec.mem_size;
        let file_end = sec.file_offset
            + if sec.section_type == elf::SHT_NOBITS {
                0
            } else {
                sec.data.len() as u64
            };

        let seg = if is_exec {
            &mut rx_seg
        } else if is_write {
            &mut rw_seg
        } else {
            &mut ro_seg
        };

        let flags = if is_exec {
            elf::PF_R | elf::PF_X
        } else if is_write {
            elf::PF_R | elf::PF_W
        } else {
            elf::PF_R
        };

        match seg {
            Some(ref mut info) => {
                if sec.virtual_address < info.min_vaddr {
                    info.min_vaddr = sec.virtual_address;
                    info.min_offset = sec.file_offset;
                }
                if sec_end_vaddr > info.max_vaddr {
                    info.max_vaddr = sec_end_vaddr;
                }
                info.mem_size = info.max_vaddr - info.min_vaddr;
                info.file_size = file_end.saturating_sub(info.min_offset);
            }
            None => {
                *seg = Some(SegmentInfo {
                    flags,
                    min_vaddr: sec.virtual_address,
                    max_vaddr: sec_end_vaddr,
                    min_offset: sec.file_offset,
                    file_size: file_end.saturating_sub(sec.file_offset),
                    mem_size: sec_end_vaddr.saturating_sub(sec.virtual_address),
                });
            }
        }
    }

    // Emit PT_LOAD segments in standard order: R+X, R, R+W.
    // The ELF spec requires p_offset % p_align == p_vaddr % p_align.
    // For the first LOAD segment (executable code), we extend it back to file
    // offset 0 so that the ELF and program headers are included in the mapping.
    // This is standard Linux ELF practice and ensures the congruence property
    // is satisfied across all architectures (i686, x86-64, AArch64, RISC-V 64).
    let mut is_first_load = true;
    for seg_opt in [&rx_seg, &ro_seg, &rw_seg] {
        if let Some(ref info) = seg_opt {
            let (p_offset, p_vaddr, p_filesz, p_memsz) = if is_first_load {
                // First LOAD: start at offset 0 to include ELF headers.
                // Virtual address is page-aligned base, file size covers through
                // end of segment's last section.
                let base_vaddr = info.min_vaddr & !(page_size - 1); // page-align down
                let extra_prefix = info.min_offset; // bytes before first section
                (
                    0u64,
                    base_vaddr,
                    info.file_size + extra_prefix,
                    info.mem_size + (info.min_vaddr - base_vaddr),
                )
            } else {
                // Subsequent LOAD segments: ensure congruence property.
                // p_offset must satisfy p_offset % p_align == p_vaddr % p_align.
                let vaddr_page_offset = info.min_vaddr % page_size;
                let aligned_file_offset = if info.min_offset % page_size == vaddr_page_offset {
                    info.min_offset
                } else {
                    // Adjust file offset down to satisfy congruence.
                    (info.min_offset & !(page_size - 1)) + vaddr_page_offset
                };
                let extra = info.min_offset.saturating_sub(aligned_file_offset);
                (
                    aligned_file_offset,
                    info.min_vaddr - extra,
                    info.file_size + extra,
                    info.mem_size + extra,
                )
            };

            writer.add_program_header(elf::OutputPhdr {
                p_type: elf::PT_LOAD,
                p_flags: info.flags,
                p_offset,
                p_vaddr,
                p_paddr: p_vaddr,
                p_filesz,
                p_memsz,
                p_align: page_size,
            });
            is_first_load = false;
        }
    }

    // PT_DYNAMIC — if a .dynamic section exists (shared libraries).
    for sec in merged {
        if sec.section_type == elf::SHT_DYNAMIC && sec.flags & elf::SHF_ALLOC != 0 {
            writer.add_program_header(elf::OutputPhdr {
                p_type: elf::PT_DYNAMIC,
                p_flags: elf::PF_R | elf::PF_W,
                p_offset: sec.file_offset,
                p_vaddr: sec.virtual_address,
                p_paddr: sec.virtual_address,
                p_filesz: sec.data.len() as u64,
                p_memsz: sec.data.len() as u64,
                p_align: if _is_64bit { 8 } else { 4 },
            });
            break;
        }
    }

    // PT_GNU_STACK — marks the stack as non-executable (W^X security).
    writer.add_program_header(elf::OutputPhdr {
        p_type: elf::PT_GNU_STACK,
        p_flags: elf::PF_R | elf::PF_W,
        p_offset: 0,
        p_vaddr: 0,
        p_paddr: 0,
        p_filesz: 0,
        p_memsz: 0,
        p_align: 0,
    });
}

// ============================================================================
// Default section ordering
// ============================================================================

/// Returns the default section ordering for ELF linker output.
///
/// This ordering matches standard GNU ld behaviour: executable code first,
/// then read-only data, then read-write data, then BSS, then debug/metadata.
fn default_section_order() -> Vec<&'static str> {
    vec![
        ".init",
        ".plt",
        ".text",
        ".fini",
        ".rodata",
        ".eh_frame",
        ".eh_frame_hdr",
        ".init_array",
        ".fini_array",
        ".data",
        ".got",
        ".got.plt",
        ".bss",
        ".comment",
    ]
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::target::TargetConfig;

    #[test]
    fn test_output_mode_variants() {
        let reloc = OutputMode::Relocatable;
        let exec = OutputMode::StaticExecutable;
        let shared = OutputMode::SharedLibrary;

        assert_eq!(reloc, OutputMode::Relocatable);
        assert_eq!(exec, OutputMode::StaticExecutable);
        assert_eq!(shared, OutputMode::SharedLibrary);
        assert_ne!(reloc, exec);
        assert_ne!(exec, shared);
    }

    #[test]
    fn test_output_mode_debug_format() {
        assert_eq!(format!("{:?}", OutputMode::Relocatable), "Relocatable");
        assert_eq!(
            format!("{:?}", OutputMode::StaticExecutable),
            "StaticExecutable"
        );
        assert_eq!(
            format!("{:?}", OutputMode::SharedLibrary),
            "SharedLibrary"
        );
    }

    #[test]
    fn test_output_mode_clone_copy() {
        let mode = OutputMode::StaticExecutable;
        let cloned = mode.clone();
        let copied = mode;
        assert_eq!(cloned, copied);
    }

    #[test]
    fn test_linker_config_default() {
        let target = TargetConfig::x86_64();
        let config = LinkerConfig::new(target);

        assert_eq!(config.output_mode, OutputMode::StaticExecutable);
        assert_eq!(config.output_path, PathBuf::from("a.out"));
        assert!(config.library_paths.is_empty());
        assert!(config.libraries.is_empty());
        assert!(!config.force_static);
        assert_eq!(config.entry_point, "_start");
    }

    #[test]
    fn test_linker_config_custom() {
        let target = TargetConfig::i686();
        let mut config = LinkerConfig::new(target);
        config.output_mode = OutputMode::SharedLibrary;
        config.output_path = PathBuf::from("libtest.so");
        config.library_paths.push(PathBuf::from("/opt/lib"));
        config.libraries.push(String::from("m"));
        config.force_static = true;
        config.entry_point = String::from("my_start");

        assert_eq!(config.output_mode, OutputMode::SharedLibrary);
        assert_eq!(config.output_path, PathBuf::from("libtest.so"));
        assert_eq!(config.library_paths.len(), 1);
        assert_eq!(config.libraries, vec!["m"]);
        assert!(config.force_static);
        assert_eq!(config.entry_point, "my_start");
    }

    #[test]
    fn test_linker_error_display_undefined_symbol() {
        let err = LinkerError::UndefinedSymbol(String::from("main"));
        assert_eq!(format!("{}", err), "undefined reference to `main`");
    }

    #[test]
    fn test_linker_error_display_duplicate_symbol() {
        let err = LinkerError::DuplicateSymbol(String::from("foo"));
        assert_eq!(format!("{}", err), "multiple definition of `foo`");
    }

    #[test]
    fn test_linker_error_display_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = LinkerError::IoError(io_err);
        assert!(format!("{}", err).contains("I/O error"));
    }

    #[test]
    fn test_linker_error_display_invalid_elf() {
        let err = LinkerError::InvalidElfObject(String::from("bad magic"));
        assert_eq!(format!("{}", err), "invalid ELF object: bad magic");
    }

    #[test]
    fn test_linker_error_display_invalid_archive() {
        let err = LinkerError::InvalidArchive(String::from("truncated"));
        assert_eq!(format!("{}", err), "invalid archive: truncated");
    }

    #[test]
    fn test_linker_error_display_unsupported_relocation() {
        let err = LinkerError::UnsupportedRelocation(String::from("R_UNKNOWN_42"));
        assert_eq!(
            format!("{}", err),
            "unsupported relocation: R_UNKNOWN_42"
        );
    }

    #[test]
    fn test_linker_error_display_internal() {
        let err = LinkerError::InternalError(String::from("section overflow"));
        assert_eq!(
            format!("{}", err),
            "internal linker error: section overflow"
        );
    }

    #[test]
    fn test_linker_error_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let err: LinkerError = io_err.into();
        match err {
            LinkerError::IoError(_) => {}
            _ => panic!("expected IoError variant"),
        }
    }

    #[test]
    fn test_linker_error_source() {
        use std::error::Error;
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "not found");
        let err = LinkerError::IoError(io_err);
        assert!(err.source().is_some());

        let err2 = LinkerError::UndefinedSymbol(String::from("x"));
        assert!(err2.source().is_none());
    }

    #[test]
    fn test_crt_search_paths_x86_64() {
        let target = TargetConfig::x86_64();
        let config = LinkerConfig::new(target);
        let paths = build_crt_search_paths(&config);
        assert!(paths
            .iter()
            .any(|p| p.to_str().unwrap().contains("x86_64")));
    }

    #[test]
    fn test_crt_search_paths_i686() {
        let target = TargetConfig::i686();
        let config = LinkerConfig::new(target);
        let paths = build_crt_search_paths(&config);
        assert!(paths
            .iter()
            .any(|p| p.to_str().unwrap().contains("i386")
                || p.to_str().unwrap().contains("lib32")
                || p.to_str().unwrap().contains("i686")));
    }

    #[test]
    fn test_crt_search_paths_aarch64() {
        let target = TargetConfig::aarch64();
        let config = LinkerConfig::new(target);
        let paths = build_crt_search_paths(&config);
        assert!(paths
            .iter()
            .any(|p| p.to_str().unwrap().contains("aarch64")));
    }

    #[test]
    fn test_crt_search_paths_riscv64() {
        let target = TargetConfig::riscv64();
        let config = LinkerConfig::new(target);
        let paths = build_crt_search_paths(&config);
        assert!(paths
            .iter()
            .any(|p| p.to_str().unwrap().contains("riscv64")));
    }

    #[test]
    fn test_crt_not_needed_for_relocatable() {
        let target = TargetConfig::x86_64();
        let mut config = LinkerConfig::new(target);
        config.output_mode = OutputMode::Relocatable;
        let crt = find_crt_objects(&config).unwrap();
        assert!(crt.crt1.is_none());
        assert!(crt.crti.is_none());
        assert!(crt.crtn.is_none());
    }

    #[test]
    fn test_crt_not_needed_for_shared_library() {
        let target = TargetConfig::x86_64();
        let mut config = LinkerConfig::new(target);
        config.output_mode = OutputMode::SharedLibrary;
        let crt = find_crt_objects(&config).unwrap();
        assert!(crt.crt1.is_none());
        assert!(crt.crti.is_none());
        assert!(crt.crtn.is_none());
    }

    #[test]
    fn test_library_name_resolution_not_found() {
        let target = TargetConfig::x86_64();
        let mut config = LinkerConfig::new(target);
        config.libraries.push(String::from("nonexistent_lib_xyz_123"));
        config.force_static = true;
        let libs = resolve_libraries(&config).unwrap();
        // Library not found is silently skipped (consistent with ld behavior).
        assert!(libs.is_empty());
    }

    #[test]
    fn test_section_type_to_elf_mapping() {
        assert_eq!(
            section_type_to_elf(&codegen::SectionType::Text),
            elf::SHT_PROGBITS
        );
        assert_eq!(
            section_type_to_elf(&codegen::SectionType::Data),
            elf::SHT_PROGBITS
        );
        assert_eq!(
            section_type_to_elf(&codegen::SectionType::Rodata),
            elf::SHT_PROGBITS
        );
        assert_eq!(
            section_type_to_elf(&codegen::SectionType::Bss),
            elf::SHT_NOBITS
        );
        assert_eq!(section_type_to_elf(&codegen::SectionType::Custom(42)), 42);
    }

    #[test]
    fn test_section_flags_text() {
        let flags_text = codegen::SectionFlags::text();
        let elf_flags = section_flags_to_elf(&flags_text);
        assert!(elf_flags & elf::SHF_ALLOC != 0);
        assert!(elf_flags & elf::SHF_EXECINSTR != 0);
        assert!(elf_flags & elf::SHF_WRITE == 0);
    }

    #[test]
    fn test_section_flags_data() {
        let flags_data = codegen::SectionFlags::data();
        let elf_flags = section_flags_to_elf(&flags_data);
        assert!(elf_flags & elf::SHF_ALLOC != 0);
        assert!(elf_flags & elf::SHF_WRITE != 0);
        assert!(elf_flags & elf::SHF_EXECINSTR == 0);
    }

    #[test]
    fn test_section_flags_rodata() {
        let flags_ro = codegen::SectionFlags::rodata();
        let elf_flags = section_flags_to_elf(&flags_ro);
        assert!(elf_flags & elf::SHF_ALLOC != 0);
        assert!(elf_flags & elf::SHF_WRITE == 0);
        assert!(elf_flags & elf::SHF_EXECINSTR == 0);
    }

    #[test]
    fn test_section_flags_bss() {
        let flags_bss = codegen::SectionFlags::bss();
        let elf_flags = section_flags_to_elf(&flags_bss);
        assert!(elf_flags & elf::SHF_ALLOC != 0);
        assert!(elf_flags & elf::SHF_WRITE != 0);
    }

    #[test]
    fn test_default_section_order_text_before_data() {
        let order = default_section_order();
        let text_pos = order.iter().position(|&s| s == ".text").unwrap();
        let data_pos = order.iter().position(|&s| s == ".data").unwrap();
        assert!(text_pos < data_pos, ".text must come before .data");
    }

    #[test]
    fn test_default_section_order_rodata_before_data() {
        let order = default_section_order();
        let rodata_pos = order.iter().position(|&s| s == ".rodata").unwrap();
        let data_pos = order.iter().position(|&s| s == ".data").unwrap();
        assert!(rodata_pos < data_pos, ".rodata must come before .data");
    }

    #[test]
    fn test_default_section_order_data_before_bss() {
        let order = default_section_order();
        let data_pos = order.iter().position(|&s| s == ".data").unwrap();
        let bss_pos = order.iter().position(|&s| s == ".bss").unwrap();
        assert!(data_pos < bss_pos, ".data must come before .bss");
    }

    #[test]
    fn test_debug_sections_creation() {
        let dbg = DebugSections {
            debug_info: vec![1, 2, 3],
            debug_abbrev: vec![4, 5],
            debug_line: vec![6],
            debug_str: vec![7, 8, 9, 10],
            debug_aranges: Vec::new(),
            debug_frame: vec![11],
            debug_loc: Vec::new(),
            relocations: Vec::new(),
        };
        assert_eq!(dbg.debug_info.len(), 3);
        assert_eq!(dbg.debug_abbrev.len(), 2);
        assert_eq!(dbg.debug_line.len(), 1);
        assert_eq!(dbg.debug_str.len(), 4);
        assert!(dbg.debug_aranges.is_empty());
        assert_eq!(dbg.debug_frame.len(), 1);
        assert!(dbg.debug_loc.is_empty());
        assert!(dbg.relocations.is_empty());
    }

    #[test]
    fn test_linker_input_empty() {
        let input = LinkerInput {
            objects: Vec::new(),
            debug_sections: None,
        };
        assert!(input.objects.is_empty());
        assert!(input.debug_sections.is_none());
    }

    #[test]
    fn test_linker_input_with_debug() {
        let input = LinkerInput {
            objects: Vec::new(),
            debug_sections: Some(DebugSections {
                debug_info: vec![0xAB],
                debug_abbrev: Vec::new(),
                debug_line: Vec::new(),
                debug_str: Vec::new(),
                debug_aranges: Vec::new(),
                debug_frame: Vec::new(),
                debug_loc: Vec::new(),
                relocations: Vec::new(),
            }),
        };
        assert!(input.debug_sections.is_some());
        assert_eq!(input.debug_sections.unwrap().debug_info, vec![0xAB]);
    }

    #[test]
    fn test_find_file_in_paths_nonexistent() {
        let paths = vec![PathBuf::from("/nonexistent/path")];
        let result = find_file_in_paths("nonexistent.o", &paths);
        assert!(result.is_none());
    }

    #[test]
    fn test_reloc_type_to_elf_x86_64() {
        assert_eq!(
            reloc_type_to_elf(&codegen::RelocationType::X86_64_64),
            relocations::R_X86_64_64
        );
        assert_eq!(
            reloc_type_to_elf(&codegen::RelocationType::X86_64_PC32),
            relocations::R_X86_64_PC32
        );
        assert_eq!(
            reloc_type_to_elf(&codegen::RelocationType::X86_64_PLT32),
            relocations::R_X86_64_PLT32
        );
    }

    #[test]
    fn test_reloc_type_to_elf_i686() {
        assert_eq!(
            reloc_type_to_elf(&codegen::RelocationType::I386_32),
            relocations::R_386_32
        );
        assert_eq!(
            reloc_type_to_elf(&codegen::RelocationType::I386_PC32),
            relocations::R_386_PC32
        );
    }

    #[test]
    fn test_reloc_type_to_elf_aarch64() {
        assert_eq!(
            reloc_type_to_elf(&codegen::RelocationType::Aarch64_ABS64),
            relocations::R_AARCH64_ABS64
        );
        assert_eq!(
            reloc_type_to_elf(&codegen::RelocationType::Aarch64_CALL26),
            relocations::R_AARCH64_CALL26
        );
    }

    #[test]
    fn test_reloc_type_to_elf_riscv64() {
        assert_eq!(
            reloc_type_to_elf(&codegen::RelocationType::Riscv_64),
            relocations::R_RISCV_64
        );
        assert_eq!(
            reloc_type_to_elf(&codegen::RelocationType::Riscv_Call),
            relocations::R_RISCV_CALL
        );
        assert_eq!(
            reloc_type_to_elf(&codegen::RelocationType::Riscv_Jal),
            relocations::R_RISCV_JAL
        );
    }

    #[test]
    fn test_codegen_symbol_to_linker_conversion() {
        let codegen_sym = codegen::Symbol {
            name: String::from("test_func"),
            section_index: 0,
            offset: 0x100,
            size: 64,
            binding: codegen::SymbolBinding::Global,
            symbol_type: codegen::SymbolType::Function,
            visibility: codegen::SymbolVisibility::Default,
            is_definition: true,
        };
        let linker_sym = codegen_symbol_to_linker(&codegen_sym, 0);
        assert_eq!(linker_sym.name, "test_func");
        assert_eq!(linker_sym.binding, symbols::SymbolBinding::Global);
        assert_eq!(linker_sym.symbol_type, symbols::SymbolType::Function);
        assert_eq!(linker_sym.visibility, symbols::SymbolVisibility::Default);
        assert_eq!(linker_sym.section_index, 0);
        assert_eq!(linker_sym.value, 0x100);
        assert_eq!(linker_sym.size, 64);
        assert_eq!(linker_sym.source_object, 0);
        assert!(linker_sym.is_defined());
    }

    #[test]
    fn test_codegen_section_to_input_conversion() {
        let codegen_sec = codegen::Section {
            name: String::from(".text"),
            data: vec![0xCC; 128],
            section_type: codegen::SectionType::Text,
            alignment: 16,
            flags: codegen::SectionFlags::text(),
        };
        let input_sec = codegen_section_to_input(&codegen_sec, 0, 1);
        assert_eq!(input_sec.name, ".text");
        assert_eq!(input_sec.data.len(), 128);
        assert_eq!(input_sec.section_type, elf::SHT_PROGBITS);
        assert_eq!(input_sec.alignment, 16);
        assert!(input_sec.flags & elf::SHF_ALLOC != 0);
        assert!(input_sec.flags & elf::SHF_EXECINSTR != 0);
        assert_eq!(input_sec.object_index, 0);
        assert_eq!(input_sec.original_index, 1);
    }

    #[test]
    fn test_link_relocatable_empty_input() {
        let target = TargetConfig::x86_64();
        let config = LinkerConfig {
            output_mode: OutputMode::Relocatable,
            output_path: PathBuf::from("test.o"),
            library_paths: Vec::new(),
            libraries: Vec::new(),
            force_static: false,
            target,
            entry_point: String::from("_start"),
        };
        let input = LinkerInput {
            objects: Vec::new(),
            debug_sections: None,
        };
        let result = link(input, &config);
        assert!(
            result.is_ok(),
            "relocatable link with no inputs should succeed"
        );
        let elf_data = result.unwrap();
        // Verify it starts with ELF magic.
        assert!(elf_data.len() >= 4);
        assert_eq!(&elf_data[0..4], &[0x7f, b'E', b'L', b'F']);
    }

    #[test]
    fn test_lib_search_paths_include_custom() {
        let target = TargetConfig::x86_64();
        let mut config = LinkerConfig::new(target);
        config.library_paths.push(PathBuf::from("/my/custom/lib"));
        let paths = build_lib_search_paths(&config);
        assert!(paths.iter().any(|p| p == &PathBuf::from("/my/custom/lib")));
    }

    #[test]
    fn test_crt_search_paths_include_custom_library_paths() {
        let target = TargetConfig::x86_64();
        let mut config = LinkerConfig::new(target);
        config.library_paths.push(PathBuf::from("/custom/crt"));
        let paths = build_crt_search_paths(&config);
        assert!(paths.iter().any(|p| p == &PathBuf::from("/custom/crt")));
    }
}
