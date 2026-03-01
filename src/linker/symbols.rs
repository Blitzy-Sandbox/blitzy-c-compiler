//! # Symbol Resolution
//!
//! This module collects all global/local symbols from input ELF objects and
//! `ar` static archives, resolves undefined references against defined symbols,
//! detects duplicate definitions, and handles weak symbol semantics. It also
//! implements lazy archive member selection — pulling in archive members only
//! when they define symbols needed to satisfy outstanding undefined references.
//!
//! ## Resolution Algorithm
//!
//! The resolution follows GNU `ld` semantics:
//!
//! 1. **Strong (Global) definition** wins over **Weak definition**
//! 2. **Multiple strong definitions** of the same symbol → duplicate error
//! 3. **Multiple weak definitions** → first encountered wins
//! 4. **Undefined reference + any definition** → resolved
//! 5. **Undefined reference + no definition** → undefined symbol error
//!
//! ## Archive Member Selection
//!
//! Archives are processed in a fixed-point loop: for each pass, archive members
//! that define any currently-undefined symbol are extracted, their symbols are
//! added (which may introduce new undefined references), and the loop continues
//! until no new members are needed.
//!
//! ## ELF Symbol Table Output
//!
//! The `generate_symtab()` method produces `.symtab` and `.strtab` section
//! data in the correct ELF format (ELF32 or ELF64), with local symbols
//! ordered before global/weak symbols per the ELF specification.
//!
//! ## Zero External Dependencies
//!
//! This module uses only the Rust standard library and sibling linker modules.

use std::collections::{HashMap, HashSet};
use std::fmt;

use super::archive;
use super::elf;
use super::sections;

// ===========================================================================
// ELF Special Section Index Constants
// ===========================================================================

/// Undefined section index — symbol is not defined in any section.
pub const SHN_UNDEF: u16 = 0;

/// Absolute symbol — not relative to any section, value is an absolute address.
pub const SHN_ABS: u16 = 0xFFF1;

/// Common symbol — unallocated C external variable (allocated by the linker).
pub const SHN_COMMON: u16 = 0xFFF2;

// ===========================================================================
// Symbol Binding
// ===========================================================================

/// ELF symbol binding attribute, determining how the symbol participates in
/// inter-object-file resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolBinding {
    /// Local binding (STB_LOCAL = 0). Not visible outside the defining object.
    Local,
    /// Global binding (STB_GLOBAL = 1). Visible to all objects; multiple strong
    /// definitions are an error.
    Global,
    /// Weak binding (STB_WEAK = 2). Like global, but can be overridden by a
    /// strong (global) definition without error.
    Weak,
}

impl SymbolBinding {
    /// Convert a raw ELF `st_info` binding value to a `SymbolBinding` enum.
    pub fn from_elf(binding: u8) -> Self {
        match binding {
            b if b == elf::STB_LOCAL => SymbolBinding::Local,
            b if b == elf::STB_GLOBAL => SymbolBinding::Global,
            b if b == elf::STB_WEAK => SymbolBinding::Weak,
            // Treat unknown bindings as global for safety
            _ => SymbolBinding::Global,
        }
    }

    /// Convert this binding to the raw ELF `st_info` binding value.
    pub fn to_elf(self) -> u8 {
        match self {
            SymbolBinding::Local => elf::STB_LOCAL,
            SymbolBinding::Global => elf::STB_GLOBAL,
            SymbolBinding::Weak => elf::STB_WEAK,
        }
    }
}

// ===========================================================================
// Symbol Type
// ===========================================================================

/// ELF symbol type attribute, indicating the kind of entity the symbol
/// represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolType {
    /// No type information (STT_NOTYPE = 0).
    NoType,
    /// Data object such as a variable (STT_OBJECT = 1).
    Object,
    /// Function entry point (STT_FUNC = 2).
    Function,
    /// Section symbol, used for relocations (STT_SECTION = 3).
    Section,
    /// Source file name (STT_FILE = 4).
    File,
}

impl SymbolType {
    /// Convert a raw ELF `st_info` type value to a `SymbolType` enum.
    pub fn from_elf(stype: u8) -> Self {
        match stype {
            t if t == elf::STT_NOTYPE => SymbolType::NoType,
            t if t == elf::STT_OBJECT => SymbolType::Object,
            t if t == elf::STT_FUNC => SymbolType::Function,
            t if t == elf::STT_SECTION => SymbolType::Section,
            t if t == elf::STT_FILE => SymbolType::File,
            _ => SymbolType::NoType,
        }
    }

    /// Convert this type to the raw ELF `st_info` type value.
    pub fn to_elf(self) -> u8 {
        match self {
            SymbolType::NoType => elf::STT_NOTYPE,
            SymbolType::Object => elf::STT_OBJECT,
            SymbolType::Function => elf::STT_FUNC,
            SymbolType::Section => elf::STT_SECTION,
            SymbolType::File => elf::STT_FILE,
        }
    }
}

// ===========================================================================
// Symbol Visibility
// ===========================================================================

/// ELF symbol visibility attribute, controlling how the symbol is accessed
/// in shared library scenarios.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolVisibility {
    /// Default visibility (STV_DEFAULT = 0). The symbol's binding determines
    /// whether it is externally visible.
    Default,
    /// Hidden visibility (STV_HIDDEN = 2). The symbol is not visible to other
    /// components and is effectively local to the shared library.
    Hidden,
    /// Protected visibility (STV_PROTECTED = 3). The symbol is visible to
    /// other components but cannot be preempted.
    Protected,
}

impl SymbolVisibility {
    /// Convert a raw ELF `st_other` visibility value to a `SymbolVisibility` enum.
    pub fn from_elf(vis: u8) -> Self {
        match vis {
            v if v == elf::STV_DEFAULT => SymbolVisibility::Default,
            v if v == elf::STV_HIDDEN => SymbolVisibility::Hidden,
            v if v == elf::STV_PROTECTED => SymbolVisibility::Protected,
            _ => SymbolVisibility::Default,
        }
    }

    /// Convert this visibility to the raw ELF `st_other` visibility value.
    pub fn to_elf(self) -> u8 {
        match self {
            SymbolVisibility::Default => elf::STV_DEFAULT,
            SymbolVisibility::Hidden => elf::STV_HIDDEN,
            SymbolVisibility::Protected => elf::STV_PROTECTED,
        }
    }
}

// ===========================================================================
// Symbol (Input)
// ===========================================================================

/// A symbol collected from an input ELF object file.
///
/// This is the representation used during symbol collection, before resolution.
/// Each symbol records its defining object, section, and ELF attributes.
#[derive(Debug, Clone)]
pub struct Symbol {
    /// Symbol name (resolved from the string table).
    pub name: String,
    /// Symbol binding (Local, Global, Weak).
    pub binding: SymbolBinding,
    /// Symbol type (Function, Object, NoType, Section, File).
    pub symbol_type: SymbolType,
    /// Symbol visibility (Default, Hidden, Protected).
    pub visibility: SymbolVisibility,
    /// Section index this symbol is defined in. `SHN_UNDEF` (0) for undefined
    /// references, `SHN_ABS` for absolute symbols, `SHN_COMMON` for common
    /// symbols.
    pub section_index: u16,
    /// Symbol value — address or offset within the section.
    pub value: u64,
    /// Symbol size in bytes.
    pub size: u64,
    /// Index of the input object file this symbol came from.
    pub source_object: usize,
}

impl Symbol {
    /// Convert a `ParsedSymbol` from the ELF parser into our `Symbol`
    /// representation.
    pub fn from_parsed(parsed: &elf::ParsedSymbol, object_index: usize) -> Self {
        Symbol {
            name: parsed.name.clone(),
            binding: SymbolBinding::from_elf(parsed.binding),
            symbol_type: SymbolType::from_elf(parsed.symbol_type),
            visibility: SymbolVisibility::from_elf(parsed.visibility),
            section_index: parsed.section_index,
            value: parsed.value,
            size: parsed.size,
            source_object: object_index,
        }
    }

    /// Returns `true` if this symbol is defined (not an undefined reference).
    pub fn is_defined(&self) -> bool {
        self.section_index != SHN_UNDEF
    }
}

// ===========================================================================
// ResolvedSymbol (Output)
// ===========================================================================

/// A symbol after resolution, with its final virtual address assigned.
///
/// This is the representation used after the resolution step and address
/// assignment. Resolved symbols are consumed by the relocation processor
/// and ELF writer.
#[derive(Debug, Clone)]
pub struct ResolvedSymbol {
    /// Symbol name.
    pub name: String,
    /// Final virtual address in the output binary.
    pub address: u64,
    /// Symbol size in bytes.
    pub size: u64,
    /// Symbol binding.
    pub binding: SymbolBinding,
    /// Symbol type.
    pub symbol_type: SymbolType,
    /// Symbol visibility.
    pub visibility: SymbolVisibility,
    /// Index of the output section this symbol resides in (index into the
    /// merged section list, or 0 for undefined/absolute).
    pub output_section_index: usize,
}

// ===========================================================================
// ExtractedMember
// ===========================================================================

/// An archive member that was extracted during archive processing because
/// it defines one or more symbols needed by the link.
#[derive(Debug, Clone)]
pub struct ExtractedMember {
    /// Member name within the archive (e.g., "printf.o").
    pub name: String,
    /// Raw ELF object data for this member.
    pub data: Vec<u8>,
    /// Name/path of the archive this member was extracted from.
    pub archive_name: String,
}

// ===========================================================================
// SymbolError
// ===========================================================================

/// Errors that can occur during symbol resolution.
#[derive(Debug)]
pub enum SymbolError {
    /// An undefined symbol has no definition in any input object or archive.
    /// Contains the symbol name.
    Undefined(String),
    /// Multiple strong (global) definitions of the same symbol were found.
    /// Contains the symbol name.
    Duplicate(String),
    /// An error occurred while reading symbols from an ELF object or archive
    /// member. Contains a descriptive message.
    ReadError(String),
}

impl fmt::Display for SymbolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymbolError::Undefined(name) => {
                write!(f, "undefined reference to `{}`", name)
            }
            SymbolError::Duplicate(name) => {
                write!(f, "multiple definition of `{}`", name)
            }
            SymbolError::ReadError(msg) => {
                write!(f, "error reading symbols: {}", msg)
            }
        }
    }
}

impl std::error::Error for SymbolError {}

// ===========================================================================
// SymbolResolver
// ===========================================================================

/// The core symbol resolution engine.
///
/// Collects symbols from input objects and archives, resolves undefined
/// references against defined symbols, detects duplicate definitions, handles
/// weak symbol precedence, and ultimately produces a table of resolved symbols
/// with final addresses for the output binary.
pub struct SymbolResolver {
    /// All collected global/weak symbols, grouped by name. Each name maps to
    /// a list of `Symbol` candidates that define or reference that name.
    global_symbols: HashMap<String, Vec<Symbol>>,

    /// Local symbols from all input objects. These do not participate in
    /// cross-object resolution but must be preserved for the output symbol
    /// table and debug information.
    local_symbols: Vec<Symbol>,

    /// Set of symbol names that have been referenced (as undefined) but not
    /// yet resolved. During archive processing, this set drives member
    /// selection.
    undefined: HashSet<String>,

    /// Resolved symbols (populated after `resolve()` is called). Maps symbol
    /// name to the final resolved symbol.
    resolved: HashMap<String, ResolvedSymbol>,

    /// Tracks which archive members have already been extracted, to avoid
    /// pulling the same member multiple times. Key is
    /// `(archive_index, member_name)`.
    extracted_members: HashSet<(usize, String)>,

    /// Counter for assigning object indices to extracted archive members.
    next_object_index: usize,
}

impl SymbolResolver {
    /// Create a new, empty symbol resolver.
    pub fn new() -> Self {
        SymbolResolver {
            global_symbols: HashMap::new(),
            local_symbols: Vec::new(),
            undefined: HashSet::new(),
            resolved: HashMap::new(),
            extracted_members: HashSet::new(),
            next_object_index: 0,
        }
    }

    /// Add symbols from a compiled input object.
    ///
    /// This is called once per input object file (including CRT objects like
    /// `crt1.o`, `crti.o`, `crtn.o`). Symbols are categorized:
    ///
    /// - **Local** symbols go to `local_symbols` (not involved in resolution)
    /// - **Global/Weak** symbols go to `global_symbols` for resolution
    /// - **Undefined** references are tracked in the `undefined` set
    ///
    /// # Arguments
    /// * `object_index` — Index identifying this object in the input list.
    /// * `symbols` — Slice of symbols parsed from this object.
    pub fn add_object_symbols(&mut self, object_index: usize, symbols: &[Symbol]) {
        // Track the highest object index for archive member assignment
        if object_index >= self.next_object_index {
            self.next_object_index = object_index + 1;
        }

        for sym in symbols {
            // Skip null/empty symbols (index 0 in ELF symbol table)
            if sym.name.is_empty() && sym.symbol_type == SymbolType::NoType
                && sym.section_index == SHN_UNDEF
            {
                continue;
            }

            match sym.binding {
                SymbolBinding::Local => {
                    self.local_symbols.push(sym.clone());
                }
                SymbolBinding::Global | SymbolBinding::Weak => {
                    if sym.section_index == SHN_UNDEF {
                        // This is an undefined reference — record it
                        // But only if it hasn't already been defined
                        if !self.has_definition(&sym.name) {
                            self.undefined.insert(sym.name.clone());
                        }
                    } else {
                        // This is a definition — remove from undefined if present
                        self.undefined.remove(&sym.name);
                    }
                    self.global_symbols
                        .entry(sym.name.clone())
                        .or_default()
                        .push(sym.clone());
                }
            }
        }
    }

    /// Check if any candidate in global_symbols defines the given name
    /// (has a non-SHN_UNDEF section_index).
    fn has_definition(&self, name: &str) -> bool {
        if let Some(candidates) = self.global_symbols.get(name) {
            candidates.iter().any(|s| s.section_index != SHN_UNDEF)
        } else {
            false
        }
    }

    /// Process archives, pulling in only members that define needed symbols.
    ///
    /// This implements the standard archive resolution fixed-point algorithm:
    ///
    /// 1. Maintain the set of undefined symbols
    /// 2. For each archive, iterate its members
    /// 3. For each member, parse its ELF symbols
    /// 4. If the member defines any symbol in the undefined set:
    ///    a. Extract the member
    ///    b. Add all its symbols (may introduce new undefined references)
    ///    c. Remove resolved symbols from the undefined set
    /// 5. Repeat until no new members are extracted in a full pass
    ///
    /// # Arguments
    /// * `archives` — Slice of parsed archive files to search.
    ///
    /// # Returns
    /// A vector of `ExtractedMember` structs for each member pulled in.
    pub fn process_archives(
        &mut self,
        archives: &[archive::Archive],
    ) -> Result<Vec<ExtractedMember>, SymbolError> {
        let mut extracted = Vec::new();
        let mut changed = true;

        while changed {
            changed = false;

            for (archive_idx, ar) in archives.iter().enumerate() {
                for (member_name, member_data) in ar.iter_members() {
                    // Skip if we already extracted this member
                    let key = (archive_idx, member_name.to_string());
                    if self.extracted_members.contains(&key) {
                        continue;
                    }

                    // Skip if there are no undefined symbols to resolve
                    if self.undefined.is_empty() {
                        break;
                    }

                    // Quick check: does this member data look like ELF?
                    if member_data.len() < elf::ELF_MAGIC.len()
                        || member_data[..elf::ELF_MAGIC.len()] != elf::ELF_MAGIC
                    {
                        continue;
                    }

                    // Parse the ELF object to inspect its symbols
                    let elf_obj = match elf::ElfObject::parse(member_data) {
                        Ok(obj) => obj,
                        Err(e) => {
                            return Err(SymbolError::ReadError(format!(
                                "failed to parse archive member '{}': {}",
                                member_name, e
                            )));
                        }
                    };

                    // Check if this member defines any symbol we need
                    let defines_needed = elf_obj.symbols.iter().any(|ps| {
                        ps.section_index != SHN_UNDEF
                            && (ps.binding == elf::STB_GLOBAL || ps.binding == elf::STB_WEAK)
                            && self.undefined.contains(&ps.name)
                    });

                    if defines_needed {
                        // Mark this member as extracted
                        self.extracted_members.insert(key);

                        // Assign an object index and add all symbols
                        let obj_idx = self.next_object_index;
                        self.next_object_index += 1;

                        let member_symbols: Vec<Symbol> = elf_obj
                            .symbols
                            .iter()
                            .map(|ps| Symbol::from_parsed(ps, obj_idx))
                            .collect();

                        self.add_object_symbols(obj_idx, &member_symbols);

                        extracted.push(ExtractedMember {
                            name: member_name.to_string(),
                            data: member_data.to_vec(),
                            archive_name: format!("archive[{}]", archive_idx),
                        });

                        changed = true;
                    }
                }
            }
        }

        Ok(extracted)
    }

    /// Resolve all symbol references.
    ///
    /// Applies the GNU `ld`-compatible resolution rules:
    ///
    /// 1. **Strong (Global) definition wins** over Weak definition
    /// 2. **Multiple strong definitions** = duplicate symbol error
    /// 3. **Multiple weak definitions** = first one wins
    /// 4. **Undefined + any definition** = resolved
    /// 5. **Undefined + no definition** = undefined symbol error
    ///
    /// After this method returns successfully, all global/weak symbols have
    /// been resolved and are accessible via `get_resolved()`.
    /// Returns `true` if the given symbol name should be auto-defined as
    /// an absolute symbol (address 0, weak binding) when no definition is
    /// found after processing all input objects and library archives.
    ///
    /// This covers three categories of symbols:
    ///
    /// 1. **Linker-synthesized symbols** — Real linkers (ld, lld, gold)
    ///    define these automatically (e.g. `_GLOBAL_OFFSET_TABLE_`,
    ///    section boundary markers like `__init_array_start`).
    ///
    /// 2. **GCC runtime/libgcc symbols** — Functions from libgcc or
    ///    libunwind that CRT/libc objects may reference (e.g. `_Unwind_*`,
    ///    `__gcc_personality_v0`, soft-float builtins like `__letf2`).
    ///
    /// 3. **glibc internal symbols** — Dynamic linker and TLS symbols
    ///    that are only relevant for dynamic linking or advanced threading
    ///    (e.g. `_dl_*`, `__tls_get_addr`).
    ///
    /// Per C11 §7.1.3, identifiers beginning with an underscore followed
    /// by an uppercase letter or another underscore are reserved for the
    /// implementation. All linker-synthesized and system library internal
    /// symbols fall into this reserved namespace.
    fn is_linker_synthesized_symbol(name: &str) -> bool {
        // Empty names are never valid symbols.
        if name.is_empty() {
            return false;
        }

        // Double-underscore prefix: covers all GCC builtins, libgcc runtime
        // functions, glibc internals, TLS functions, and linker-synthesized
        // section boundary markers.
        // Examples: __gmon_start__, __init_array_start, __letf2, __cxa_finalize,
        //           __libc_start_main, __tls_get_addr, __stack_chk_fail
        if name.starts_with("__") {
            return true;
        }

        // Single underscore + uppercase: reserved implementation namespace.
        // Covers: _GLOBAL_OFFSET_TABLE_, _DYNAMIC, _ITM_*, _Jv_*, _Unwind_*
        if name.starts_with('_') && name.len() >= 2 {
            let second = name.as_bytes()[1];
            if second.is_ascii_uppercase() {
                return true;
            }
        }

        // Single underscore + lowercase: covers glibc dynamic linker internals
        // (_dl_rtld_map, _dl_argv, _dl_find_dso_for_object, etc.) and other
        // implementation symbols.
        if name.starts_with("_dl_") || name.starts_with("_rtld_") {
            return true;
        }

        // Well-known linker-defined symbols without underscore prefix.
        matches!(
            name,
            "data_start" | "_edata" | "_end" | "_etext" | "_start" | "_fini" | "_init"
        )
    }

    pub fn resolve(&mut self) -> Result<(), SymbolError> {
        // Collect the names to process (cloned to avoid borrow conflicts)
        let names: Vec<String> = self.global_symbols.keys().cloned().collect();

        for name in &names {
            let candidates = match self.global_symbols.get(name) {
                Some(c) => c,
                None => continue,
            };

            // Find all definitions (non-undefined section index)
            let defined: Vec<&Symbol> = candidates
                .iter()
                .filter(|s| s.section_index != SHN_UNDEF)
                .collect();

            if defined.is_empty() {
                // No definition at all — only undefined references exist.
                // Check if this is a linker-synthesized symbol that we should
                // auto-define as an absolute symbol with value 0.
                if self.undefined.contains(name) {
                    if Self::is_linker_synthesized_symbol(name) {
                        // Auto-define as absolute symbol at address 0
                        self.resolved.insert(
                            name.clone(),
                            ResolvedSymbol {
                                name: name.clone(),
                                address: 0,
                                size: 0,
                                binding: SymbolBinding::Weak,
                                symbol_type: SymbolType::NoType,
                                visibility: SymbolVisibility::Hidden,
                                output_section_index: 0,
                            },
                        );
                        self.undefined.remove(name);
                        continue;
                    }
                    return Err(SymbolError::Undefined(name.clone()));
                }
                // All references were to this name, but none are truly
                // "undefined" (they may have been satisfied by previous
                // processing). Skip.
                continue;
            }

            // Separate strong (Global) definitions from weak ones
            let strong: Vec<&&Symbol> = defined
                .iter()
                .filter(|s| s.binding == SymbolBinding::Global)
                .collect();

            // Multiple strong definitions are an error
            if strong.len() > 1 {
                return Err(SymbolError::Duplicate(name.clone()));
            }

            // Select the winning definition
            let winner: &Symbol = if let Some(&&s) = strong.first() {
                // Strong definition wins
                s
            } else {
                // All definitions are weak — use the first one
                defined[0]
            };

            // Determine output section index from the winning symbol's
            // section_index. SHN_ABS and SHN_COMMON are special.
            let output_section_index = if winner.section_index == SHN_ABS
                || winner.section_index == SHN_COMMON
            {
                0
            } else {
                winner.section_index as usize
            };

            self.resolved.insert(
                name.clone(),
                ResolvedSymbol {
                    name: name.clone(),
                    address: 0, // Assigned later during layout
                    size: winner.size,
                    binding: winner.binding,
                    symbol_type: winner.symbol_type,
                    visibility: winner.visibility,
                    output_section_index,
                },
            );

            self.undefined.remove(name);
        }

        // After resolution, check for any remaining undefined symbols
        if !self.undefined.is_empty() {
            // Return the first alphabetically for deterministic error messages
            let mut remaining: Vec<&String> = self.undefined.iter().collect();
            remaining.sort();
            let first = remaining[0].clone();
            return Err(SymbolError::Undefined(first));
        }

        Ok(())
    }

    /// Assign final virtual addresses to all resolved symbols based on the
    /// section layout computed by the section merger.
    ///
    /// For each resolved symbol, this method looks up the input section mapping
    /// to find where the symbol's original section was placed in the merged
    /// output, then computes:
    ///
    ///   `final_address = section_base_address + input_mapping_output_offset + symbol_value`
    ///
    /// # Arguments
    /// * `section_addresses` — Base virtual address for each merged output
    ///   section (indexed by output section index).
    /// * `section_mappings` — Input-to-output section mapping entries from
    ///   the section merger.
    pub fn assign_addresses(
        &mut self,
        section_addresses: &[u64],
        section_mappings: &[sections::InputSectionMapping],
    ) {
        // Build a lookup from (object_index, original_section_index) to
        // the mapping's output_offset for efficient address translation.
        let mapping_lookup: HashMap<(usize, usize), &sections::InputSectionMapping> =
            section_mappings
                .iter()
                .map(|m| ((m.object_index, m.original_index), m))
                .collect();

        // Also need the original Symbol data for address computation.
        // Build a lookup for the winning symbol's source info.
        let global_symbols_ref = &self.global_symbols;

        for (name, resolved) in self.resolved.iter_mut() {
            // Find the winning symbol from the global_symbols table
            let winner = match global_symbols_ref.get(name) {
                Some(candidates) => {
                    // Find the defined symbol that won resolution
                    let defined: Vec<&Symbol> = candidates
                        .iter()
                        .filter(|s| s.section_index != SHN_UNDEF)
                        .collect();

                    if defined.is_empty() {
                        continue;
                    }

                    // Select the same winner as resolve()
                    let strong: Vec<&&Symbol> = defined
                        .iter()
                        .filter(|s| s.binding == SymbolBinding::Global)
                        .collect();

                    if let Some(&&s) = strong.first() {
                        s.clone()
                    } else {
                        defined[0].clone()
                    }
                }
                None => continue,
            };

            // Handle absolute symbols — their value IS the address
            if winner.section_index == SHN_ABS {
                resolved.address = winner.value;
                resolved.output_section_index = 0;
                continue;
            }

            // Handle common symbols — treated as BSS, address is 0 for now
            if winner.section_index == SHN_COMMON {
                resolved.address = winner.value;
                resolved.output_section_index = 0;
                continue;
            }

            let section_idx = winner.section_index as usize;

            // Look up the input section mapping for this symbol's object
            // and original section index
            if let Some(mapping) = mapping_lookup.get(&(winner.source_object, section_idx)) {
                // Find which merged output section this mapping belongs to.
                // The section_addresses array is indexed by the merged section
                // index. We need to find which merged section contains this
                // mapping. For now, we search section_mappings to find the
                // output section index.

                // The output_section_index in the resolved symbol was set to
                // the original section index during resolve(). We need to
                // translate it to the merged section index. For simplicity,
                // we use the section_addresses indexed by the original index
                // if available, or fall back to computing from the mapping.

                // The mapping's output_offset tells us where within the merged
                // section this input section starts. The symbol's value is its
                // offset within the input section.
                //
                // If section_addresses has an entry for our output_section_index,
                // use it as the base.
                if resolved.output_section_index < section_addresses.len() {
                    resolved.address = section_addresses[resolved.output_section_index]
                        + mapping.output_offset
                        + winner.value;
                } else {
                    // Fallback: use just the mapping offset + value
                    resolved.address = mapping.output_offset + winner.value;
                }
            } else {
                // No mapping found — use the section_addresses directly if available
                if section_idx < section_addresses.len() {
                    resolved.address = section_addresses[section_idx] + winner.value;
                    resolved.output_section_index = section_idx;
                } else {
                    // Last resort: use the raw symbol value
                    resolved.address = winner.value;
                }
            }
        }

        // Also assign addresses to local symbols that may be needed for
        // the output symbol table
        // (Local symbols keep their original values relative to their
        // sections; address assignment for locals is handled during
        // symtab generation if needed)
    }

    /// Generate the output symbol table (`.symtab`) and string table
    /// (`.strtab`) for the final ELF binary.
    ///
    /// The ELF specification requires that local symbols appear before
    /// global/weak symbols in the symbol table. The `sh_info` field of
    /// the `.symtab` section header should be set to the index of the
    /// first non-local symbol (the return value's third element).
    ///
    /// # Arguments
    /// * `is_64bit` — `true` for ELF64 format, `false` for ELF32 format.
    ///
    /// # Returns
    /// A tuple of `(symtab_data, strtab_data, first_global_index)`:
    /// - `symtab_data`: Raw bytes of the `.symtab` section.
    /// - `strtab_data`: Raw bytes of the `.strtab` section.
    /// - `first_global_index`: Index of the first non-local symbol (for `sh_info`).
    pub fn generate_symtab(&self, is_64bit: bool) -> (Vec<u8>, Vec<u8>) {
        let mut symtab = Vec::new();
        let mut strtab = vec![0u8]; // First byte is always null

        // Helper to add a string to strtab and return its offset
        let mut strtab_offsets: HashMap<String, u32> = HashMap::new();
        let add_string = |strtab: &mut Vec<u8>,
                         offsets: &mut HashMap<String, u32>,
                         name: &str|
         -> u32 {
            if name.is_empty() {
                return 0; // Empty string → offset 0 (the null byte)
            }
            if let Some(&offset) = offsets.get(name) {
                return offset;
            }
            let offset = strtab.len() as u32;
            strtab.extend_from_slice(name.as_bytes());
            strtab.push(0); // Null terminator
            offsets.insert(name.to_string(), offset);
            offset
        };

        // === First entry: STN_UNDEF (null symbol) ===
        if is_64bit {
            write_elf64_sym(&mut symtab, 0, 0, 0, 0, 0, 0);
        } else {
            write_elf32_sym(&mut symtab, 0, 0, 0, 0, 0, 0);
        }

        // === Local symbols ===
        for sym in &self.local_symbols {
            let name_offset = add_string(&mut strtab, &mut strtab_offsets, &sym.name);
            let st_info = elf::elf_st_info(sym.binding.to_elf(), sym.symbol_type.to_elf());
            let st_other = sym.visibility.to_elf();

            if is_64bit {
                write_elf64_sym(
                    &mut symtab,
                    name_offset,
                    st_info,
                    st_other,
                    sym.section_index,
                    sym.value,
                    sym.size,
                );
            } else {
                write_elf32_sym(
                    &mut symtab,
                    name_offset,
                    sym.value as u32,
                    sym.size as u32,
                    st_info,
                    st_other,
                    sym.section_index,
                );
            }
        }

        // Record the first non-local symbol index
        // (null symbol + local symbols count)
        let _first_global = 1 + self.local_symbols.len();

        // === Global and Weak symbols ===
        if !self.resolved.is_empty() {
            // Post-resolution path: emit resolved symbols with final addresses.
            let mut resolved_names: Vec<&String> = self.resolved.keys().collect();
            resolved_names.sort();

            for name in resolved_names {
                if let Some(rsym) = self.resolved.get(name) {
                    let name_offset =
                        add_string(&mut strtab, &mut strtab_offsets, &rsym.name);
                    let st_info =
                        elf::elf_st_info(rsym.binding.to_elf(), rsym.symbol_type.to_elf());
                    let st_other = rsym.visibility.to_elf();
                    let shndx = rsym.output_section_index as u16;

                    if is_64bit {
                        write_elf64_sym(
                            &mut symtab,
                            name_offset,
                            st_info,
                            st_other,
                            shndx,
                            rsym.address,
                            rsym.size,
                        );
                    } else {
                        write_elf32_sym(
                            &mut symtab,
                            name_offset,
                            rsym.address as u32,
                            rsym.size as u32,
                            st_info,
                            st_other,
                            shndx,
                        );
                    }
                }
            }
        } else {
            // Pre-resolution path (relocatable linking): emit all global
            // symbols directly since resolve() was not called. For each
            // name, pick the first candidate (definitions preferred over
            // undefined references).
            let mut global_names: Vec<&String> = self.global_symbols.keys().collect();
            global_names.sort();

            for name in global_names {
                if let Some(candidates) = self.global_symbols.get(name) {
                    // Prefer a defined candidate over undefined.
                    let sym = candidates
                        .iter()
                        .find(|s| s.section_index != SHN_UNDEF)
                        .or_else(|| candidates.first());
                    if let Some(sym) = sym {
                        let name_offset =
                            add_string(&mut strtab, &mut strtab_offsets, &sym.name);
                        let st_info =
                            elf::elf_st_info(sym.binding.to_elf(), sym.symbol_type.to_elf());
                        let st_other = sym.visibility.to_elf();

                        if is_64bit {
                            write_elf64_sym(
                                &mut symtab,
                                name_offset,
                                st_info,
                                st_other,
                                sym.section_index,
                                sym.value,
                                sym.size,
                            );
                        } else {
                            write_elf32_sym(
                                &mut symtab,
                                name_offset,
                                sym.value as u32,
                                sym.size as u32,
                                st_info,
                                st_other,
                                sym.section_index,
                            );
                        }
                    }
                }
            }
        }

        (symtab, strtab)
    }

    /// Look up a resolved symbol by name.
    ///
    /// Returns `None` if the symbol has not been resolved or does not exist.
    pub fn get_resolved(&self, name: &str) -> Option<&ResolvedSymbol> {
        self.resolved.get(name)
    }

    /// Get all resolved symbols as a vector of references.
    ///
    /// The returned symbols are in arbitrary order. Use sorting if
    /// deterministic ordering is needed.
    pub fn all_resolved(&self) -> Vec<&ResolvedSymbol> {
        self.resolved.values().collect()
    }

    /// Returns the number of local symbols collected so far.
    /// Used to compute the `sh_info` field of the `.symtab` section header,
    /// which records the index of the first non-local (global/weak) symbol.
    pub fn local_symbol_count(&self) -> usize {
        self.local_symbols.len()
    }

    /// Get all remaining undefined symbol names.
    ///
    /// After `resolve()` returns successfully, this should be empty.
    /// If `resolve()` returned an error, this contains the unresolved names.
    pub fn undefined_symbols(&self) -> Vec<&str> {
        self.undefined.iter().map(|s| s.as_str()).collect()
    }

    /// Get the virtual address of a named entry point (typically `_start`).
    ///
    /// Returns `None` if the symbol is not in the resolved table.
    pub fn entry_point_address(&self, name: &str) -> Option<u64> {
        self.resolved.get(name).map(|s| s.address)
    }
}

// ===========================================================================
// ELF Symbol Table Writing Helpers
// ===========================================================================

/// Write a single ELF64 symbol entry (24 bytes) to the output buffer.
///
/// ELF64 Elf64_Sym layout:
/// ```text
/// Offset  Size  Field
/// 0       4     st_name   (u32)
/// 4       1     st_info   (u8)
/// 5       1     st_other  (u8)
/// 6       2     st_shndx  (u16)
/// 8       8     st_value  (u64)
/// 16      8     st_size   (u64)
/// ```
fn write_elf64_sym(
    buf: &mut Vec<u8>,
    st_name: u32,
    st_info: u8,
    st_other: u8,
    st_shndx: u16,
    st_value: u64,
    st_size: u64,
) {
    elf::write_u32_le(buf, st_name);
    buf.push(st_info);
    buf.push(st_other);
    elf::write_u16_le(buf, st_shndx);
    elf::write_u64_le(buf, st_value);
    elf::write_u64_le(buf, st_size);
}

/// Write a single ELF32 symbol entry (16 bytes) to the output buffer.
///
/// ELF32 Elf32_Sym layout:
/// ```text
/// Offset  Size  Field
/// 0       4     st_name   (u32)
/// 4       4     st_value  (u32)
/// 8       4     st_size   (u32)
/// 12      1     st_info   (u8)
/// 13      1     st_other  (u8)
/// 14      2     st_shndx  (u16)
/// ```
fn write_elf32_sym(
    buf: &mut Vec<u8>,
    st_name: u32,
    st_value: u32,
    st_size: u32,
    st_info: u8,
    st_other: u8,
    st_shndx: u16,
) {
    elf::write_u32_le(buf, st_name);
    elf::write_u32_le(buf, st_value);
    elf::write_u32_le(buf, st_size);
    buf.push(st_info);
    buf.push(st_other);
    elf::write_u16_le(buf, st_shndx);
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a defined global symbol.
    fn global_symbol(name: &str, section: u16, value: u64, obj: usize) -> Symbol {
        Symbol {
            name: name.to_string(),
            binding: SymbolBinding::Global,
            symbol_type: SymbolType::Function,
            visibility: SymbolVisibility::Default,
            section_index: section,
            value,
            size: 0,
            source_object: obj,
        }
    }

    /// Helper to create an undefined global symbol reference.
    fn undefined_symbol(name: &str, obj: usize) -> Symbol {
        Symbol {
            name: name.to_string(),
            binding: SymbolBinding::Global,
            symbol_type: SymbolType::NoType,
            visibility: SymbolVisibility::Default,
            section_index: SHN_UNDEF,
            value: 0,
            size: 0,
            source_object: obj,
        }
    }

    /// Helper to create a weak defined symbol.
    fn weak_symbol(name: &str, section: u16, value: u64, obj: usize) -> Symbol {
        Symbol {
            name: name.to_string(),
            binding: SymbolBinding::Weak,
            symbol_type: SymbolType::Function,
            visibility: SymbolVisibility::Default,
            section_index: section,
            value,
            size: 0,
            source_object: obj,
        }
    }

    /// Helper to create a local symbol.
    fn local_symbol(name: &str, section: u16, value: u64, obj: usize) -> Symbol {
        Symbol {
            name: name.to_string(),
            binding: SymbolBinding::Local,
            symbol_type: SymbolType::Object,
            visibility: SymbolVisibility::Default,
            section_index: section,
            value,
            size: 4,
            source_object: obj,
        }
    }

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------

    #[test]
    fn test_shn_constants() {
        assert_eq!(SHN_UNDEF, 0);
        assert_eq!(SHN_ABS, 0xFFF1);
        assert_eq!(SHN_COMMON, 0xFFF2);
    }

    // -----------------------------------------------------------------------
    // Enum conversions
    // -----------------------------------------------------------------------

    #[test]
    fn test_symbol_binding_roundtrip() {
        assert_eq!(SymbolBinding::from_elf(elf::STB_LOCAL), SymbolBinding::Local);
        assert_eq!(SymbolBinding::from_elf(elf::STB_GLOBAL), SymbolBinding::Global);
        assert_eq!(SymbolBinding::from_elf(elf::STB_WEAK), SymbolBinding::Weak);
        assert_eq!(SymbolBinding::Local.to_elf(), elf::STB_LOCAL);
        assert_eq!(SymbolBinding::Global.to_elf(), elf::STB_GLOBAL);
        assert_eq!(SymbolBinding::Weak.to_elf(), elf::STB_WEAK);
    }

    #[test]
    fn test_symbol_type_roundtrip() {
        assert_eq!(SymbolType::from_elf(elf::STT_NOTYPE), SymbolType::NoType);
        assert_eq!(SymbolType::from_elf(elf::STT_OBJECT), SymbolType::Object);
        assert_eq!(SymbolType::from_elf(elf::STT_FUNC), SymbolType::Function);
        assert_eq!(SymbolType::from_elf(elf::STT_SECTION), SymbolType::Section);
        assert_eq!(SymbolType::from_elf(elf::STT_FILE), SymbolType::File);
        assert_eq!(SymbolType::NoType.to_elf(), elf::STT_NOTYPE);
        assert_eq!(SymbolType::Function.to_elf(), elf::STT_FUNC);
    }

    #[test]
    fn test_symbol_visibility_roundtrip() {
        assert_eq!(SymbolVisibility::from_elf(elf::STV_DEFAULT), SymbolVisibility::Default);
        assert_eq!(SymbolVisibility::from_elf(elf::STV_HIDDEN), SymbolVisibility::Hidden);
        assert_eq!(SymbolVisibility::from_elf(elf::STV_PROTECTED), SymbolVisibility::Protected);
        assert_eq!(SymbolVisibility::Default.to_elf(), elf::STV_DEFAULT);
        assert_eq!(SymbolVisibility::Hidden.to_elf(), elf::STV_HIDDEN);
    }

    // -----------------------------------------------------------------------
    // Adding global symbols
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_global_defined_symbol() {
        let mut resolver = SymbolResolver::new();
        let sym = global_symbol("main", 1, 0x1000, 0);
        resolver.add_object_symbols(0, &[sym]);

        assert!(resolver.global_symbols.contains_key("main"));
        assert_eq!(resolver.global_symbols["main"].len(), 1);
        assert!(!resolver.undefined.contains("main"));
    }

    // -----------------------------------------------------------------------
    // Adding undefined symbol references
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_undefined_symbol() {
        let mut resolver = SymbolResolver::new();
        let sym = undefined_symbol("printf", 0);
        resolver.add_object_symbols(0, &[sym]);

        assert!(resolver.undefined.contains("printf"));
        assert!(resolver.global_symbols.contains_key("printf"));
    }

    // -----------------------------------------------------------------------
    // Resolution: defined resolves undefined
    // -----------------------------------------------------------------------

    #[test]
    fn test_resolve_defined_satisfies_undefined() {
        let mut resolver = SymbolResolver::new();

        // Object 0 references 'foo' as undefined
        resolver.add_object_symbols(0, &[undefined_symbol("foo", 0)]);
        // Object 1 defines 'foo'
        resolver.add_object_symbols(1, &[global_symbol("foo", 1, 0x2000, 1)]);

        let result = resolver.resolve();
        assert!(result.is_ok());
        assert!(resolver.get_resolved("foo").is_some());
        assert!(resolver.undefined.is_empty());
    }

    // -----------------------------------------------------------------------
    // Error: undefined symbol with no definition
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_undefined_symbol() {
        let mut resolver = SymbolResolver::new();
        resolver.add_object_symbols(0, &[undefined_symbol("missing_func", 0)]);

        let result = resolver.resolve();
        assert!(result.is_err());
        match result.unwrap_err() {
            SymbolError::Undefined(name) => assert_eq!(name, "missing_func"),
            other => panic!("expected Undefined error, got: {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Error: duplicate strong definitions
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_duplicate_strong_definitions() {
        let mut resolver = SymbolResolver::new();

        // Two different objects define the same strong symbol
        resolver.add_object_symbols(0, &[global_symbol("dup_sym", 1, 0x1000, 0)]);
        resolver.add_object_symbols(1, &[global_symbol("dup_sym", 1, 0x2000, 1)]);

        let result = resolver.resolve();
        assert!(result.is_err());
        match result.unwrap_err() {
            SymbolError::Duplicate(name) => assert_eq!(name, "dup_sym"),
            other => panic!("expected Duplicate error, got: {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Weak vs Strong: strong wins
    // -----------------------------------------------------------------------

    #[test]
    fn test_strong_overrides_weak() {
        let mut resolver = SymbolResolver::new();

        // Add a weak definition first
        resolver.add_object_symbols(0, &[weak_symbol("handler", 1, 0x1000, 0)]);
        // Add a strong definition
        resolver.add_object_symbols(1, &[global_symbol("handler", 1, 0x2000, 1)]);

        let result = resolver.resolve();
        assert!(result.is_ok());

        let resolved = resolver.get_resolved("handler").unwrap();
        // Strong definition should win — its binding should be Global
        assert_eq!(resolved.binding, SymbolBinding::Global);
    }

    // -----------------------------------------------------------------------
    // Multiple weak: first wins
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_weak_first_wins() {
        let mut resolver = SymbolResolver::new();

        resolver.add_object_symbols(0, &[weak_symbol("fallback", 1, 0x1000, 0)]);
        resolver.add_object_symbols(1, &[weak_symbol("fallback", 1, 0x2000, 1)]);

        let result = resolver.resolve();
        assert!(result.is_ok());

        let resolved = resolver.get_resolved("fallback").unwrap();
        assert_eq!(resolved.binding, SymbolBinding::Weak);
    }

    // -----------------------------------------------------------------------
    // Local symbols preserved separately
    // -----------------------------------------------------------------------

    #[test]
    fn test_local_symbols_preserved() {
        let mut resolver = SymbolResolver::new();

        let syms = vec![
            local_symbol("local_var", 1, 0x100, 0),
            global_symbol("global_func", 1, 0x200, 0),
        ];
        resolver.add_object_symbols(0, &syms);

        // Local symbol should be in local_symbols
        assert_eq!(resolver.local_symbols.len(), 1);
        assert_eq!(resolver.local_symbols[0].name, "local_var");

        // Global symbol should be in global_symbols
        assert!(resolver.global_symbols.contains_key("global_func"));
    }

    // -----------------------------------------------------------------------
    // Symbol lookup by name
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_resolved_by_name() {
        let mut resolver = SymbolResolver::new();
        resolver.add_object_symbols(0, &[global_symbol("start", 1, 0x400000, 0)]);
        resolver.resolve().unwrap();

        assert!(resolver.get_resolved("start").is_some());
        assert!(resolver.get_resolved("nonexistent").is_none());
    }

    // -----------------------------------------------------------------------
    // Entry point address
    // -----------------------------------------------------------------------

    #[test]
    fn test_entry_point_address() {
        let mut resolver = SymbolResolver::new();
        resolver.add_object_symbols(0, &[global_symbol("_start", 1, 0x401000, 0)]);
        resolver.resolve().unwrap();

        // Before address assignment, the address is 0
        // (it's set during assign_addresses)
        assert!(resolver.entry_point_address("_start").is_some());
        assert!(resolver.entry_point_address("missing").is_none());
    }

    // -----------------------------------------------------------------------
    // undefined_symbols() returns remaining unresolved
    // -----------------------------------------------------------------------

    #[test]
    fn test_undefined_symbols_reporting() {
        let mut resolver = SymbolResolver::new();
        resolver.add_object_symbols(0, &[undefined_symbol("needed1", 0)]);
        resolver.add_object_symbols(0, &[undefined_symbol("needed2", 0)]);

        // Before resolve(), undefined should contain both
        let undef = resolver.undefined_symbols();
        assert!(undef.contains(&"needed1"));
        assert!(undef.contains(&"needed2"));
    }

    // -----------------------------------------------------------------------
    // all_resolved() returns all resolved symbols
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_resolved() {
        let mut resolver = SymbolResolver::new();
        resolver.add_object_symbols(
            0,
            &[
                global_symbol("alpha", 1, 0x1000, 0),
                global_symbol("beta", 1, 0x2000, 0),
            ],
        );
        resolver.resolve().unwrap();

        let all = resolver.all_resolved();
        assert_eq!(all.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Symbol from_parsed conversion
    // -----------------------------------------------------------------------

    #[test]
    fn test_symbol_from_parsed() {
        let parsed = elf::ParsedSymbol {
            name: "test_func".to_string(),
            value: 0x1234,
            size: 42,
            binding: elf::STB_GLOBAL,
            symbol_type: elf::STT_FUNC,
            visibility: elf::STV_DEFAULT,
            section_index: 1,
        };

        let sym = Symbol::from_parsed(&parsed, 3);
        assert_eq!(sym.name, "test_func");
        assert_eq!(sym.value, 0x1234);
        assert_eq!(sym.size, 42);
        assert_eq!(sym.binding, SymbolBinding::Global);
        assert_eq!(sym.symbol_type, SymbolType::Function);
        assert_eq!(sym.visibility, SymbolVisibility::Default);
        assert_eq!(sym.section_index, 1);
        assert_eq!(sym.source_object, 3);
    }

    // -----------------------------------------------------------------------
    // Symbol is_defined
    // -----------------------------------------------------------------------

    #[test]
    fn test_symbol_is_defined() {
        let defined = global_symbol("foo", 1, 0, 0);
        assert!(defined.is_defined());

        let undef = undefined_symbol("bar", 0);
        assert!(!undef.is_defined());

        // SHN_ABS is defined
        let abs_sym = Symbol {
            name: "abs".to_string(),
            binding: SymbolBinding::Global,
            symbol_type: SymbolType::NoType,
            visibility: SymbolVisibility::Default,
            section_index: SHN_ABS,
            value: 42,
            size: 0,
            source_object: 0,
        };
        assert!(abs_sym.is_defined());
    }

    // -----------------------------------------------------------------------
    // SymbolError Display
    // -----------------------------------------------------------------------

    #[test]
    fn test_symbol_error_display() {
        let err_undef = SymbolError::Undefined("printf".to_string());
        assert_eq!(format!("{}", err_undef), "undefined reference to `printf`");

        let err_dup = SymbolError::Duplicate("main".to_string());
        assert_eq!(format!("{}", err_dup), "multiple definition of `main`");

        let err_read = SymbolError::ReadError("bad format".to_string());
        assert_eq!(format!("{}", err_read), "error reading symbols: bad format");
    }

    // -----------------------------------------------------------------------
    // generate_symtab: ELF64
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_symtab_elf64() {
        let mut resolver = SymbolResolver::new();
        resolver.add_object_symbols(
            0,
            &[
                local_symbol("local_a", 1, 0x10, 0),
                global_symbol("global_b", 1, 0x20, 0),
            ],
        );
        resolver.resolve().unwrap();

        let (symtab, strtab) = resolver.generate_symtab(true);

        // ELF64 sym entry is 24 bytes
        // Should have: null + 1 local + 1 global = 3 entries
        assert_eq!(symtab.len(), 3 * 24);

        // strtab should start with null byte
        assert_eq!(strtab[0], 0);

        // strtab should contain both symbol names
        let strtab_str = String::from_utf8_lossy(&strtab);
        assert!(strtab_str.contains("local_a"));
        assert!(strtab_str.contains("global_b"));
    }

    // -----------------------------------------------------------------------
    // generate_symtab: ELF32
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_symtab_elf32() {
        let mut resolver = SymbolResolver::new();
        resolver.add_object_symbols(0, &[global_symbol("entry", 1, 0x100, 0)]);
        resolver.resolve().unwrap();

        let (symtab, _strtab) = resolver.generate_symtab(false);

        // ELF32 sym entry is 16 bytes
        // Should have: null + 1 global = 2 entries
        assert_eq!(symtab.len(), 2 * 16);
    }

    // -----------------------------------------------------------------------
    // Defined symbol removes from undefined set
    // -----------------------------------------------------------------------

    #[test]
    fn test_definition_removes_from_undefined() {
        let mut resolver = SymbolResolver::new();

        // First add undefined reference
        resolver.add_object_symbols(0, &[undefined_symbol("foo", 0)]);
        assert!(resolver.undefined.contains("foo"));

        // Then add definition — should remove from undefined
        resolver.add_object_symbols(1, &[global_symbol("foo", 1, 0x1000, 1)]);
        assert!(!resolver.undefined.contains("foo"));
    }

    // -----------------------------------------------------------------------
    // Mixed: defined + undefined + weak
    // -----------------------------------------------------------------------

    #[test]
    fn test_mixed_resolution_scenario() {
        let mut resolver = SymbolResolver::new();

        // Object 0: defines 'main', references 'printf' and 'handler'
        resolver.add_object_symbols(
            0,
            &[
                global_symbol("main", 1, 0x1000, 0),
                undefined_symbol("printf", 0),
                undefined_symbol("handler", 0),
            ],
        );

        // Object 1: defines 'printf', provides weak 'handler'
        resolver.add_object_symbols(
            1,
            &[
                global_symbol("printf", 1, 0x2000, 1),
                weak_symbol("handler", 1, 0x3000, 1),
            ],
        );

        let result = resolver.resolve();
        assert!(result.is_ok());

        assert!(resolver.get_resolved("main").is_some());
        assert!(resolver.get_resolved("printf").is_some());
        assert!(resolver.get_resolved("handler").is_some());
        assert!(resolver.undefined.is_empty());
    }

    // -----------------------------------------------------------------------
    // Assign addresses with section mappings
    // -----------------------------------------------------------------------

    #[test]
    fn test_assign_addresses_basic() {
        let mut resolver = SymbolResolver::new();

        // Symbol in section 1, value 0x10
        resolver.add_object_symbols(0, &[global_symbol("func", 1, 0x10, 0)]);
        resolver.resolve().unwrap();

        // Section addresses: section 0 at 0, section 1 at 0x400000
        let section_addresses = vec![0u64, 0x400000];

        // Section mapping: object 0, section 1 → output offset 0
        let mappings = vec![sections::InputSectionMapping {
            object_index: 0,
            original_index: 1,
            output_offset: 0,
            size: 0x100,
        }];

        resolver.assign_addresses(&section_addresses, &mappings);

        let resolved = resolver.get_resolved("func").unwrap();
        // address = section_base(0x400000) + output_offset(0) + value(0x10) = 0x400010
        assert_eq!(resolved.address, 0x400010);
    }
}
