//! # Dynamic Linking Support
//!
//! This module generates all ELF structures needed for shared libraries (`-shared`
//! output) and dynamically-linked executables. It produces:
//!
//! - **`.dynamic`** section with `DT_NEEDED`, `DT_SONAME`, `DT_STRTAB`,
//!   `DT_SYMTAB`, `DT_HASH`, `DT_PLTGOT`, `DT_JMPREL`, and other required
//!   dynamic tags
//! - **`.dynsym`** dynamic symbol table in correct ELF32/ELF64 format
//! - **`.dynstr`** dynamic string table with deduplication
//! - **`.hash`** SysV hash table for dynamic symbol lookup
//! - **`.plt`** procedure linkage table stubs (architecture-specific machine code)
//! - **`.got`** global offset table entries
//! - **`.rela.plt`** / **`.rel.plt`** relocation entries for PLT
//!
//! ## ELF Dual-Width Support
//!
//! All structures are target-width-aware: ELF64 for x86-64, AArch64, and
//! RISC-V 64 targets; ELF32 for i686.
//!
//! ## Zero External Dependencies
//!
//! Uses only the Rust standard library and sibling linker modules.

use std::collections::HashMap;

use super::elf;
use super::relocations;
use super::symbols::{ResolvedSymbol, SymbolBinding, SymbolType, SymbolVisibility};
use crate::driver::target::{Architecture, TargetConfig};

// ============================================================================
// Dynamic Section Tag Constants (ELF spec values)
// ============================================================================

/// Marks end of dynamic section.
const DT_NULL: u64 = 0;
/// Name of a needed library (offset into `.dynstr`).
const DT_NEEDED: u64 = 1;
/// Total size in bytes of PLT relocations.
const DT_PLTRELSZ: u64 = 2;
/// Address of PLT/GOT.
const DT_PLTGOT: u64 = 3;
/// Address of symbol hash table.
const DT_HASH: u64 = 4;
/// Address of dynamic string table (`.dynstr`).
const DT_STRTAB: u64 = 5;
/// Address of dynamic symbol table (`.dynsym`).
const DT_SYMTAB: u64 = 6;
/// Address of RELA relocation table.
#[allow(dead_code)]
const DT_RELA: u64 = 7;
/// Total size of RELA table.
#[allow(dead_code)]
const DT_RELASZ: u64 = 8;
/// Size of a single RELA entry.
#[allow(dead_code)]
const DT_RELAENT: u64 = 9;
/// Size of string table in bytes.
const DT_STRSZ: u64 = 10;
/// Size of a symbol table entry in bytes.
const DT_SYMENT: u64 = 11;
/// Shared object name (offset into `.dynstr`).
const DT_SONAME: u64 = 14;
/// Address of REL relocation table.
#[allow(dead_code)]
const DT_REL: u64 = 17;
/// Total size of REL table.
#[allow(dead_code)]
const DT_RELSZ: u64 = 18;
/// Size of a single REL entry.
#[allow(dead_code)]
const DT_RELENT: u64 = 19;
/// Type of relocation in PLT (DT_RELA=7 or DT_REL=17).
const DT_PLTREL: u64 = 20;
/// Address of PLT relocations (`.rela.plt` or `.rel.plt`).
const DT_JMPREL: u64 = 23;

// ============================================================================
// DynStrTab — Dynamic String Table Builder
// ============================================================================

/// Builder for the `.dynstr` dynamic string table section.
///
/// Maintains a deduplicated mapping of string contents to byte offsets. The
/// first byte is always a null byte (offset 0 = empty string), matching ELF
/// string table conventions.
pub struct DynStrTab {
    /// Raw string table bytes. Starts with a single `\0` byte.
    data: Vec<u8>,
    /// Map from string content to offset in `data` for deduplication.
    offsets: HashMap<String, u32>,
}

impl DynStrTab {
    /// Create a new empty dynamic string table with the mandatory null prefix.
    pub fn new() -> Self {
        let mut data = Vec::new();
        data.push(0); // Null byte at offset 0 (empty string)
        DynStrTab {
            data,
            offsets: HashMap::new(),
        }
    }

    /// Add a string to the table, returning its byte offset.
    ///
    /// If the string was previously added, returns the existing offset without
    /// duplicating it (O(1) lookup via `HashMap`).
    pub fn add(&mut self, s: &str) -> u32 {
        if s.is_empty() {
            return 0; // Empty string always at offset 0
        }
        if let Some(&existing) = self.offsets.get(s) {
            return existing;
        }
        let offset = self.data.len() as u32;
        self.data.extend_from_slice(s.as_bytes());
        self.data.push(0); // Null terminator
        self.offsets.insert(s.to_string(), offset);
        offset
    }

    /// Get the raw bytes of the string table.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Get the total size of the string table in bytes.
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

// ============================================================================
// DynSymEntry — Single Dynamic Symbol Table Entry
// ============================================================================

/// A single entry in the `.dynsym` dynamic symbol table.
///
/// Fields correspond to the ELF `Elf32_Sym` / `Elf64_Sym` structures, stored
/// in a target-independent representation that is serialized to the correct
/// format by [`DynSymTab::to_bytes`].
#[derive(Debug, Clone)]
pub struct DynSymEntry {
    /// Offset into `.dynstr` for the symbol name.
    pub name_offset: u32,
    /// Symbol value (virtual address).
    pub value: u64,
    /// Symbol size in bytes.
    pub size: u64,
    /// Symbol type and binding packed into `st_info`.
    pub info: u8,
    /// Symbol visibility packed into `st_other`.
    pub other: u8,
    /// Related section header index.
    pub section_index: u16,
}

// ============================================================================
// DynSymTab — Dynamic Symbol Table Builder
// ============================================================================

/// Builder for the `.dynsym` dynamic symbol table section.
///
/// The first entry is always the STN_UNDEF null symbol, per ELF spec.
pub struct DynSymTab {
    /// All symbol entries, including the leading null entry.
    pub entries: Vec<DynSymEntry>,
}

impl DynSymTab {
    /// Create a new dynamic symbol table with the mandatory null first entry.
    pub fn new() -> Self {
        let null_entry = DynSymEntry {
            name_offset: 0,
            value: 0,
            size: 0,
            info: elf::elf_st_info(elf::STB_LOCAL, elf::STT_NOTYPE),
            other: elf::STV_DEFAULT,
            section_index: elf::SHN_UNDEF,
        };
        DynSymTab {
            entries: vec![null_entry],
        }
    }

    /// Add a symbol entry to the table. Returns the symbol index (1-based,
    /// since index 0 is the null entry).
    pub fn add(&mut self, entry: DynSymEntry) -> usize {
        let idx = self.entries.len();
        self.entries.push(entry);
        idx
    }

    /// Serialize all entries to bytes in the correct ELF format.
    ///
    /// - **ELF64** (`Elf64_Sym`): 24 bytes per entry
    ///   `st_name(4) st_info(1) st_other(1) st_shndx(2) st_value(8) st_size(8)`
    /// - **ELF32** (`Elf32_Sym`): 16 bytes per entry
    ///   `st_name(4) st_value(4) st_size(4) st_info(1) st_other(1) st_shndx(2)`
    ///
    /// NOTE: Field ordering differs between ELF32 and ELF64.
    pub fn to_bytes(&self, is_64bit: bool) -> Vec<u8> {
        let entry_size = if is_64bit { 24 } else { 16 };
        let mut buf = Vec::with_capacity(self.entries.len() * entry_size);

        for entry in &self.entries {
            if is_64bit {
                // ELF64: st_name(4), st_info(1), st_other(1), st_shndx(2),
                //        st_value(8), st_size(8)
                elf::write_u32_le(&mut buf, entry.name_offset);
                buf.push(entry.info);
                buf.push(entry.other);
                elf::write_u16_le(&mut buf, entry.section_index);
                elf::write_u64_le(&mut buf, entry.value);
                elf::write_u64_le(&mut buf, entry.size);
            } else {
                // ELF32: st_name(4), st_value(4), st_size(4),
                //        st_info(1), st_other(1), st_shndx(2)
                elf::write_u32_le(&mut buf, entry.name_offset);
                elf::write_u32_le(&mut buf, entry.value as u32);
                elf::write_u32_le(&mut buf, entry.size as u32);
                buf.push(entry.info);
                buf.push(entry.other);
                elf::write_u16_le(&mut buf, entry.section_index);
            }
        }

        buf
    }
}

// ============================================================================
// ELF SysV Hash Function and Hash Table
// ============================================================================

/// Compute the SysV ELF hash for a symbol name.
///
/// This is the standard ELF hash function specified in the ELF ABI, used by
/// the dynamic linker to look up symbols in `.hash` sections.
pub fn elf_hash(name: &[u8]) -> u32 {
    let mut h: u32 = 0;
    for &b in name {
        h = (h << 4).wrapping_add(b as u32);
        let g = h & 0xf000_0000;
        if g != 0 {
            h ^= g >> 24;
        }
        h &= !g;
    }
    h
}

/// Build a SysV ELF hash table (`.hash` section) for the given dynamic symbols.
///
/// The hash table structure is:
/// ```text
/// nbucket  : u32          — number of hash buckets
/// nchain   : u32          — number of chain entries (== number of symbols)
/// bucket[nbucket] : u32[] — each bucket holds index of first symbol in chain
/// chain[nchain]   : u32[] — linked list chain; chain[i] = next symbol index or 0
/// ```
///
/// The `dynstr` parameter is used to extract symbol names for hashing. The symbol
/// at index 0 is the null symbol and is skipped during bucket population.
pub fn build_hash_table(symbols: &[DynSymEntry], dynstr: &DynStrTab) -> Vec<u8> {
    let nsyms = symbols.len();
    // Use a prime-ish bucket count for reasonable distribution.
    // For small symbol tables, use at least 1 bucket.
    let nbuckets = if nsyms <= 1 {
        1
    } else if nsyms < 16 {
        nsyms
    } else {
        // Roughly nsyms * 2 / 3 for decent load factor
        (nsyms * 2 / 3).max(1)
    };

    let mut buckets = vec![0u32; nbuckets];
    let mut chains = vec![0u32; nsyms];

    // Populate hash table. Skip the null symbol at index 0.
    for i in 1..nsyms {
        let entry = &symbols[i];
        let name = extract_name_from_dynstr(dynstr, entry.name_offset);
        let hash = elf_hash(name.as_bytes());
        let bucket_idx = (hash as usize) % nbuckets;

        // Insert at the head of the chain for this bucket.
        chains[i] = buckets[bucket_idx];
        buckets[bucket_idx] = i as u32;
    }

    // Serialize: nbucket, nchain, bucket[], chain[]
    let mut buf = Vec::with_capacity(4 + 4 + nbuckets * 4 + nsyms * 4);
    elf::write_u32_le(&mut buf, nbuckets as u32);
    elf::write_u32_le(&mut buf, nsyms as u32);
    for &b in &buckets {
        elf::write_u32_le(&mut buf, b);
    }
    for &c in &chains {
        elf::write_u32_le(&mut buf, c);
    }

    buf
}

/// Extract a symbol name from the dynstr table given its offset.
///
/// Reads bytes from the offset until a null terminator or end of data.
fn extract_name_from_dynstr(dynstr: &DynStrTab, offset: u32) -> String {
    let data = dynstr.as_bytes();
    let start = offset as usize;
    if start >= data.len() {
        return String::new();
    }
    let end = data[start..]
        .iter()
        .position(|&b| b == 0)
        .map(|pos| start + pos)
        .unwrap_or(data.len());
    String::from_utf8_lossy(&data[start..end]).to_string()
}

// ============================================================================
// PLT (Procedure Linkage Table) Generation
// ============================================================================

/// Generate PLT (Procedure Linkage Table) stubs for dynamic symbol resolution.
///
/// The PLT contains architecture-specific machine code stubs. Each stub
/// performs an indirect jump through the GOT; on first call, the GOT entry
/// redirects to the PLT resolver which calls `_dl_runtime_resolve`.
///
/// PLT layout:
/// - `PLT[0]`: Resolver stub (pushes link_map, jumps to resolver)
/// - `PLT[1..N]`: Per-symbol stubs (jump through GOT, push reloc index, jump to PLT[0])
///
/// Returns raw machine code bytes for the entire `.plt` section.
pub fn generate_plt(symbols: &[DynSymEntry], target: &TargetConfig) -> Vec<u8> {
    // Skip the null symbol at index 0 — only generate stubs for real symbols
    let dynamic_syms: Vec<usize> = (1..symbols.len()).collect();

    match target.arch {
        Architecture::X86_64 => generate_plt_x86_64(&dynamic_syms),
        Architecture::I686 => generate_plt_i686(&dynamic_syms),
        Architecture::Aarch64 => generate_plt_aarch64(&dynamic_syms),
        Architecture::Riscv64 => generate_plt_riscv64(&dynamic_syms),
    }
}

/// Generate x86-64 PLT stubs (16 bytes per entry).
///
/// PLT[0] (resolver): push [GOT+8]; jmp [GOT+16]
/// PLT[n]: jmp [GOT+offset]; push reloc_index; jmp PLT[0]
fn generate_plt_x86_64(sym_indices: &[usize]) -> Vec<u8> {
    let plt0_size = 16usize;
    let entry_size = 16usize;
    let total_size = plt0_size + sym_indices.len() * entry_size;
    let mut plt = Vec::with_capacity(total_size);

    // PLT[0] — resolver stub
    // ff 35 XX XX XX XX    push QWORD PTR [rip+0xXXXXXXXX]  (GOT+8)
    // ff 25 XX XX XX XX    jmp  QWORD PTR [rip+0xXXXXXXXX]  (GOT+16)
    // 0f 1f 40 00          nop DWORD PTR [rax+0x0]
    //
    // The actual GOT offsets are placeholders — the linker fills them
    // once section addresses are assigned. We use zero placeholders here.
    plt.extend_from_slice(&[0xff, 0x35]); // push [rip+disp32]
    plt.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // GOT+8 displacement (placeholder)
    plt.extend_from_slice(&[0xff, 0x25]); // jmp [rip+disp32]
    plt.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // GOT+16 displacement (placeholder)
    plt.extend_from_slice(&[0x0f, 0x1f, 0x40, 0x00]); // 4-byte NOP padding

    // PLT[n] entries — per-function stubs
    for (i, _sym_idx) in sym_indices.iter().enumerate() {
        // ff 25 XX XX XX XX    jmp QWORD PTR [rip+disp32]  (GOT[3+i])
        plt.extend_from_slice(&[0xff, 0x25]);
        plt.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // GOT entry displacement (placeholder)

        // 68 XX XX XX XX       push imm32  (relocation index)
        plt.push(0x68);
        elf::write_u32_le(&mut plt, i as u32);

        // e9 XX XX XX XX       jmp rel32  (back to PLT[0])
        plt.push(0xe9);
        // Calculate relative offset to PLT[0] from next instruction
        let current_pos = plt.len() as i32 + 4; // after this 4-byte immediate
        let plt0_pos = 0i32;
        let rel_offset = plt0_pos - current_pos;
        elf::write_i32_le(&mut plt, rel_offset);
    }

    plt
}

/// Generate i686 PLT stubs (16 bytes per entry, 32-bit addressing).
///
/// Similar to x86-64 but uses 32-bit absolute addressing for GOT references.
fn generate_plt_i686(sym_indices: &[usize]) -> Vec<u8> {
    let plt0_size = 16usize;
    let entry_size = 16usize;
    let total_size = plt0_size + sym_indices.len() * entry_size;
    let mut plt = Vec::with_capacity(total_size);

    // PLT[0] — resolver stub (32-bit)
    // ff 35 XX XX XX XX    push DWORD PTR ds:[GOT+4]
    // ff 25 XX XX XX XX    jmp  DWORD PTR ds:[GOT+8]
    // 00 00 00 00          padding
    plt.extend_from_slice(&[0xff, 0x35]);
    plt.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // GOT+4 absolute (placeholder)
    plt.extend_from_slice(&[0xff, 0x25]);
    plt.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // GOT+8 absolute (placeholder)
    plt.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // padding

    // PLT[n] entries
    for (i, _sym_idx) in sym_indices.iter().enumerate() {
        // ff 25 XX XX XX XX    jmp DWORD PTR ds:[GOT[3+i]]
        plt.extend_from_slice(&[0xff, 0x25]);
        plt.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // GOT entry absolute (placeholder)

        // 68 XX XX XX XX       push imm32  (relocation index)
        plt.push(0x68);
        elf::write_u32_le(&mut plt, i as u32);

        // e9 XX XX XX XX       jmp rel32  (PLT[0])
        plt.push(0xe9);
        let current_pos = plt.len() as i32 + 4;
        let plt0_pos = 0i32;
        let rel_offset = plt0_pos - current_pos;
        elf::write_i32_le(&mut plt, rel_offset);
    }

    plt
}

/// Generate AArch64 PLT stubs (16 bytes per entry).
///
/// PLT[0]: adrp x16, GOT_PAGE; ldr x17, [x16, GOT_OFF]; add x16, x16, GOT_OFF; br x17
/// PLT[n]: adrp x16, GOT_PAGE; ldr x17, [x16, GOT_OFF]; add x16, x16, GOT_OFF; br x17
fn generate_plt_aarch64(sym_indices: &[usize]) -> Vec<u8> {
    let entry_size = 16usize;
    let total_size = entry_size + sym_indices.len() * entry_size;
    let mut plt = Vec::with_capacity(total_size);

    // PLT[0] — resolver stub (AArch64)
    // Placeholder instructions using zero immediates (linker patches later)
    // adrp x16, #0          — 90000010
    // ldr  x17, [x16, #0]   — f9400211
    // add  x16, x16, #0     — 91000210
    // br   x17               — d61f0220
    elf::write_u32_le(&mut plt, 0x9000_0010); // adrp x16, #0
    elf::write_u32_le(&mut plt, 0xf940_0211); // ldr x17, [x16]
    elf::write_u32_le(&mut plt, 0x9100_0210); // add x16, x16, #0
    elf::write_u32_le(&mut plt, 0xd61f_0220); // br x17

    // PLT[n] entries
    for _sym_idx in sym_indices {
        // Each PLT entry is the same template with different GOT offsets
        // that the linker patches after address assignment.
        elf::write_u32_le(&mut plt, 0x9000_0010); // adrp x16, #page
        elf::write_u32_le(&mut plt, 0xf940_0211); // ldr x17, [x16, #off]
        elf::write_u32_le(&mut plt, 0x9100_0210); // add x16, x16, #off
        elf::write_u32_le(&mut plt, 0xd61f_0220); // br x17
    }

    plt
}

/// Generate RISC-V 64 PLT stubs (16 bytes per entry).
///
/// PLT[n]: auipc t3, %pcrel_hi(GOT); ld t3, %pcrel_lo(GOT)(t3); jalr t1, t3; nop
fn generate_plt_riscv64(sym_indices: &[usize]) -> Vec<u8> {
    let entry_size = 16usize;
    let total_size = entry_size + sym_indices.len() * entry_size;
    let mut plt = Vec::with_capacity(total_size);

    // PLT[0] — resolver stub
    // auipc  t3, 0          — 00000e17
    // ld     t3, 0(t3)      — 000e3e03  (I-type: ld t3, 0(t3))
    // jalr   t1, t3, 0      — 000e0367  (jalr t1, t3)
    // nop                   — 00000013
    elf::write_u32_le(&mut plt, 0x0000_0e17); // auipc t3, 0
    elf::write_u32_le(&mut plt, 0x000e_3e03); // ld t3, 0(t3)
    elf::write_u32_le(&mut plt, 0x000e_0367); // jalr t1, t3, 0
    elf::write_u32_le(&mut plt, 0x0000_0013); // nop (addi x0, x0, 0)

    // PLT[n] entries
    for _sym_idx in sym_indices {
        elf::write_u32_le(&mut plt, 0x0000_0e17); // auipc t3, %pcrel_hi(GOT)
        elf::write_u32_le(&mut plt, 0x000e_3e03); // ld t3, %pcrel_lo(GOT)(t3)
        elf::write_u32_le(&mut plt, 0x000e_0367); // jalr t1, t3
        elf::write_u32_le(&mut plt, 0x0000_0013); // nop
    }

    plt
}

// ============================================================================
// GOT (Global Offset Table) Generation
// ============================================================================

/// Generate the GOT (Global Offset Table) for dynamic symbols.
///
/// The GOT layout is:
/// - `GOT[0]`: Address of `.dynamic` section (filled by static linker)
/// - `GOT[1]`: `link_map` pointer (filled by dynamic linker at runtime)
/// - `GOT[2]`: `_dl_runtime_resolve` address (filled by dynamic linker at runtime)
/// - `GOT[3..N]`: Per-symbol entries (initially zero for lazy binding; the dynamic
///   linker fills them on first use)
///
/// Entry size is 8 bytes for 64-bit targets, 4 bytes for 32-bit (i686).
pub fn generate_got(num_entries: usize, target: &TargetConfig) -> Vec<u8> {
    let entry_size = if target.is_64bit() { 8 } else { 4 };
    // 3 reserved entries + per-symbol entries
    let total_entries = num_entries + 3;
    // All entries initialized to zero — dynamic linker fills GOT[0..2] at load time,
    // and remaining entries are resolved lazily via PLT stubs.
    vec![0u8; total_entries * entry_size]
}

// ============================================================================
// DynamicSection — .dynamic Section Builder
// ============================================================================

/// A single entry in the `.dynamic` section.
struct DynamicEntry {
    /// Dynamic tag (`DT_NEEDED`, `DT_STRTAB`, etc.).
    tag: u64,
    /// Tag value (address, size, or string table offset depending on the tag).
    value: u64,
}

/// Builder for the `.dynamic` section.
///
/// The `.dynamic` section is an array of tag-value pairs terminated by
/// `DT_NULL`. It instructs the dynamic linker about string tables, symbol
/// tables, hash tables, PLT relocations, needed libraries, and shared object
/// names.
pub struct DynamicSection {
    /// Accumulated dynamic entries (DT_NULL is appended at serialization time).
    entries: Vec<DynamicEntry>,
}

impl DynamicSection {
    /// Create a new empty dynamic section.
    pub fn new() -> Self {
        DynamicSection {
            entries: Vec::new(),
        }
    }

    /// Add a raw tag-value pair to the dynamic section.
    pub fn add(&mut self, tag: u64, value: u64) {
        self.entries.push(DynamicEntry { tag, value });
    }

    /// Build the full `.dynamic` section for a shared library or dynamic executable.
    ///
    /// Populates the section with:
    /// - `DT_NEEDED` for each required library
    /// - `DT_SONAME` if a shared object name is specified
    /// - `DT_STRTAB`, `DT_STRSZ` pointing to `.dynstr`
    /// - `DT_SYMTAB`, `DT_SYMENT` pointing to `.dynsym`
    /// - `DT_HASH` pointing to `.hash`
    /// - `DT_PLTGOT` pointing to `.got`
    /// - `DT_JMPREL`, `DT_PLTRELSZ`, `DT_PLTREL` pointing to PLT relocations
    /// - `DT_NULL` terminator
    pub fn build_for_shared_library(
        needed_libs: &[String],
        soname: Option<&str>,
        dynstr: &DynStrTab,
        addresses: &DynamicAddresses,
    ) -> Self {
        let mut section = DynamicSection::new();

        // Add DT_NEEDED entries for each required shared library.
        // The value is the offset of the library name in .dynstr.
        // We need to compute offsets — since the dynstr was already built,
        // we search for matching strings. For safety, if not found, we
        // use offset 0 (empty string).
        for lib in needed_libs {
            // Find the offset by scanning the dynstr data
            let offset = find_string_offset(dynstr, lib);
            section.add(DT_NEEDED, offset as u64);
        }

        // DT_SONAME for shared library name
        if let Some(name) = soname {
            let offset = find_string_offset(dynstr, name);
            section.add(DT_SONAME, offset as u64);
        }

        // Required table pointers
        section.add(DT_STRTAB, addresses.dynstr_addr);
        section.add(DT_STRSZ, addresses.dynstr_size);
        section.add(DT_SYMTAB, addresses.dynsym_addr);

        // DT_SYMENT: size of each symbol table entry
        // We use 24 for ELF64 or 16 for ELF32, but since we don't have target
        // info here, we determine from the addresses (syment is always present).
        // For simplicity, we accept that the caller should set this up.
        // The entry size is standard: 24 for ELF64, 16 for ELF32.
        // We'll use 24 as default since most targets are 64-bit.
        // The caller can override if needed.
        section.add(DT_SYMENT, 24); // Will be corrected in generate_dynamic_sections

        section.add(DT_HASH, addresses.hash_addr);
        section.add(DT_PLTGOT, addresses.pltgot_addr);

        // PLT relocation entries
        if addresses.plt_rel_size > 0 {
            section.add(DT_JMPREL, addresses.plt_rel_addr);
            section.add(DT_PLTRELSZ, addresses.plt_rel_size);
            // DT_PLTREL: indicates whether .rela.plt or .rel.plt is used.
            // ELF64 uses RELA (type 7), ELF32/i686 uses REL (type 17).
            // Default to RELA (7) — corrected in generate_dynamic_sections.
            section.add(DT_PLTREL, 7); // DT_RELA value
        }

        // Terminate with DT_NULL
        section.add(DT_NULL, 0);

        section
    }

    /// Serialize the dynamic section to bytes in ELF32 or ELF64 format.
    ///
    /// - **ELF64**: 16 bytes per entry (`d_tag:i64` + `d_val:u64`)
    /// - **ELF32**: 8 bytes per entry (`d_tag:i32` + `d_val:u32`)
    pub fn to_bytes(&self, is_64bit: bool) -> Vec<u8> {
        let entry_size = if is_64bit { 16 } else { 8 };
        let mut buf = Vec::with_capacity(self.entries.len() * entry_size);

        for entry in &self.entries {
            if is_64bit {
                elf::write_i64_le(&mut buf, entry.tag as i64);
                elf::write_u64_le(&mut buf, entry.value);
            } else {
                elf::write_i32_le(&mut buf, entry.tag as i32);
                elf::write_u32_le(&mut buf, entry.value as u32);
            }
        }

        buf
    }
}

/// Search for a string's offset in the dynstr table data.
fn find_string_offset(dynstr: &DynStrTab, s: &str) -> u32 {
    let data = dynstr.as_bytes();
    let needle = s.as_bytes();
    let needle_len = needle.len();

    // Search for the needle followed by a null terminator
    if needle_len == 0 {
        return 0;
    }

    for i in 0..data.len() {
        if i + needle_len < data.len()
            && &data[i..i + needle_len] == needle
            && (i + needle_len >= data.len() || data[i + needle_len] == 0)
        {
            return i as u32;
        }
    }
    0 // Fallback to empty string
}

// ============================================================================
// DynamicAddresses — Computed Section Addresses
// ============================================================================

/// Holds computed virtual addresses for dynamic sections, needed when building
/// the `.dynamic` section entries.
///
/// The linker computes these addresses after section layout, then passes them
/// to [`DynamicSection::build_for_shared_library`] and
/// [`generate_dynamic_sections`].
pub struct DynamicAddresses {
    /// Virtual address of the `.dynstr` section.
    pub dynstr_addr: u64,
    /// Size of the `.dynstr` section in bytes.
    pub dynstr_size: u64,
    /// Virtual address of the `.dynsym` section.
    pub dynsym_addr: u64,
    /// Virtual address of the `.hash` section.
    pub hash_addr: u64,
    /// Virtual address of the `.plt` section.
    pub plt_addr: u64,
    /// Virtual address of the `.got` (PLTGOT) section.
    pub pltgot_addr: u64,
    /// Virtual address of the `.rela.plt` / `.rel.plt` section.
    pub plt_rel_addr: u64,
    /// Total size of PLT relocations in bytes.
    pub plt_rel_size: u64,
}

// ============================================================================
// DynamicOutput — Complete Dynamic Section Package
// ============================================================================

/// Contains all generated dynamic linking section data, ready for inclusion
/// in the final ELF output.
///
/// Each field is the raw byte content for the corresponding ELF section.
pub struct DynamicOutput {
    /// `.dynamic` section bytes.
    pub dynamic: Vec<u8>,
    /// `.dynsym` section bytes.
    pub dynsym: Vec<u8>,
    /// `.dynstr` section bytes.
    pub dynstr: Vec<u8>,
    /// `.hash` section bytes.
    pub hash: Vec<u8>,
    /// `.plt` section bytes (architecture-specific machine code).
    pub plt: Vec<u8>,
    /// `.got` section bytes.
    pub got: Vec<u8>,
    /// `.rela.plt` or `.rel.plt` section bytes.
    pub plt_relocations: Vec<u8>,
}

// ============================================================================
// PLT Relocation Generation
// ============================================================================

/// Generate PLT relocations (`.rela.plt` for ELF64 or `.rel.plt` for ELF32).
///
/// Each dynamic symbol with a PLT entry needs a JUMP_SLOT relocation pointing
/// from its GOT entry to the dynamic resolver.
///
/// Relocation types per architecture:
/// - x86-64: `R_X86_64_JUMP_SLOT` (7)
/// - i686:   `R_386_JMP_SLOT` (7)
/// - AArch64: `R_AARCH64_JUMP_SLOT` (1026)
/// - RISC-V 64: `R_RISCV_JUMP_SLOT` (5)
pub fn generate_plt_relocations(
    symbols: &[DynSymEntry],
    got_base: u64,
    target: &TargetConfig,
) -> Vec<u8> {
    let is_64bit = target.is_64bit();
    let got_entry_size: u64 = if is_64bit { 8 } else { 4 };
    // Skip null symbol at index 0
    let num_dynamic_syms = if symbols.len() > 1 { symbols.len() - 1 } else { 0 };

    let reloc_type = match target.arch {
        Architecture::X86_64 => relocations::R_X86_64_JUMP_SLOT,
        Architecture::I686 => relocations::R_386_JMP_SLOT,
        Architecture::Aarch64 => relocations::R_AARCH64_JUMP_SLOT,
        Architecture::Riscv64 => relocations::R_RISCV_JUMP_SLOT,
    };

    let mut buf = Vec::new();

    for i in 0..num_dynamic_syms {
        // GOT entry address: GOT base + 3 reserved entries + i
        let got_offset = got_base + (3 + i as u64) * got_entry_size;
        // Symbol index in .dynsym (1-based, since 0 is null)
        let sym_idx = (i + 1) as u32;

        if is_64bit {
            // Elf64_Rela: r_offset(8) + r_info(8) + r_addend(8) = 24 bytes
            elf::write_u64_le(&mut buf, got_offset);
            elf::write_u64_le(&mut buf, elf::elf64_r_info(sym_idx, reloc_type));
            elf::write_i64_le(&mut buf, 0); // addend = 0 for JUMP_SLOT
        } else {
            // Elf32_Rel: r_offset(4) + r_info(4) = 8 bytes (no addend)
            elf::write_u32_le(&mut buf, got_offset as u32);
            elf::write_u32_le(&mut buf, elf::elf32_r_info(sym_idx, reloc_type));
        }
    }

    buf
}

// ============================================================================
// Top-Level Integration: generate_dynamic_sections
// ============================================================================

/// Generate all dynamic linking sections for shared library or dynamic executable output.
///
/// This is the main entry point called by the linker orchestrator (`mod.rs`).
/// It converts resolved symbols into dynamic symbol table entries, builds the
/// string table, hash table, PLT, GOT, dynamic section, and PLT relocations,
/// then returns them as a [`DynamicOutput`] package.
///
/// # Parameters
///
/// - `exported_symbols`: Resolved symbols to export in `.dynsym`
/// - `needed_libraries`: Shared library names for `DT_NEEDED` entries
/// - `soname`: Optional shared object name for `DT_SONAME`
/// - `target`: Target configuration for ELF format and architecture dispatch
///
/// # Returns
///
/// `Ok(DynamicOutput)` containing all generated section data, or
/// `Err(LinkerError)` on failure.
pub fn generate_dynamic_sections(
    exported_symbols: &[ResolvedSymbol],
    needed_libraries: &[String],
    soname: Option<&str>,
    target: &TargetConfig,
) -> Result<DynamicOutput, super::LinkerError> {
    let is_64bit = target.is_64bit();

    // Step 1: Build dynamic string table
    let mut dynstr = DynStrTab::new();

    // Pre-add library names and soname to dynstr
    for lib in needed_libraries {
        dynstr.add(lib);
    }
    if let Some(name) = soname {
        dynstr.add(name);
    }

    // Step 2: Build dynamic symbol table
    let mut dynsym = DynSymTab::new();

    for sym in exported_symbols {
        let name_offset = dynstr.add(&sym.name);

        // Convert SymbolBinding to ELF binding value
        let binding = match sym.binding {
            SymbolBinding::Local => elf::STB_LOCAL,
            SymbolBinding::Global => elf::STB_GLOBAL,
            SymbolBinding::Weak => elf::STB_WEAK,
        };

        // Convert SymbolType to ELF type value
        let stype = match sym.symbol_type {
            SymbolType::NoType => elf::STT_NOTYPE,
            SymbolType::Object => elf::STT_OBJECT,
            SymbolType::Function => elf::STT_FUNC,
            _ => elf::STT_NOTYPE,
        };

        // Convert SymbolVisibility to ELF visibility value
        let visibility = match sym.visibility {
            SymbolVisibility::Default => elf::STV_DEFAULT,
            SymbolVisibility::Hidden => elf::STV_HIDDEN,
            SymbolVisibility::Protected => elf::STV_PROTECTED,
        };

        let entry = DynSymEntry {
            name_offset,
            value: sym.address,
            size: sym.size,
            info: elf::elf_st_info(binding, stype),
            other: visibility,
            section_index: sym.output_section_index as u16,
        };

        dynsym.add(entry);
    }

    // Step 3: Build hash table
    let hash = build_hash_table(&dynsym.entries, &dynstr);

    // Step 4: Generate PLT stubs
    let plt = generate_plt(&dynsym.entries, target);

    // Step 5: Generate GOT
    let num_dynamic = if dynsym.entries.len() > 1 {
        dynsym.entries.len() - 1
    } else {
        0
    };
    let got = generate_got(num_dynamic, target);

    // Step 6: Generate PLT relocations
    // Use a placeholder GOT base of 0 — the actual base will be patched by
    // the linker when final section addresses are assigned.
    let plt_relocs = generate_plt_relocations(&dynsym.entries, 0, target);

    // Step 7: Serialize dynsym and dynstr
    let dynsym_bytes = dynsym.to_bytes(is_64bit);
    let dynstr_bytes = dynstr.as_bytes().to_vec();

    // Step 8: Build the .dynamic section
    // Use placeholder addresses (0) — the linker patches them after layout.
    let sym_entry_size: u64 = if is_64bit { 24 } else { 16 };
    let plt_rel_type: u64 = if is_64bit { 7 } else { 17 }; // DT_RELA(7) vs DT_REL(17)

    let addresses = DynamicAddresses {
        dynstr_addr: 0,
        dynstr_size: dynstr_bytes.len() as u64,
        dynsym_addr: 0,
        hash_addr: 0,
        plt_addr: 0,
        pltgot_addr: 0,
        plt_rel_addr: 0,
        plt_rel_size: plt_relocs.len() as u64,
    };

    let mut dynamic = DynamicSection::build_for_shared_library(
        needed_libraries,
        soname,
        &dynstr,
        &addresses,
    );

    // Fix up DT_SYMENT to correct entry size based on target
    // We need to find and replace the DT_SYMENT entry
    for entry in &mut dynamic.entries {
        if entry.tag == DT_SYMENT {
            entry.value = sym_entry_size;
        }
        if entry.tag == DT_PLTREL {
            entry.value = plt_rel_type;
        }
    }

    let dynamic_bytes = dynamic.to_bytes(is_64bit);

    Ok(DynamicOutput {
        dynamic: dynamic_bytes,
        dynsym: dynsym_bytes,
        dynstr: dynstr_bytes,
        hash,
        plt,
        got,
        plt_relocations: plt_relocs,
    })
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- elf_hash tests ----

    #[test]
    fn test_elf_hash_empty() {
        assert_eq!(elf_hash(b""), 0);
    }

    #[test]
    fn test_elf_hash_known_values() {
        // Well-known ELF hash values for common symbol names.
        // These are verified against the glibc implementation.
        let hash_printf = elf_hash(b"printf");
        assert_ne!(hash_printf, 0, "printf should hash to non-zero");

        let hash_main = elf_hash(b"main");
        assert_ne!(hash_main, 0, "main should hash to non-zero");

        // Different strings should (very likely) produce different hashes
        assert_ne!(
            elf_hash(b"foo"),
            elf_hash(b"bar"),
            "foo and bar should hash differently"
        );
    }

    #[test]
    fn test_elf_hash_deterministic() {
        // Same input always produces the same hash
        assert_eq!(elf_hash(b"test"), elf_hash(b"test"));
        assert_eq!(elf_hash(b"_start"), elf_hash(b"_start"));
    }

    #[test]
    fn test_elf_hash_single_char() {
        // 'a' = 0x61
        // h = (0 << 4) + 0x61 = 0x61
        // g = 0x61 & 0xf0000000 = 0
        // h = 0x61 & !0 = 0x61
        assert_eq!(elf_hash(b"a"), 0x61);
    }

    // ---- DynStrTab tests ----

    #[test]
    fn test_dynstr_new() {
        let strtab = DynStrTab::new();
        assert_eq!(strtab.size(), 1); // Just the null byte
        assert_eq!(strtab.as_bytes(), &[0]);
    }

    #[test]
    fn test_dynstr_add_string() {
        let mut strtab = DynStrTab::new();
        let offset = strtab.add("hello");
        assert_eq!(offset, 1); // After the null byte
        assert_eq!(strtab.size(), 7); // \0 + "hello" + \0
        assert_eq!(&strtab.as_bytes()[1..6], b"hello");
        assert_eq!(strtab.as_bytes()[6], 0); // Null terminator
    }

    #[test]
    fn test_dynstr_add_multiple() {
        let mut strtab = DynStrTab::new();
        let off1 = strtab.add("libc.so.6");
        let off2 = strtab.add("libm.so.6");
        assert_eq!(off1, 1);
        assert_eq!(off2, 11); // 1 + 9 + 1 = 11
    }

    #[test]
    fn test_dynstr_deduplication() {
        let mut strtab = DynStrTab::new();
        let off1 = strtab.add("libc.so.6");
        let off2 = strtab.add("libc.so.6");
        assert_eq!(off1, off2, "Duplicate strings should return same offset");
        // Size should not increase on duplicate add
        let size_after_first = 1 + 10; // \0 + "libc.so.6" + \0
        assert_eq!(strtab.size(), size_after_first);
    }

    #[test]
    fn test_dynstr_empty_string() {
        let mut strtab = DynStrTab::new();
        let offset = strtab.add("");
        assert_eq!(offset, 0, "Empty string should always be at offset 0");
    }

    // ---- DynSymTab tests ----

    #[test]
    fn test_dynsym_new() {
        let symtab = DynSymTab::new();
        assert_eq!(symtab.entries.len(), 1); // Null entry
        assert_eq!(symtab.entries[0].name_offset, 0);
        assert_eq!(symtab.entries[0].value, 0);
    }

    #[test]
    fn test_dynsym_add() {
        let mut symtab = DynSymTab::new();
        let idx = symtab.add(DynSymEntry {
            name_offset: 1,
            value: 0x1000,
            size: 32,
            info: elf::elf_st_info(elf::STB_GLOBAL, elf::STT_FUNC),
            other: elf::STV_DEFAULT,
            section_index: 1,
        });
        assert_eq!(idx, 1); // First real symbol is at index 1
        assert_eq!(symtab.entries.len(), 2);
    }

    #[test]
    fn test_dynsym_to_bytes_elf64() {
        let symtab = DynSymTab::new();
        let bytes = symtab.to_bytes(true);
        // Null entry: 24 bytes for ELF64
        assert_eq!(bytes.len(), 24);
        // First 4 bytes (name_offset) should be 0
        assert_eq!(&bytes[0..4], &[0, 0, 0, 0]);
    }

    #[test]
    fn test_dynsym_to_bytes_elf32() {
        let symtab = DynSymTab::new();
        let bytes = symtab.to_bytes(false);
        // Null entry: 16 bytes for ELF32
        assert_eq!(bytes.len(), 16);
        // First 4 bytes (name_offset) should be 0
        assert_eq!(&bytes[0..4], &[0, 0, 0, 0]);
    }

    #[test]
    fn test_dynsym_elf64_field_order() {
        // Verify ELF64 field ordering: name(4) info(1) other(1) shndx(2) value(8) size(8)
        let mut symtab = DynSymTab::new();
        symtab.add(DynSymEntry {
            name_offset: 0x01020304,
            value: 0x1122334455667788,
            size: 0xAABBCCDDEEFF0011,
            info: 0x12,
            other: 0x03,
            section_index: 0x0506,
        });
        let bytes = symtab.to_bytes(true);
        // Second entry starts at offset 24
        let entry = &bytes[24..48];
        // st_name: 04 03 02 01 (LE)
        assert_eq!(entry[0..4], [0x04, 0x03, 0x02, 0x01]);
        // st_info: 0x12
        assert_eq!(entry[4], 0x12);
        // st_other: 0x03
        assert_eq!(entry[5], 0x03);
        // st_shndx: 06 05 (LE)
        assert_eq!(entry[6..8], [0x06, 0x05]);
        // st_value: 88 77 66 55 44 33 22 11 (LE)
        assert_eq!(
            entry[8..16],
            [0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11]
        );
        // st_size: 11 00 FF EE DD CC BB AA (LE)
        assert_eq!(
            entry[16..24],
            [0x11, 0x00, 0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA]
        );
    }

    #[test]
    fn test_dynsym_elf32_field_order() {
        // Verify ELF32 field ordering: name(4) value(4) size(4) info(1) other(1) shndx(2)
        let mut symtab = DynSymTab::new();
        symtab.add(DynSymEntry {
            name_offset: 0x01020304,
            value: 0x11223344,
            size: 0xAABBCCDD,
            info: 0x12,
            other: 0x03,
            section_index: 0x0506,
        });
        let bytes = symtab.to_bytes(false);
        // Second entry starts at offset 16
        let entry = &bytes[16..32];
        // st_name: 04 03 02 01 (LE)
        assert_eq!(entry[0..4], [0x04, 0x03, 0x02, 0x01]);
        // st_value: 44 33 22 11 (LE)
        assert_eq!(entry[4..8], [0x44, 0x33, 0x22, 0x11]);
        // st_size: DD CC BB AA (LE)
        assert_eq!(entry[8..12], [0xDD, 0xCC, 0xBB, 0xAA]);
        // st_info: 0x12
        assert_eq!(entry[12], 0x12);
        // st_other: 0x03
        assert_eq!(entry[13], 0x03);
        // st_shndx: 06 05 (LE)
        assert_eq!(entry[14..16], [0x06, 0x05]);
    }

    // ---- DynamicSection tests ----

    #[test]
    fn test_dynamic_section_new() {
        let ds = DynamicSection::new();
        assert!(ds.entries.is_empty());
    }

    #[test]
    fn test_dynamic_section_add() {
        let mut ds = DynamicSection::new();
        ds.add(DT_NEEDED, 42);
        ds.add(DT_NULL, 0);
        assert_eq!(ds.entries.len(), 2);
    }

    #[test]
    fn test_dynamic_section_to_bytes_elf64() {
        let mut ds = DynamicSection::new();
        ds.add(DT_NEEDED, 1);
        ds.add(DT_NULL, 0);
        let bytes = ds.to_bytes(true);
        // 2 entries * 16 bytes each = 32 bytes
        assert_eq!(bytes.len(), 32);

        // First entry: tag=DT_NEEDED(1), value=1
        assert_eq!(&bytes[0..8], 1i64.to_le_bytes()); // tag
        assert_eq!(&bytes[8..16], 1u64.to_le_bytes()); // value

        // Second entry: DT_NULL(0), value=0
        assert_eq!(&bytes[16..24], 0i64.to_le_bytes());
        assert_eq!(&bytes[24..32], 0u64.to_le_bytes());
    }

    #[test]
    fn test_dynamic_section_to_bytes_elf32() {
        let mut ds = DynamicSection::new();
        ds.add(DT_NEEDED, 1);
        ds.add(DT_NULL, 0);
        let bytes = ds.to_bytes(false);
        // 2 entries * 8 bytes each = 16 bytes
        assert_eq!(bytes.len(), 16);

        // First entry: tag=DT_NEEDED(1), value=1
        assert_eq!(&bytes[0..4], 1i32.to_le_bytes());
        assert_eq!(&bytes[4..8], 1u32.to_le_bytes());

        // Second entry: DT_NULL(0), value=0
        assert_eq!(&bytes[8..12], 0i32.to_le_bytes());
        assert_eq!(&bytes[12..16], 0u32.to_le_bytes());
    }

    #[test]
    fn test_dynamic_section_dt_null_termination() {
        let dynstr = DynStrTab::new();
        let addrs = DynamicAddresses {
            dynstr_addr: 0x1000,
            dynstr_size: 1,
            dynsym_addr: 0x2000,
            hash_addr: 0x3000,
            plt_addr: 0x4000,
            pltgot_addr: 0x5000,
            plt_rel_addr: 0,
            plt_rel_size: 0,
        };
        let ds = DynamicSection::build_for_shared_library(&[], None, &dynstr, &addrs);
        // Last entry must be DT_NULL
        let last = ds.entries.last().unwrap();
        assert_eq!(last.tag, DT_NULL);
        assert_eq!(last.value, 0);
    }

    // ---- PLT generation tests ----

    #[test]
    fn test_plt_generation_x86_64_size() {
        let mut symtab = DynSymTab::new();
        // Add 3 symbols
        for _ in 0..3 {
            symtab.add(DynSymEntry {
                name_offset: 1,
                value: 0,
                size: 0,
                info: 0x12,
                other: 0,
                section_index: 1,
            });
        }
        let target = TargetConfig::x86_64();
        let plt = generate_plt(&symtab.entries, &target);
        // PLT[0] = 16 bytes + 3 entries * 16 bytes = 64 bytes
        assert_eq!(plt.len(), 16 + 3 * 16);
    }

    #[test]
    fn test_plt_generation_i686_size() {
        let mut symtab = DynSymTab::new();
        symtab.add(DynSymEntry {
            name_offset: 1,
            value: 0,
            size: 0,
            info: 0x12,
            other: 0,
            section_index: 1,
        });
        let target = TargetConfig::i686();
        let plt = generate_plt(&symtab.entries, &target);
        // PLT[0] = 16 bytes + 1 entry * 16 bytes = 32 bytes
        assert_eq!(plt.len(), 32);
    }

    #[test]
    fn test_plt_generation_aarch64_size() {
        let mut symtab = DynSymTab::new();
        symtab.add(DynSymEntry {
            name_offset: 1,
            value: 0,
            size: 0,
            info: 0x12,
            other: 0,
            section_index: 1,
        });
        let target = TargetConfig::aarch64();
        let plt = generate_plt(&symtab.entries, &target);
        // PLT[0] = 16 bytes + 1 entry * 16 bytes = 32 bytes
        assert_eq!(plt.len(), 32);
    }

    #[test]
    fn test_plt_generation_riscv64_size() {
        let mut symtab = DynSymTab::new();
        symtab.add(DynSymEntry {
            name_offset: 1,
            value: 0,
            size: 0,
            info: 0x12,
            other: 0,
            section_index: 1,
        });
        let target = TargetConfig::riscv64();
        let plt = generate_plt(&symtab.entries, &target);
        // PLT[0] = 16 bytes + 1 entry * 16 bytes = 32 bytes
        assert_eq!(plt.len(), 32);
    }

    #[test]
    fn test_plt_empty_symbols() {
        // Only the null symbol — no PLT entries beyond PLT[0]
        let symtab = DynSymTab::new();
        let target = TargetConfig::x86_64();
        let plt = generate_plt(&symtab.entries, &target);
        assert_eq!(plt.len(), 16); // Just PLT[0]
    }

    // ---- GOT generation tests ----

    #[test]
    fn test_got_64bit_size() {
        let target = TargetConfig::x86_64();
        let got = generate_got(5, &target);
        // (5 + 3) * 8 = 64 bytes
        assert_eq!(got.len(), 64);
    }

    #[test]
    fn test_got_32bit_size() {
        let target = TargetConfig::i686();
        let got = generate_got(5, &target);
        // (5 + 3) * 4 = 32 bytes
        assert_eq!(got.len(), 32);
    }

    #[test]
    fn test_got_zero_entries() {
        let target = TargetConfig::x86_64();
        let got = generate_got(0, &target);
        // 3 reserved entries * 8 bytes = 24
        assert_eq!(got.len(), 24);
    }

    #[test]
    fn test_got_all_zeros() {
        let target = TargetConfig::x86_64();
        let got = generate_got(2, &target);
        // All entries should be initialized to zero
        assert!(got.iter().all(|&b| b == 0));
    }

    // ---- Hash table tests ----

    #[test]
    fn test_hash_table_construction() {
        let mut dynstr = DynStrTab::new();
        let name1 = dynstr.add("printf");
        let name2 = dynstr.add("malloc");

        let symbols = vec![
            DynSymEntry {
                // Null entry
                name_offset: 0,
                value: 0,
                size: 0,
                info: 0,
                other: 0,
                section_index: 0,
            },
            DynSymEntry {
                name_offset: name1,
                value: 0x1000,
                size: 16,
                info: 0x12,
                other: 0,
                section_index: 1,
            },
            DynSymEntry {
                name_offset: name2,
                value: 0x2000,
                size: 16,
                info: 0x12,
                other: 0,
                section_index: 1,
            },
        ];

        let hash = build_hash_table(&symbols, &dynstr);
        // Header: nbucket(4) + nchain(4) = 8 bytes
        // Then: bucket[nbucket] + chain[nsyms]
        assert!(hash.len() >= 8);

        // Parse header
        let nbuckets = u32::from_le_bytes([hash[0], hash[1], hash[2], hash[3]]);
        let nchain = u32::from_le_bytes([hash[4], hash[5], hash[6], hash[7]]);
        assert!(nbuckets > 0);
        assert_eq!(nchain, 3); // 3 symbols (including null)
        assert_eq!(hash.len(), (2 + nbuckets + nchain) as usize * 4);
    }

    #[test]
    fn test_hash_table_single_symbol() {
        let mut dynstr = DynStrTab::new();
        let name = dynstr.add("_start");
        let symbols = vec![
            DynSymEntry {
                name_offset: 0,
                value: 0,
                size: 0,
                info: 0,
                other: 0,
                section_index: 0,
            },
            DynSymEntry {
                name_offset: name,
                value: 0x1000,
                size: 0,
                info: 0x12,
                other: 0,
                section_index: 1,
            },
        ];

        let hash = build_hash_table(&symbols, &dynstr);
        let nchain = u32::from_le_bytes([hash[4], hash[5], hash[6], hash[7]]);
        assert_eq!(nchain, 2); // null + _start
    }

    // ---- PLT relocation tests ----

    #[test]
    fn test_plt_relocations_elf64() {
        let mut symtab = DynSymTab::new();
        symtab.add(DynSymEntry {
            name_offset: 1,
            value: 0,
            size: 0,
            info: 0x12,
            other: 0,
            section_index: 1,
        });
        let target = TargetConfig::x86_64();
        let relocs = generate_plt_relocations(&symtab.entries, 0x3000, &target);
        // 1 RELA entry = 24 bytes for ELF64
        assert_eq!(relocs.len(), 24);

        // Check r_offset: GOT[3] = 0x3000 + 3*8 = 0x3018
        let r_offset = u64::from_le_bytes(relocs[0..8].try_into().unwrap());
        assert_eq!(r_offset, 0x3000 + 3 * 8);

        // Check r_info contains R_X86_64_JUMP_SLOT (7) and sym index 1
        let r_info = u64::from_le_bytes(relocs[8..16].try_into().unwrap());
        assert_eq!(r_info, elf::elf64_r_info(1, relocations::R_X86_64_JUMP_SLOT));
    }

    #[test]
    fn test_plt_relocations_elf32() {
        let mut symtab = DynSymTab::new();
        symtab.add(DynSymEntry {
            name_offset: 1,
            value: 0,
            size: 0,
            info: 0x12,
            other: 0,
            section_index: 1,
        });
        let target = TargetConfig::i686();
        let relocs = generate_plt_relocations(&symtab.entries, 0x3000, &target);
        // 1 REL entry = 8 bytes for ELF32
        assert_eq!(relocs.len(), 8);

        // Check r_offset: GOT[3] = 0x3000 + 3*4 = 0x300C
        let r_offset = u32::from_le_bytes(relocs[0..4].try_into().unwrap());
        assert_eq!(r_offset, 0x3000 + 3 * 4);

        // Check r_info contains R_386_JMP_SLOT (7) and sym index 1
        let r_info = u32::from_le_bytes(relocs[4..8].try_into().unwrap());
        assert_eq!(r_info, elf::elf32_r_info(1, relocations::R_386_JMP_SLOT));
    }

    // ---- generate_dynamic_sections integration test ----

    #[test]
    fn test_generate_dynamic_sections_basic() {
        let symbols = vec![ResolvedSymbol {
            name: "my_func".to_string(),
            address: 0x1000,
            size: 32,
            binding: SymbolBinding::Global,
            symbol_type: SymbolType::Function,
            visibility: SymbolVisibility::Default,
            output_section_index: 1,
        }];

        let target = TargetConfig::x86_64();
        let result = generate_dynamic_sections(
            &symbols,
            &["libc.so.6".to_string()],
            Some("libtest.so.1"),
            &target,
        );

        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(!output.dynamic.is_empty());
        assert!(!output.dynsym.is_empty());
        assert!(!output.dynstr.is_empty());
        assert!(!output.hash.is_empty());
        assert!(!output.plt.is_empty());
        assert!(!output.got.is_empty());
        assert!(!output.plt_relocations.is_empty());
    }

    #[test]
    fn test_generate_dynamic_sections_empty() {
        let target = TargetConfig::x86_64();
        let result = generate_dynamic_sections(&[], &[], None, &target);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(!output.dynamic.is_empty()); // At least DT_NULL
        assert!(!output.dynstr.is_empty()); // At least null byte
    }

    #[test]
    fn test_generate_dynamic_sections_i686() {
        let symbols = vec![ResolvedSymbol {
            name: "test_sym".to_string(),
            address: 0x8000,
            size: 16,
            binding: SymbolBinding::Global,
            symbol_type: SymbolType::Object,
            visibility: SymbolVisibility::Default,
            output_section_index: 2,
        }];

        let target = TargetConfig::i686();
        let result = generate_dynamic_sections(&symbols, &[], None, &target);
        assert!(result.is_ok());
        let output = result.unwrap();

        // Verify .dynsym uses ELF32 format (16 bytes per entry)
        // 2 entries (null + test_sym) = 32 bytes
        assert_eq!(output.dynsym.len(), 32);

        // Verify .dynamic uses ELF32 format (8 bytes per entry)
        assert_eq!(output.dynamic.len() % 8, 0);
    }

    // ---- DynamicAddresses tests ----

    #[test]
    fn test_dynamic_addresses_struct() {
        let addrs = DynamicAddresses {
            dynstr_addr: 0x1000,
            dynstr_size: 256,
            dynsym_addr: 0x2000,
            hash_addr: 0x3000,
            plt_addr: 0x4000,
            pltgot_addr: 0x5000,
            plt_rel_addr: 0x6000,
            plt_rel_size: 48,
        };
        assert_eq!(addrs.dynstr_addr, 0x1000);
        assert_eq!(addrs.dynstr_size, 256);
        assert_eq!(addrs.plt_rel_size, 48);
    }

    // ---- DT_* constant validation ----

    #[test]
    fn test_dt_constants() {
        // Verify DT_* constants match ELF specification values
        assert_eq!(DT_NULL, 0);
        assert_eq!(DT_NEEDED, 1);
        assert_eq!(DT_PLTRELSZ, 2);
        assert_eq!(DT_PLTGOT, 3);
        assert_eq!(DT_HASH, 4);
        assert_eq!(DT_STRTAB, 5);
        assert_eq!(DT_SYMTAB, 6);
        assert_eq!(DT_STRSZ, 10);
        assert_eq!(DT_SYMENT, 11);
        assert_eq!(DT_SONAME, 14);
        assert_eq!(DT_PLTREL, 20);
        assert_eq!(DT_JMPREL, 23);
    }

    // ---- build_for_shared_library tests ----

    #[test]
    fn test_build_for_shared_library_with_needed() {
        let mut dynstr = DynStrTab::new();
        dynstr.add("libc.so.6");
        dynstr.add("libm.so.6");
        dynstr.add("libtest.so.1");

        let addrs = DynamicAddresses {
            dynstr_addr: 0x1000,
            dynstr_size: dynstr.size() as u64,
            dynsym_addr: 0x2000,
            hash_addr: 0x3000,
            plt_addr: 0x4000,
            pltgot_addr: 0x5000,
            plt_rel_addr: 0x6000,
            plt_rel_size: 24,
        };

        let libs = vec!["libc.so.6".to_string(), "libm.so.6".to_string()];
        let ds = DynamicSection::build_for_shared_library(
            &libs,
            Some("libtest.so.1"),
            &dynstr,
            &addrs,
        );

        // Should contain DT_NEEDED entries
        let needed_entries: Vec<_> = ds
            .entries
            .iter()
            .filter(|e| e.tag == DT_NEEDED)
            .collect();
        assert_eq!(needed_entries.len(), 2);

        // Should contain DT_SONAME
        let soname_entries: Vec<_> = ds
            .entries
            .iter()
            .filter(|e| e.tag == DT_SONAME)
            .collect();
        assert_eq!(soname_entries.len(), 1);

        // Should contain DT_STRTAB
        let strtab_entries: Vec<_> = ds
            .entries
            .iter()
            .filter(|e| e.tag == DT_STRTAB)
            .collect();
        assert_eq!(strtab_entries.len(), 1);
        assert_eq!(strtab_entries[0].value, 0x1000);

        // Should end with DT_NULL
        let last = ds.entries.last().unwrap();
        assert_eq!(last.tag, DT_NULL);
    }
}
