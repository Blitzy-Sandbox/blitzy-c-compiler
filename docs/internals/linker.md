# Linker Internals

The bcc compiler includes a fully integrated ELF linker that reads relocatable object files and
static archives, resolves symbols across compilation units and system libraries, applies
architecture-specific relocations, merges sections into loadable segments, and emits complete
ELF binaries — all without invoking any external linker such as `ld`, `lld`, or `gold`. Every
step of the linking process executes within the single `bcc` process, ensuring zero external
toolchain dependencies at link time.

The integrated linker is implemented in the `src/linker/` directory and consists of the
following source modules:

| Source File | Responsibility |
|---|---|
| `src/linker/mod.rs` | Linker orchestrator and public entry point (`link()` function) |
| `src/linker/elf.rs` | ELF format structures for reading and writing ELF32/ELF64 binaries |
| `src/linker/archive.rs` | `ar` static archive reader for extracting library member objects |
| `src/linker/relocations.rs` | Architecture-specific relocation computation and application |
| `src/linker/sections.rs` | Section merging, ordering, and virtual address layout |
| `src/linker/symbols.rs` | Symbol collection, resolution, and duplicate detection |
| `src/linker/dynamic.rs` | Shared library support: `.dynamic`, `.dynsym`, PLT/GOT generation |
| `src/linker/script.rs` | Default linker script behavior: section-to-segment mapping and layout |

## Output Modes

The linker supports three distinct output modes selected via CLI flags:

| Mode | CLI Flags | ELF Type | Description |
|---|---|---|---|
| **Static Executable** | *(default)* | `ET_EXEC` | Fully linked executable with resolved symbols and CRT startup code |
| **Shared Library** | `-shared` + `-fPIC` | `ET_DYN` | Position-independent shared object with dynamic symbol tables and PLT/GOT |
| **Relocatable Object** | `-c` | `ET_REL` | Unlinked object file; no symbol resolution or relocation application performed |

When producing an executable, the linker automatically locates and links the system CRT startup
objects (`crt1.o`, `crti.o`, `crtn.o`) and resolves references against static archives such as
`libc.a`. When producing a shared library, the linker generates the full set of dynamic linking
sections (`.dynamic`, `.dynsym`, `.dynstr`, `.plt`, `.got`). When producing a relocatable object
(`-c` mode), the codegen output is written directly as an ELF relocatable object without invoking
the linking pipeline.

## Input Sources

The linker accepts the following categories of input:

1. **Compiled object code** — Machine code bytes, relocation entries, and symbol definitions
   produced by the bcc code generation backends (`src/codegen/`)
2. **System CRT objects** — Pre-compiled relocatable objects (`crt1.o`, `crti.o`, `crtn.o`)
   read from the target system's library directories
3. **Static archives** — `ar`-format archive files (e.g., `libc.a`) containing collections of
   relocatable objects, searched lazily to resolve undefined symbol references
4. **DWARF debug sections** — When `-g` is specified, debug information sections produced by
   `src/debug/` are included in the output with associated relocations

---

## ELF Format Structures

**Source:** `src/linker/elf.rs`

The ELF (Executable and Linkable Format) is the binary format used for all linker input and
output. The linker must handle both **ELF32** (for the i686 target) and **ELF64** (for x86-64,
AArch64, and RISC-V 64 targets). The `elf.rs` module provides Rust structures for reading and
writing both ELF widths through a unified abstraction.

### Target-to-ELF-Class Mapping

| Target | ELF Class | Header Size | Section Header Entry Size | Program Header Entry Size |
|---|---|---|---|---|
| `x86_64-linux-gnu` | ELF64 | 64 bytes | 64 bytes | 56 bytes |
| `i686-linux-gnu` | ELF32 | 52 bytes | 40 bytes | 32 bytes |
| `aarch64-linux-gnu` | ELF64 | 64 bytes | 64 bytes | 56 bytes |
| `riscv64-linux-gnu` | ELF64 | 64 bytes | 64 bytes | 56 bytes |

### ELF Header (`Elf32_Ehdr` / `Elf64_Ehdr`)

The ELF header is the first structure in every ELF file and identifies the file's class,
encoding, target architecture, and layout.

| Field | ELF32 Size (bytes) | ELF64 Size (bytes) | Description |
|---|---|---|---|
| `e_ident[16]` | 16 | 16 | Magic number, class, data encoding, version, OS/ABI, padding |
| `e_type` | 2 | 2 | Object file type: `ET_REL` (1), `ET_EXEC` (2), `ET_DYN` (3) |
| `e_machine` | 2 | 2 | Architecture: `EM_386` (3), `EM_X86_64` (62), `EM_AARCH64` (183), `EM_RISCV` (243) |
| `e_version` | 4 | 4 | ELF version: `EV_CURRENT` (1) |
| `e_entry` | 4 | 8 | Entry point virtual address (`_start` for executables, 0 for shared libs) |
| `e_phoff` | 4 | 8 | File offset to the program header table |
| `e_shoff` | 4 | 8 | File offset to the section header table |
| `e_flags` | 4 | 4 | Architecture-specific flags (e.g., RISC-V ISA extension flags) |
| `e_ehsize` | 2 | 2 | Size of ELF header itself: 52 for ELF32, 64 for ELF64 |
| `e_phentsize` | 2 | 2 | Size of one program header entry: 32 for ELF32, 56 for ELF64 |
| `e_phnum` | 2 | 2 | Number of program header entries |
| `e_shentsize` | 2 | 2 | Size of one section header entry: 40 for ELF32, 64 for ELF64 |
| `e_shnum` | 2 | 2 | Number of section header entries |
| `e_shstrndx` | 2 | 2 | Section header index of the section name string table (`.shstrtab`) |

**Total header size:** 52 bytes (ELF32) / 64 bytes (ELF64).

### `e_ident` Byte Layout

The 16-byte identification array at the start of every ELF file:

| Byte Index | Name | Value | Description |
|---|---|---|---|
| 0–3 | `EI_MAG0`–`EI_MAG3` | `\x7fELF` | Magic number identifying the file as ELF |
| 4 | `EI_CLASS` | 1 or 2 | `ELFCLASS32` (1) for i686; `ELFCLASS64` (2) for x86-64, AArch64, RISC-V 64 |
| 5 | `EI_DATA` | 1 | `ELFDATA2LSB` (1) — little-endian for all bcc targets |
| 6 | `EI_VERSION` | 1 | `EV_CURRENT` (1) |
| 7 | `EI_OSABI` | 0 | `ELFOSABI_NONE` (0) — generic System V / Linux ABI |
| 8 | `EI_ABIVERSION` | 0 | ABI version (unused, set to 0) |
| 9–15 | `EI_PAD` | 0 | Padding bytes (must be zero) |

### Section Header (`Elf32_Shdr` / `Elf64_Shdr`)

Each section in an ELF file is described by a section header entry. The linker reads section
headers from input relocatable objects and writes section headers for the output binary.

| Field | ELF32 Size (bytes) | ELF64 Size (bytes) | Description |
|---|---|---|---|
| `sh_name` | 4 | 4 | Offset into the `.shstrtab` section for this section's name |
| `sh_type` | 4 | 4 | Section type (see below) |
| `sh_flags` | 4 | 8 | Section attribute flags (see below) |
| `sh_addr` | 4 | 8 | Virtual address of the section in memory (0 for non-allocated sections) |
| `sh_offset` | 4 | 8 | File offset to the section's data |
| `sh_size` | 4 | 8 | Size of the section in bytes |
| `sh_link` | 4 | 4 | Link to an associated section (interpretation depends on `sh_type`) |
| `sh_info` | 4 | 4 | Additional section information (interpretation depends on `sh_type`) |
| `sh_addralign` | 4 | 8 | Required alignment (must be a power of 2; 0 or 1 means no constraint) |
| `sh_entsize` | 4 | 8 | Size of each entry for sections containing fixed-size entries (0 otherwise) |

**Total entry size:** 40 bytes (ELF32) / 64 bytes (ELF64).

**Common section types (`sh_type`):**

| Constant | Value | Description |
|---|---|---|
| `SHT_NULL` | 0 | Inactive section header (index 0 is always `SHT_NULL`) |
| `SHT_PROGBITS` | 1 | Program-defined data (`.text`, `.data`, `.rodata`, DWARF sections) |
| `SHT_SYMTAB` | 2 | Symbol table |
| `SHT_STRTAB` | 3 | String table (`.strtab`, `.dynstr`, `.shstrtab`) |
| `SHT_RELA` | 4 | Relocation entries with explicit addends |
| `SHT_HASH` | 5 | Symbol hash table for dynamic linking |
| `SHT_DYNAMIC` | 6 | Dynamic linking information (`.dynamic` section) |
| `SHT_NOBITS` | 8 | Section occupies no file space (`.bss`) |
| `SHT_REL` | 9 | Relocation entries without explicit addends (used by i686) |
| `SHT_DYNSYM` | 11 | Dynamic symbol table |

**Common section flags (`sh_flags`):**

| Constant | Value | Description |
|---|---|---|
| `SHF_WRITE` | 0x1 | Section is writable at runtime |
| `SHF_ALLOC` | 0x2 | Section occupies memory during execution |
| `SHF_EXECINSTR` | 0x4 | Section contains executable machine instructions |

### Program Header (`Elf32_Phdr` / `Elf64_Phdr`)

Program headers describe segments — contiguous regions of the ELF file that are mapped into
memory by the operating system loader. They are present in executables and shared libraries
but not in relocatable objects.

**ELF64 field layout:**

| Field | Size (bytes) | Description |
|---|---|---|
| `p_type` | 4 | Segment type (see below) |
| `p_flags` | 4 | Segment permission flags (PF_R, PF_W, PF_X) |
| `p_offset` | 8 | File offset of the segment |
| `p_vaddr` | 8 | Virtual address where the segment is loaded |
| `p_paddr` | 8 | Physical address (typically same as `p_vaddr` on Linux) |
| `p_filesz` | 8 | Size of the segment in the file |
| `p_memsz` | 8 | Size of the segment in memory (≥ `p_filesz`; difference is zero-filled) |
| `p_align` | 8 | Segment alignment (must be a power of 2) |

**ELF32 field layout** (note: `p_flags` appears after `p_memsz`, not after `p_type`):

| Field | Size (bytes) | Description |
|---|---|---|
| `p_type` | 4 | Segment type |
| `p_offset` | 4 | File offset of the segment |
| `p_vaddr` | 4 | Virtual address |
| `p_paddr` | 4 | Physical address |
| `p_filesz` | 4 | Size in file |
| `p_memsz` | 4 | Size in memory |
| `p_flags` | 4 | Permission flags |
| `p_align` | 4 | Alignment |

> **Important:** The field ordering differs between ELF32 and ELF64. In ELF64, `p_flags`
> immediately follows `p_type`. In ELF32, `p_flags` comes after `p_memsz`. The
> `src/linker/elf.rs` implementation must account for this difference when reading and writing
> program headers.

**Total entry size:** 32 bytes (ELF32) / 56 bytes (ELF64).

**Segment types (`p_type`):**

| Constant | Value | Description |
|---|---|---|
| `PT_NULL` | 0 | Unused entry |
| `PT_LOAD` | 1 | Loadable segment — mapped into memory by the OS loader |
| `PT_DYNAMIC` | 2 | Dynamic linking information (references `.dynamic` section) |
| `PT_INTERP` | 3 | Path to the dynamic linker (e.g., `/lib64/ld-linux-x86-64.so.2`) |
| `PT_PHDR` | 6 | The program header table itself |
| `PT_GNU_STACK` | 0x6474e551 | GNU extension: stack executability flags (NX stack when `PF_X` is absent) |

**Segment permission flags (`p_flags`):**

| Flag | Value | Meaning |
|---|---|---|
| `PF_X` | 0x1 | Execute permission |
| `PF_W` | 0x2 | Write permission |
| `PF_R` | 0x4 | Read permission |

---

## `ar` Static Archive Reading

**Source:** `src/linker/archive.rs`

Static archives (`.a` files) are collections of relocatable ELF objects bundled together using
the `ar` archive format. The linker reads archives such as `libc.a` to resolve undefined symbol
references. The `archive.rs` module parses the archive structure and extracts individual member
objects on demand.

### Archive Global Header

Every `ar` archive begins with an 8-byte magic string:

```
!<arch>\n
```

(Bytes: `21 3C 61 72 63 68 3E 0A`)

### Member Header Format

Each archive member is preceded by a 60-byte fixed-format ASCII header:

| Field | Size (bytes) | Description |
|---|---|---|
| `ar_name` | 16 | Member name, space-padded, terminated with `/` (or extended name reference) |
| `ar_date` | 12 | Last modification timestamp (decimal ASCII seconds since epoch) |
| `ar_uid` | 6 | Owner user ID (decimal ASCII) |
| `ar_gid` | 6 | Owner group ID (decimal ASCII) |
| `ar_mode` | 8 | File mode / permissions (octal ASCII) |
| `ar_size` | 10 | Member data size in bytes (decimal ASCII) |
| `ar_fmag` | 2 | Header magic: `` ` `` followed by `\n` (bytes `60 0A`) |

Member data immediately follows the header. If the member data size is odd, one byte of
padding (`\n`) is appended to maintain even-byte alignment for the next member header.

### Symbol Table (First Member: `/`)

The first member in most archives is the **symbol table** (also called the archive index),
identified by the name `/`. It maps symbol names to the archive members that define them,
enabling the linker to perform efficient lazy extraction.

**Format:**

1. **Count** — 4-byte big-endian unsigned integer: the number of symbols in the index
2. **Offsets** — `count` × 4-byte big-endian unsigned integers: file offsets pointing to the
   member header of each symbol's defining object
3. **Names** — Concatenated null-terminated symbol name strings, one per symbol in the same
   order as the offsets

The linker uses this index to determine which archive members to extract: when an undefined
symbol is encountered, the symbol table is consulted to find the member that defines it.

### Extended Name Section (Member: `//`)

When a member's filename exceeds 15 characters, it cannot fit in the 16-byte `ar_name` field.
Instead, long filenames are stored in a special member named `//` (the extended name section).
References use the format `/offset` in the `ar_name` field, where `offset` is a decimal byte
offset into the `//` member's data. The referenced name is terminated by `/\n` within the
extended name section.

### Lazy Loading Strategy

The linker does not extract every member from an archive. Instead, it employs a **demand-driven
extraction** strategy:

1. Read the archive symbol table (`/` member) to build a symbol-to-offset map
2. For each undefined symbol in the link, check if any archive provides a definition
3. If a match is found, extract that specific member and process it as a relocatable object
4. Newly extracted members may introduce additional undefined symbols, so repeat from step 2
5. Iterate until no new undefined symbols can be resolved by remaining archive members

This approach minimizes memory usage and link time by avoiding the extraction and processing
of archive members that are never referenced.

---

## Symbol Resolution

**Source:** `src/linker/symbols.rs`

Symbol resolution is the process of matching every undefined symbol reference in the input
objects to a corresponding definition, producing a complete mapping from symbol names to their
final virtual addresses.

### Symbol Resolution Algorithm

The linker resolves symbols through an iterative multi-pass process:

1. **Collect defined symbols** — Scan all directly-specified input relocatable objects and
   record every symbol with a definition (non-`SHN_UNDEF` section index)
2. **Collect undefined references** — Record every symbol reference that has no definition
   in the current symbol set
3. **Archive member extraction** — For each undefined symbol, search the archive symbol tables
   for a member that provides a definition. Extract matching members and add their symbols
   (both defined and undefined) to the working sets
4. **Iterate** — Newly extracted members may introduce new undefined references. Repeat
   steps 2–3 until either:
   - All undefined symbols are resolved, OR
   - No archive member can resolve any remaining undefined symbol
5. **Error reporting** — Any symbols that remain undefined after exhaustive archive searching
   are reported as linker errors

### Symbol Binding

Each symbol has a **binding** attribute that governs its visibility and conflict resolution:

| Binding | ELF Constant | Value | Behavior |
|---|---|---|---|
| Local | `STB_LOCAL` | 0 | Visible only within the defining object file; never participates in cross-object resolution |
| Global | `STB_GLOBAL` | 1 | Visible across all input objects; exactly one definition permitted (multiple = error) |
| Weak | `STB_WEAK` | 2 | Visible across all objects; can be overridden by a `STB_GLOBAL` definition without error |

**Conflict resolution rules:**

- Two `STB_GLOBAL` definitions of the same symbol → **multiple definition error**
- One `STB_GLOBAL` + one `STB_WEAK` → the `STB_GLOBAL` definition wins
- Two `STB_WEAK` definitions → the first one encountered wins (implementation-defined)
- `STB_LOCAL` symbols are never in conflict (they are scoped to their object)

### Symbol Visibility

For shared library output, symbol **visibility** controls external accessibility:

| Visibility | ELF Constant | Value | Behavior |
|---|---|---|---|
| Default | `STV_DEFAULT` | 0 | Symbol is visible to other shared objects and can be preempted |
| Hidden | `STV_HIDDEN` | 2 | Symbol is not visible outside the shared object |
| Protected | `STV_PROTECTED` | 3 | Symbol is visible outside but cannot be preempted by another definition |

Visibility is encoded in the `st_other` field of the ELF symbol table entry.

### Symbol Table Entry Structure

The linker reads and writes symbol table entries in the following format:

**`Elf64_Sym` (24 bytes):**

| Field | Size (bytes) | Description |
|---|---|---|
| `st_name` | 4 | Index into the associated string table for the symbol's name |
| `st_info` | 1 | Binding (upper 4 bits) and type (lower 4 bits): `(binding << 4) | type` |
| `st_other` | 1 | Visibility (lower 2 bits): `visibility & 0x3` |
| `st_shndx` | 2 | Section header index where the symbol is defined (`SHN_UNDEF` = 0 for undefined) |
| `st_value` | 8 | Symbol value (virtual address for defined symbols) |
| `st_size` | 8 | Size of the associated data object (0 if unknown or not applicable) |

**`Elf32_Sym` (16 bytes):**

| Field | Size (bytes) | Description |
|---|---|---|
| `st_name` | 4 | String table index |
| `st_value` | 4 | Symbol value |
| `st_size` | 4 | Symbol size |
| `st_info` | 1 | Binding and type |
| `st_other` | 1 | Visibility |
| `st_shndx` | 2 | Section index |

> **Note:** The field ordering differs between ELF32 and ELF64 symbol table entries. In ELF64,
> `st_info`/`st_other`/`st_shndx` appear before `st_value`/`st_size`. In ELF32, `st_value` and
> `st_size` appear before `st_info`.

**Common symbol types (`st_info` lower 4 bits):**

| Type | ELF Constant | Value | Description |
|---|---|---|---|
| No type | `STT_NOTYPE` | 0 | Unspecified type |
| Object | `STT_OBJECT` | 1 | Data object (variable, array) |
| Function | `STT_FUNC` | 2 | Function or executable code |
| Section | `STT_SECTION` | 3 | Associated with a section (used in relocations) |
| File | `STT_FILE` | 4 | Source file name |

### Duplicate Symbol Detection

When two input objects both define the same `STB_GLOBAL` symbol, the linker must report a
**multiple definition error**. The diagnostic includes:

- The symbol name
- The first definition's source file and section
- The conflicting definition's source file and section

Error format (GCC-compatible):

```
<file2>: multiple definition of `<symbol>'; <file1>: first defined here
```

---

## Relocation Processing

**Source:** `src/linker/relocations.rs`

After symbol resolution assigns final addresses to all symbols and section merging determines
the output layout, the linker applies **relocations** to patch the machine code and data
sections with the correct addresses. Each relocation entry describes a site in the object code
that needs to be fixed up with a symbol's resolved address.

### Relocation Application Process

For each relocation entry in each input object:

1. **Decode** the relocation entry to extract the target symbol index, relocation type, and
   addend (explicit for `Rela`, implicit for `Rel`)
2. **Resolve** the target symbol to its final virtual address using the completed symbol table
3. **Compute** the relocation value using the type-specific formula (see per-architecture
   tables below)
4. **Patch** the computed value into the output section data at the relocation site offset
5. **Overflow check** — verify the computed value fits within the relocation type's bit width;
   report an error if it exceeds the representable range

### Relocation Entry Structures

**`Elf64_Rela` (24 bytes) — used by x86-64, AArch64, RISC-V 64:**

| Field | Size (bytes) | Description |
|---|---|---|
| `r_offset` | 8 | Byte offset within the section where the relocation applies |
| `r_info` | 8 | Packed symbol index and relocation type: `(symbol_index << 32) | type` |
| `r_addend` | 8 | Explicit addend value added to the computed relocation |

**`Elf32_Rel` (8 bytes) — used by i686 for `SHT_REL` sections:**

| Field | Size (bytes) | Description |
|---|---|---|
| `r_offset` | 4 | Byte offset within the section |
| `r_info` | 4 | Packed: `(symbol_index << 8) | type` |

For `Elf32_Rel`, the addend is **implicit** — it is read from the existing contents at the
relocation site in the section data.

**`Elf32_Rela` (12 bytes) — alternative with explicit addend:**

| Field | Size (bytes) | Description |
|---|---|---|
| `r_offset` | 4 | Byte offset within the section |
| `r_info` | 4 | Packed: `(symbol_index << 8) | type` |
| `r_addend` | 4 | Explicit addend |

### Relocation Variable Definitions

All per-architecture relocation formulas use the following variables:

| Variable | Definition |
|---|---|
| **S** | Final virtual address of the symbol referenced by the relocation |
| **A** | Addend value (explicit from `r_addend` or implicit from section data) |
| **P** | Virtual address of the relocation site (where the patch is applied) |
| **G** | Offset of the symbol's GOT entry from the GOT base |
| **GOT** | Virtual address of the Global Offset Table base |
| **L** | Virtual address of the symbol's PLT entry |
| **Page(x)** | Address `x` rounded down to a 4096-byte page boundary: `x & ~0xFFF` |

### x86-64 Relocations

| Type | Value | Calculation | Width | Description |
|---|---|---|---|---|
| `R_X86_64_64` | 1 | S + A | 64-bit | Absolute 64-bit address |
| `R_X86_64_PC32` | 2 | S + A − P | 32-bit signed | PC-relative 32-bit displacement |
| `R_X86_64_GOT32` | 3 | G + A | 32-bit | Offset into the GOT |
| `R_X86_64_PLT32` | 4 | L + A − P | 32-bit signed | PC-relative PLT entry address |
| `R_X86_64_GOTPCREL` | 9 | G + GOT + A − P | 32-bit signed | PC-relative GOT entry address |
| `R_X86_64_32` | 10 | S + A | 32-bit unsigned | Absolute 32-bit (zero-extended); overflow if > 0xFFFFFFFF |
| `R_X86_64_32S` | 11 | S + A | 32-bit signed | Absolute 32-bit (sign-extended); overflow if outside ±2 GiB |

### i686 (x86-32) Relocations

| Type | Value | Calculation | Width | Description |
|---|---|---|---|---|
| `R_386_32` | 1 | S + A | 32-bit | Absolute 32-bit address |
| `R_386_PC32` | 2 | S + A − P | 32-bit signed | PC-relative 32-bit displacement |
| `R_386_GOT32` | 3 | G + A | 32-bit | GOT entry offset |
| `R_386_PLT32` | 4 | L + A − P | 32-bit signed | PC-relative PLT entry |
| `R_386_GOTOFF` | 9 | S + A − GOT | 32-bit | Offset from the GOT base |
| `R_386_GOTPC` | 10 | GOT + A − P | 32-bit | PC-relative GOT base address |

> **Note:** The i686 target primarily uses `SHT_REL` sections (implicit addend) rather than
> `SHT_RELA`. The linker reads the existing 32-bit value at the relocation site as the addend
> before applying the relocation.

### AArch64 Relocations

| Type | Value | Calculation | Width | Description |
|---|---|---|---|---|
| `R_AARCH64_ABS64` | 257 | S + A | 64-bit | Absolute 64-bit data address |
| `R_AARCH64_ABS32` | 258 | S + A | 32-bit | Absolute 32-bit data address |
| `R_AARCH64_CALL26` | 283 | (S + A − P) >> 2 | 26-bit signed | PC-relative branch-and-link (`BL` instruction) |
| `R_AARCH64_JUMP26` | 282 | (S + A − P) >> 2 | 26-bit signed | PC-relative branch (`B` instruction) |
| `R_AARCH64_ADR_PREL_PG_HI21` | 275 | Page(S + A) − Page(P) | 21-bit signed | ADRP page-relative high 21 bits |
| `R_AARCH64_ADD_ABS_LO12_NC` | 277 | (S + A) & 0xFFF | 12-bit | ADD immediate: low 12 bits, no overflow check |
| `R_AARCH64_LDST64_ABS_LO12_NC` | 286 | ((S + A) & 0xFFF) >> 3 | 9-bit | 64-bit load/store offset: low 12 bits scaled by 8 |

AArch64 relocations that modify instruction encodings must patch specific bit fields within
the 32-bit fixed-width instruction word without disturbing other fields.

### RISC-V 64 Relocations

| Type | Value | Calculation | Width | Description |
|---|---|---|---|---|
| `R_RISCV_64` | 2 | S + A | 64-bit | Absolute 64-bit data address |
| `R_RISCV_32` | 1 | S + A | 32-bit | Absolute 32-bit data address |
| `R_RISCV_BRANCH` | 16 | S + A − P | 12-bit signed | Conditional branch offset (B-type instruction encoding) |
| `R_RISCV_JAL` | 17 | S + A − P | 20-bit signed | Unconditional jump offset (J-type instruction encoding) |
| `R_RISCV_CALL` | 18 | S + A − P | 32-bit signed | AUIPC + JALR pair: upper 20 bits in AUIPC, lower 12 in JALR |
| `R_RISCV_PCREL_HI20` | 23 | (S + A − P + 0x800) >> 12 | 20-bit | PC-relative high 20 bits (AUIPC); `+0x800` compensates for sign extension of low 12 |
| `R_RISCV_PCREL_LO12_I` | 24 | S + A − P | 12-bit signed | PC-relative low 12 bits (I-type instruction) |
| `R_RISCV_HI20` | 26 | (S + A + 0x800) >> 12 | 20-bit | Absolute high 20 bits (LUI); `+0x800` compensates for sign extension |
| `R_RISCV_LO12_I` | 27 | (S + A) & 0xFFF | 12-bit signed | Absolute low 12 bits (I-type ADDI/load) |

RISC-V uses a split immediate encoding where upper and lower halves of an address are loaded
by separate instructions (e.g., `LUI` + `ADDI` or `AUIPC` + `JALR`). The `+0x800` bias in
`HI20` relocations compensates for the sign extension that occurs when the low 12 bits are
treated as a signed immediate by the `ADDI` or load instruction.

### Relocation Overflow Checking

Several relocation types have limited bit widths. When the computed value exceeds the
representable range, the linker must report an error:

| Architecture | Relocation | Max Positive | Max Negative | Common Cause |
|---|---|---|---|---|
| x86-64 | `R_X86_64_PC32` | +2 GiB | −2 GiB | Code/data > 2 GiB apart |
| x86-64 | `R_X86_64_32S` | +2 GiB | −2 GiB | Address outside 32-bit signed range |
| x86-64 | `R_X86_64_32` | +4 GiB | 0 | Address outside 32-bit unsigned range |
| AArch64 | `R_AARCH64_CALL26` | +128 MiB | −128 MiB | Branch target too far (±128 MiB) |
| AArch64 | `R_AARCH64_ADR_PREL_PG_HI21` | ±4 GiB | — | Page offset exceeds 21-bit range |
| RISC-V | `R_RISCV_BRANCH` | +4 KiB | −4 KiB | Conditional branch exceeds 12-bit range |
| RISC-V | `R_RISCV_JAL` | +1 MiB | −1 MiB | Jump exceeds 20-bit range |

Overflow errors are reported in GCC-compatible diagnostic format:

```
<object_file>:<section>: relocation R_<ARCH>_<TYPE> overflow: value <computed> does not fit in <N> bits
```

---

## Section Merging and Layout

**Source:** `src/linker/sections.rs`

The section merger combines same-named sections from all input objects into unified output
sections, computes their sizes and alignments, assigns virtual addresses, and maps sections
into loadable segments for the program header table.

### Section Merging Strategy

1. **Collection** — Gather all sections with the same name from every input object. For
   example, all `.text` sections are collected into a single merged `.text` output section.
2. **Alignment** — Each input section contribution is aligned to the maximum of:
   - The input section's `sh_addralign` value
   - The output section's alignment requirement
3. **Concatenation** — Input section data is concatenated sequentially within the merged
   output section, with padding bytes (zeros) inserted as needed to satisfy alignment.
4. **Symbol adjustment** — All symbols defined in a merged input section have their
   `st_value` updated to reflect their new offset within the merged output section:
   `new_value = old_value + input_section_offset_within_merged_section`
5. **Relocation adjustment** — All relocation entries targeting a merged input section have
   their `r_offset` updated: `new_offset = old_offset + input_section_offset_within_merged_section`

### Default Section Ordering

The linker places sections in the output file following a standard ordering that groups
sections by access permissions (enabling efficient segment mapping):

| Order | Section | Type | Flags | Segment |
|---|---|---|---|---|
| 1 | `.text` | `SHT_PROGBITS` | `SHF_ALLOC \| SHF_EXECINSTR` | PT_LOAD (R+X) |
| 2 | `.rodata` | `SHT_PROGBITS` | `SHF_ALLOC` | PT_LOAD (R+X) |
| 3 | `.data` | `SHT_PROGBITS` | `SHF_ALLOC \| SHF_WRITE` | PT_LOAD (R+W) |
| 4 | `.bss` | `SHT_NOBITS` | `SHF_ALLOC \| SHF_WRITE` | PT_LOAD (R+W) |
| 5 | `.symtab` | `SHT_SYMTAB` | *(none)* | *(not loaded)* |
| 6 | `.strtab` | `SHT_STRTAB` | *(none)* | *(not loaded)* |
| 7 | `.shstrtab` | `SHT_STRTAB` | *(none)* | *(not loaded)* |
| 8+ | `.debug_*` | `SHT_PROGBITS` | *(none)* | *(not loaded)* |

DWARF debug sections (`.debug_info`, `.debug_abbrev`, `.debug_line`, `.debug_str`,
`.debug_aranges`, `.debug_frame`, `.debug_loc`) are included when the `-g` flag is specified.
They are not loaded into memory at runtime (no `SHF_ALLOC` flag) but are present in the file
for debugger consumption.

### Section-to-Segment Mapping

For executables and shared libraries, allocated sections are mapped into PT_LOAD segments
based on their permission requirements:

**Segment 1 — Code (Read + Execute):**
- Contains: `.text`, `.rodata`
- Permissions: `PF_R | PF_X`
- The ELF header and program header table are also included in this segment

**Segment 2 — Data (Read + Write):**
- Contains: `.data`, `.bss`
- Permissions: `PF_R | PF_W`
- `.bss` occupies memory space but no file space: `p_filesz` accounts for `.data` only,
  while `p_memsz` includes both `.data` and `.bss` sizes

**Segment alignment requirements:**
- Each PT_LOAD segment must begin at a page-aligned file offset and virtual address
- The standard page size is **4096 bytes** (0x1000)
- The file offset and virtual address must be **congruent modulo the page size**:
  `p_vaddr % p_align == p_offset % p_align`
- This congruence requirement ensures that the OS can `mmap()` the segment efficiently
  without copying

### Virtual Address Layout

The linker assigns virtual addresses to segments starting from a target-specific base address:

| Target | ELF Class | Default Base Address | Rationale |
|---|---|---|---|
| x86-64 | ELF64 | `0x400000` | Standard Linux x86-64 executable base |
| i686 | ELF32 | `0x08048000` | Traditional Linux i386 executable base |
| AArch64 | ELF64 | `0x400000` | Standard Linux AArch64 executable base |
| RISC-V 64 | ELF64 | `0x400000` | Standard Linux RISC-V 64 executable base |

**Address assignment process:**

1. Place the ELF header at the base address
2. Place program headers immediately after the ELF header
3. Place the first PT_LOAD segment (code) starting at the next page-aligned address
4. Place the second PT_LOAD segment (data) starting at the next page-aligned address after
   the code segment ends
5. Non-loaded sections (`.symtab`, `.strtab`, `.shstrtab`, `.debug_*`) are appended after all
   loaded segments and do not require virtual address assignment
6. The section header table is placed at the end of the file

**`.bss` handling:**

The `.bss` section is a special `SHT_NOBITS` section that occupies zero bytes in the file
but occupies memory at runtime (initialized to zero by the OS loader). It is placed
immediately after `.data` in the R+W segment. The segment's `p_filesz` covers only `.data`,
while `p_memsz` covers both `.data` and `.bss`. The difference (`p_memsz - p_filesz`) is
zero-filled by the kernel during loading.

---

## CRT Object Linkage

When producing a static executable, the linker must incorporate the **C Runtime (CRT) startup
objects** that provide the entry point and initialization/finalization hooks expected by the
Linux kernel and the C library.

### CRT Object Roles

| Object | Purpose |
|---|---|
| `crt1.o` | Defines `_start` — the actual ELF entry point. Calls `__libc_start_main`, which initializes libc and calls `main()` |
| `crti.o` | Provides the prologue of the `.init` and `.fini` sections (function preamble instructions) |
| `crtn.o` | Provides the epilogue of the `.init` and `.fini` sections (function return instructions) |

The `.init` section runs global constructors before `main()`, and `.fini` runs global
destructors after `main()` returns. The linker concatenates `crti.o`'s `.init` prologue,
any user-provided `.init` content, and `crtn.o`'s `.init` epilogue into a complete
initialization function. The same pattern applies to `.fini`.

### Link Order

CRT objects must be linked in a specific order to ensure correct section concatenation:

```
crt1.o → crti.o → [user object files] → [static archive libraries] → crtn.o
```

1. `crt1.o` provides `_start` and references `__libc_start_main` and `main`
2. `crti.o` opens the `.init` and `.fini` section functions
3. User objects provide `main()` and application code
4. Libraries (e.g., `libc.a`) resolve remaining undefined symbols
5. `crtn.o` closes the `.init` and `.fini` section functions

### CRT Search Paths

The linker searches for CRT objects in architecture-specific system directories:

| Target | CRT Search Path | Notes |
|---|---|---|
| `x86_64-linux-gnu` | `/usr/lib/x86_64-linux-gnu/` | Native system libraries |
| `i686-linux-gnu` | `/usr/i686-linux-gnu/lib/` | Cross-compilation sysroot (may also check `/usr/lib/i386-linux-gnu/`) |
| `aarch64-linux-gnu` | `/usr/aarch64-linux-gnu/lib/` | Cross-compilation sysroot |
| `riscv64-linux-gnu` | `/usr/riscv64-linux-gnu/lib/` | Cross-compilation sysroot |

Additional search paths can be specified with the `-L` command-line flag, which are searched
before the default system paths.

### CRT Objects per Output Mode

| Output Mode | CRT Objects Used |
|---|---|
| Static executable (default) | `crt1.o` + `crti.o` + user objects + libraries + `crtn.o` |
| Shared library (`-shared`) | `crti.o` + `crtbeginS.o` + user objects + libraries + `crtendS.o` + `crtn.o` (no `crt1.o` — shared libs have no `_start`) |
| Relocatable object (`-c`) | None — CRT linkage is deferred to the final link step |

---

## Dynamic Linking Support

**Source:** `src/linker/dynamic.rs`

When the `-shared` flag is specified (combined with `-fPIC`), the linker produces a shared
library (`ET_DYN` ELF type) with full dynamic linking structures. This enables the output
to be loaded by the system dynamic linker (`ld-linux.so`) at runtime.

### Shared Library Characteristics

- **ELF type:** `ET_DYN` (3) — dynamic shared object
- **No `_start` entry point** — shared libraries do not contain `main`; `e_entry` is set to 0
  or to an initialization function address if present
- **Position-independent code** — all code accesses data through the GOT and calls external
  functions through the PLT, enabling the library to be loaded at any virtual address

### Generated Dynamic Sections

The linker generates the following sections for shared library output:

#### `.dynamic` Section

An array of `Elf64_Dyn` (or `Elf32_Dyn`) entries providing metadata for the dynamic linker:

| Tag | Value Content | Description |
|---|---|---|
| `DT_NEEDED` | String table offset | Names of required shared libraries (one entry per dependency) |
| `DT_SONAME` | String table offset | The shared object's own canonical name |
| `DT_SYMTAB` | Virtual address | Address of the `.dynsym` section |
| `DT_STRTAB` | Virtual address | Address of the `.dynstr` section |
| `DT_STRSZ` | Size in bytes | Size of the `.dynstr` section |
| `DT_HASH` / `DT_GNU_HASH` | Virtual address | Symbol hash table for efficient lookup |
| `DT_PLTGOT` | Virtual address | Address of the `.got.plt` section |
| `DT_PLTRELSZ` | Size in bytes | Total size of PLT relocation entries |
| `DT_PLTREL` | `DT_RELA` (7) or `DT_REL` (17) | Type of PLT relocations |
| `DT_JMPREL` | Virtual address | Address of the `.rela.plt` section |
| `DT_RELA` | Virtual address | Address of the `.rela.dyn` section |
| `DT_RELASZ` | Size in bytes | Total size of `.rela.dyn` entries |
| `DT_RELAENT` | Entry size | Size of one `Elf64_Rela` / `Elf32_Rela` entry |
| `DT_NULL` | 0 | Terminates the `.dynamic` array |

#### `.dynsym` — Dynamic Symbol Table

A subset of the full symbol table containing only symbols with `STV_DEFAULT` or `STV_PROTECTED`
visibility. These are the symbols exported by the shared library (or imported from other shared
libraries). The first entry (index 0) is always an undefined null symbol.

#### `.dynstr` — Dynamic String Table

String table containing the names referenced by `.dynsym` entries and `DT_NEEDED` / `DT_SONAME`
values in the `.dynamic` section.

#### `.plt` — Procedure Linkage Table

The PLT provides indirect call stubs for external function references. Each PLT entry performs
an indirect jump through a corresponding GOT entry. On first invocation, the GOT entry redirects
to the dynamic linker's resolver, which patches the GOT entry with the actual function address
for subsequent calls (lazy binding).

**PLT entry 0 (resolver stub):**
- Pushes identification data and jumps to the dynamic linker resolver
- Architecture-specific implementation

**PLT entries 1..N (per-symbol stubs):**

| Architecture | PLT Entry Sequence | Entry Size |
|---|---|---|
| x86-64 | `jmp *GOT[n](%rip); push <reloc_index>; jmp PLT[0]` | 16 bytes |
| i686 | `jmp *GOT[n]; push <reloc_index>; jmp PLT[0]` | 16 bytes |
| AArch64 | `adrp x16, GOT_PAGE; ldr x17, [x16, GOT_LO12]; br x17` | 16 bytes |
| RISC-V 64 | `auipc t3, GOT_HI20; ld t3, GOT_LO12(t3); jalr t1, t3` | 16 bytes |

#### `.got` / `.got.plt` — Global Offset Table

The GOT contains absolute addresses for data and function references that must be resolved at
load time. For lazy binding of PLT functions:

- `.got.plt[0]` — Address of the `.dynamic` section (used by the resolver)
- `.got.plt[1]` — Reserved for the dynamic linker's link map pointer
- `.got.plt[2]` — Reserved for the dynamic linker's resolver function address
- `.got.plt[3..N]` — Initially point back to the PLT stub's push/jmp sequence; overwritten
  with actual addresses on first call

#### `.rela.dyn` — Data Relocations

Contains `Elf64_Rela` (or `Elf32_Rela`) entries for data references that the dynamic linker
must resolve at load time (e.g., global variable addresses in the GOT).

#### `.rela.plt` — PLT Relocations

Contains `Elf64_Rela` entries describing the GOT entries that correspond to PLT stubs. The
dynamic linker processes these to resolve function addresses (either eagerly at load time or
lazily on first call).

---

## Default Linker Script Behavior

**Source:** `src/linker/script.rs`

The bcc linker does not read external linker script files. Instead, it implements a built-in
default layout strategy equivalent to a standard linker script. The `script.rs` module
encapsulates this default behavior.

### Default Layout Rules

| Rule | Description |
|---|---|
| **Entry point** | `_start` symbol from `crt1.o` for executables; no entry point for shared libraries |
| **Section ordering** | `.text` → `.rodata` → `.data` → `.bss` → non-allocated sections (see Section Merging) |
| **Segment alignment** | All PT_LOAD segments are page-aligned (4096-byte boundaries) |
| **Base address** | `0x400000` for ELF64, `0x08048000` for ELF32 |
| **BSS placement** | Immediately after `.data` in the same R+W segment (occupies no file space) |
| **PT_GNU_STACK** | Always emitted with `PF_R | PF_W` (no `PF_X`) for non-executable stack (NX) |
| **PT_INTERP** | Emitted for dynamically-linked executables; points to the system dynamic linker path |
| **PT_PHDR** | Emitted for executables and shared libraries; describes the program header table itself |

### Linking Mode Comparison

The behavior of the linker varies based on the selected linking mode:

| Aspect | Static (default / `-static`) | Dynamic | Shared (`-shared`) |
|---|---|---|---|
| ELF type | `ET_EXEC` | `ET_EXEC` | `ET_DYN` |
| Entry point | `_start` | `_start` | None (0 or init function) |
| CRT objects | `crt1.o`, `crti.o`, `crtn.o` | `crt1.o`, `crti.o`, `crtn.o` | `crti.o`, `crtn.o` |
| PT_INTERP | Not emitted | Emitted | Not emitted |
| `.dynamic` | Not present | Present | Present |
| PLT / GOT | Not present | Present (for shared lib calls) | Present |
| Symbol resolution | All at link time | Static + dynamic deferred | All deferred to load time |

### Dynamic Linker Paths

When PT_INTERP is emitted (dynamic linking), the path varies by architecture:

| Target | Dynamic Linker Path |
|---|---|
| `x86_64-linux-gnu` | `/lib64/ld-linux-x86-64.so.2` |
| `i686-linux-gnu` | `/lib/ld-linux.so.2` |
| `aarch64-linux-gnu` | `/lib/ld-linux-aarch64.so.1` |
| `riscv64-linux-gnu` | `/lib/ld-linux-riscv64-lp64d.so.1` |

---

## Linking Pipeline Flow

The end-to-end linking process is orchestrated by `src/linker/mod.rs` and follows a strict
sequential pipeline:

### Step 1: Input Collection

**Input:** CLI arguments specifying object files, library paths (`-L`), library names (`-l`),
and output mode flags (`-shared`, `-static`, `-c`).

**Process:**
- Gather all directly-specified object files from the command line
- For executable mode: locate and prepend CRT objects (`crt1.o`, `crti.o`) and append `crtn.o`
- For each `-l<name>` flag: search `-L` directories and system paths for `lib<name>.a`
  (or `lib<name>.so` for dynamic linking)

**Output:** Ordered list of input objects and archives.

**Errors:** File not found for specified objects or libraries; missing CRT objects.

### Step 2: Symbol Collection

**Input:** Ordered list of relocatable objects (including CRT objects).

**Process:**
- Parse ELF headers and section headers for each input object
- Read the `.symtab` and `.strtab` sections to extract all symbol definitions and references
- Build a global symbol table mapping symbol names to their defining object and section
- Record all undefined symbol references

**Output:** Global symbol table with defined and undefined symbol sets.

**Errors:** Invalid ELF format in input objects; corrupt symbol tables.

### Step 3: Archive Resolution

**Input:** Undefined symbol set and archive files.

**Process:**
- For each archive, read the archive symbol table (`/` member)
- For each undefined symbol, check if any archive provides a definition
- Extract matching members as relocatable objects and add their symbols to the global table
- Repeat until a fixed point is reached (no new undefined symbols resolved)

**Output:** Complete set of input objects (original + extracted archive members); updated
symbol table.

**Errors:** Undefined symbols remaining after exhaustive archive search (reported as link
errors at step completion).

### Step 4: Section Merging

**Input:** All input objects with their section data.

**Process:**
- Group sections by name across all input objects
- Concatenate section data within each group, inserting alignment padding as needed
- Compute the merged size and alignment for each output section
- Track the offset of each input section contribution within the merged output

**Output:** Merged section data with size and alignment information; input-to-output offset
mapping.

**Errors:** None typically at this stage (alignment and size computations are deterministic).

### Step 5: Address Assignment

**Input:** Merged sections with sizes and alignments.

**Process:**
- Select the base virtual address based on the target's ELF class
- Lay out segments according to the default linker script:
  - Code segment (R+X): ELF header + program headers + `.text` + `.rodata`
  - Data segment (R+W): `.data` + `.bss`
- Assign page-aligned virtual addresses to each segment
- Compute the virtual address of each section within its segment
- For shared libraries: also lay out `.dynamic`, `.dynsym`, `.plt`, `.got` sections

**Output:** Final virtual address assignments for all sections and segments.

**Errors:** Address space overflow (extremely unlikely for typical programs).

### Step 6: Symbol Finalization

**Input:** Global symbol table; section-to-virtual-address mapping.

**Process:**
- For each defined symbol, compute its final virtual address:
  `final_address = section_virtual_address + symbol_offset_within_section`
- For dynamic symbols (shared library output): populate `.dynsym` with export entries

**Output:** Complete symbol-to-address mapping for relocation application.

**Errors:** None (all symbols resolved in step 3; addresses assigned in step 5).

### Step 7: Relocation Application

**Input:** All relocation entries from input objects; final symbol addresses; section data.

**Process:**
- For each relocation entry:
  1. Look up the target symbol's final address
  2. Compute the relocation site's virtual address (section address + relocation offset)
  3. Apply the architecture-specific relocation formula
  4. Write the computed value to the relocation site in the merged section data
- Perform overflow checking for range-limited relocation types

**Output:** Fully relocated section data ready for output.

**Errors:** Relocation overflow; unsupported relocation type; unresolved symbol at relocation
site (should have been caught in step 3).

### Step 8: Output Emission

**Input:** ELF header fields, program headers, relocated section data, section headers.

**Process:**
- Write the ELF header with correct `e_type`, `e_machine`, `e_entry`, and table offsets
- Write program header table (for executables and shared libraries)
- Write section data in file-offset order, with padding for alignment
- Write section header table at the end of the file
- Set the output file's executable permission bit (for executables)

**Output:** Complete ELF binary file on disk.

**Errors:** I/O errors during file writing.

### Pipeline Summary Diagram

```
┌─────────────────┐
│  Input Collection │ ← CLI args, CRT paths, -L/-l flags
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Symbol Collection │ ← Parse .symtab from each object
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│ Archive Resolution│ ← Lazy extraction from .a files
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│ Section Merging  │ ← Concatenate same-named sections
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│ Address Assignment│ ← Assign virtual addresses, lay out segments
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│Symbol Finalization│ ← Compute final symbol addresses
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│Relocation Applied │ ← Patch code/data with resolved addresses
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│ Output Emission  │ ← Write complete ELF binary to disk
└─────────────────┘
```

---

## Error Handling

All linker errors are reported through the shared diagnostic infrastructure in
`src/common/diagnostics.rs`, using GCC-compatible formatting on stderr.

### Error Categories

| Error | Diagnostic Format | Severity |
|---|---|---|
| **Undefined symbol** | `<object>: undefined reference to '<symbol>'` | Error (fatal) |
| **Multiple definition** | `<file2>: multiple definition of '<symbol>'; <file1>: first defined here` | Error (fatal) |
| **Relocation overflow** | `<object>:<section>: relocation R_<ARCH>_<TYPE> overflow: value <val> does not fit in <N> bits` | Error (fatal) |
| **Invalid ELF format** | `<file>: file format not recognized` | Error (fatal) |
| **Missing CRT object** | `cannot find <crt_file> in <search_paths>` | Error (fatal) |
| **Unsupported relocation** | `<object>:<section>: unsupported relocation type <type_num>` | Error (fatal) |
| **Missing library** | `cannot find -l<name>: No such file or directory` | Error (fatal) |
| **Truncated archive** | `<archive>: truncated archive member header` | Error (fatal) |
| **Invalid archive** | `<file>: not a valid ar archive` | Error (fatal) |

### Error Behavior

- **Any linker error causes the process to exit with code 1.** No partial output is written.
- **Multiple errors may be reported** before exiting (e.g., several undefined symbols can all
  be reported in a single link attempt).
- **Warnings** (e.g., unused library, symbol visibility changes) are reported on stderr but
  do not cause link failure.
- All error messages include sufficient context (file names, section names, symbol names) to
  enable developers to diagnose and fix the issue.

---

## Cross-References

- **Architecture overview:** See [`docs/architecture.md`](../architecture.md) Section 3.7 for
  the linker's role in the overall bcc compilation pipeline
- **Target details:** See [`docs/targets.md`](../targets.md) for per-architecture ABI
  specifications, ELF format details, and CRT object locations
- **DWARF integration:** See [`docs/internals/dwarf.md`](dwarf.md) for how DWARF debug
  sections are generated and passed to the linker for inclusion in the output binary
- **Code generation output:** See `src/codegen/mod.rs` for the `ObjectCode` structure that
  the linker receives from the code generation backends
- **Diagnostics:** See `src/common/diagnostics.rs` for the `DiagnosticEmitter` used by
  all linker error reporting
