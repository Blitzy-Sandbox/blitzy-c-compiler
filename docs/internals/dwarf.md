# DWARF v4 Implementation

The `bcc` compiler generates DWARF version 4 debug information when the `-g` command-line flag is specified. DWARF v4 debug sections embed source-level metadata—file names, line numbers, type descriptions, variable locations, and call frame information—directly into the ELF output binary. This enables source-level debugging with industry-standard debuggers including GDB and LLDB across all four supported target architectures (x86-64, i686, AArch64, and RISC-V 64).

## Activation

Debug information generation is **off by default**. It is activated exclusively by passing the `-g` flag on the `bcc` command line:

```
bcc -g source.c -o program
```

When `-g` is absent, no `.debug_*` sections are emitted and the debug info generator is not invoked, keeping output binaries compact and compilation fast.

## Source Module Reference

The debug information subsystem lives in the `src/debug/` directory:

| Source File | Responsibility |
|---|---|
| `src/debug/mod.rs` | Debug info generator entry point; coordinates all DWARF section emission |
| `src/debug/dwarf.rs` | DWARF v4 core structures: compilation unit headers, abbreviation tables, string table, address range tables |
| `src/debug/line_program.rs` | `.debug_line` section: line number program state machine encoding source-to-address mappings |
| `src/debug/info.rs` | `.debug_info` section: subprogram DIEs, variable DIEs, type DIEs, scope DIEs; also `.debug_loc` location lists |
| `src/debug/frame.rs` | `.debug_frame` section: Call Frame Information (CFI) for stack unwinding |

## High-Level Data Flow

The debug info generator sits alongside the code generator in the compilation pipeline and feeds its output into the integrated linker:

```
                    ┌─────────────┐
  IR + SourceMap ──▶│  Debug Info  │──▶ DWARF sections (Vec<u8> + relocations)
  + CodeAddrMap     │  Generator   │        │
                    └─────────────┘        │
                                            ▼
                                     ┌─────────────┐
                                     │  Integrated  │──▶ Final ELF binary
                                     │    Linker    │    (with .debug_* sections)
                                     └─────────────┘
```

**Inputs:**

- **IR** — The intermediate representation of the compiled translation unit (provides type information, function boundaries, and variable declarations)
- **Source Map** (`src/common/source_map.rs`) — Maps byte offsets and macro expansions back to original source file/line/column positions
- **Machine Code Address Map** — Produced by the code generator, maps IR instructions and basic blocks to their final machine code byte offsets

**Outputs:**

- Seven DWARF sections, each produced as a `Vec<u8>` byte vector paired with a list of relocation entries
- These sections are passed to the integrated linker (`src/linker/`) which writes them into the final ELF output as non-loadable sections

---

## DWARF Sections Overview

The compiler generates seven ELF sections containing DWARF v4 data:

| ELF Section | DWARF Purpose | Source File |
|---|---|---|
| `.debug_info` | Compilation unit and DIE tree containing types, variables, and functions | `src/debug/info.rs` |
| `.debug_abbrev` | Abbreviation table encoding DIE tag/attribute schemas | `src/debug/dwarf.rs` |
| `.debug_line` | Line number program mapping machine code addresses to source locations | `src/debug/line_program.rs` |
| `.debug_str` | String table for DWARF string attributes (`DW_FORM_strp` references) | `src/debug/dwarf.rs` |
| `.debug_aranges` | Address ranges per compilation unit (enables fast CU lookup by debuggers) | `src/debug/dwarf.rs` |
| `.debug_frame` | Call Frame Information (CFI) for stack unwinding | `src/debug/frame.rs` |
| `.debug_loc` | Location lists for variables whose locations change across code ranges | `src/debug/info.rs` |

### Cross-Section References

DWARF sections are heavily interconnected through offset-based cross-references:

- `.debug_info` references `.debug_abbrev` via the compilation unit header's `debug_abbrev_offset` field
- `.debug_info` references `.debug_str` via `DW_FORM_strp` attribute values (4-byte offsets into the string table)
- `.debug_info` references `.debug_line` via the `DW_AT_stmt_list` attribute on `DW_TAG_compile_unit` DIEs
- `.debug_info` references `.debug_loc` via `DW_FORM_sec_offset` attribute values on variable DIEs with non-contiguous locations
- `.debug_aranges` references `.debug_info` via `debug_info_offset` fields pointing to the start of each compilation unit
- `.debug_frame` contains relocations referencing function start addresses in `.text`

All inter-section references that depend on final addresses (e.g., function start addresses, global variable addresses) are emitted as relocation entries. The integrated linker (`src/linker/`) resolves these relocations when writing the final ELF output, just as it does for code section relocations.

---

## Compilation Unit Header Format

Each compilation unit in `.debug_info` begins with a fixed-format header defined by the DWARF v4 standard. The `bcc` compiler uses the **32-bit DWARF format** (where `unit_length` fits in 4 bytes) even when targeting 64-bit architectures; this is standard practice and supported by all debuggers.

### Header Layout

```
Offset  Size    Field                   Value / Description
──────  ──────  ──────────────────────  ─────────────────────────────────────
0x00    4       unit_length             Total byte length of the CU contribution
                                        excluding this field itself
0x04    2       version                 4 (DWARF version 4)
0x06    4       debug_abbrev_offset     Byte offset into .debug_abbrev for this
                                        CU's abbreviation table
0x0A    1       address_size            Target pointer size in bytes:
                                          4 for i686
                                          8 for x86-64, AArch64, RISC-V 64
```

**Total header size: 11 bytes.**

The `unit_length` value encompasses everything after it: the version, abbreviation offset, address size, and the entire DIE tree for the compilation unit. The next CU (if any) begins at byte offset `unit_length + 4` from the start of this CU header.

Immediately following the header is the root DIE for the compilation unit (`DW_TAG_compile_unit`), serialized according to the abbreviation table referenced by `debug_abbrev_offset`.

---

## DIE Tree Construction

Debugging Information Entries (DIEs) form a tree structure rooted at the `DW_TAG_compile_unit` DIE. Each DIE has a tag identifying its kind, a set of attributes carrying data, and optionally child DIEs forming a subtree.

### DIE Hierarchy

```
DW_TAG_compile_unit (root — one per translation unit)
├── DW_TAG_base_type              (int, char, float, etc.)
├── DW_TAG_pointer_type           (pointer-to-T)
├── DW_TAG_typedef                (typedef aliases)
├── DW_TAG_structure_type         (struct definitions)
│   └── DW_TAG_member             (struct fields)
├── DW_TAG_array_type             (array types)
│   └── DW_TAG_subrange_type      (array bounds)
├── DW_TAG_variable               (global variables)
└── DW_TAG_subprogram             (function definitions)
    ├── DW_TAG_formal_parameter   (function parameters)
    ├── DW_TAG_variable           (local variables)
    └── DW_TAG_lexical_block      (nested scopes)
        └── DW_TAG_variable       (block-scoped variables)
```

Children lists are terminated by a **null DIE** (a single zero byte) in the serialized `.debug_info` stream. DIEs without children are marked with `DW_CHILDREN_no` in their abbreviation and have no null terminator.

### DIE Tag Attributes

#### `DW_TAG_compile_unit`

Represents the entire translation unit. Always the root DIE.

| Attribute | Form | Description |
|---|---|---|
| `DW_AT_producer` | `DW_FORM_strp` | Compiler identification string (e.g., `"bcc 1.0.0"`) |
| `DW_AT_language` | `DW_FORM_data2` | Source language: `DW_LANG_C11` (`0x001d`) |
| `DW_AT_name` | `DW_FORM_strp` | Source file name (e.g., `"main.c"`) |
| `DW_AT_comp_dir` | `DW_FORM_strp` | Compilation working directory (absolute path) |
| `DW_AT_low_pc` | `DW_FORM_addr` | Start address of the CU's code (relocatable) |
| `DW_AT_high_pc` | `DW_FORM_data4` or `DW_FORM_data8` | Size of the CU's code range in bytes |
| `DW_AT_stmt_list` | `DW_FORM_sec_offset` | Offset into `.debug_line` for this CU's line program |

Using `DW_FORM_data4`/`DW_FORM_data8` for `DW_AT_high_pc` encodes the code range as a size relative to `DW_AT_low_pc`, which is more compact and relocation-friendly than encoding a second absolute address.

#### `DW_TAG_subprogram`

Represents a function definition. One per function in the translation unit.

| Attribute | Form | Description |
|---|---|---|
| `DW_AT_name` | `DW_FORM_strp` | Function name |
| `DW_AT_decl_file` | `DW_FORM_data1` or `DW_FORM_data2` | Source file index (from line program file table) |
| `DW_AT_decl_line` | `DW_FORM_data2` or `DW_FORM_data4` | Source line number of the declaration |
| `DW_AT_type` | `DW_FORM_ref4` | Reference to the return type DIE (offset within `.debug_info`) |
| `DW_AT_low_pc` | `DW_FORM_addr` | Function start address (relocatable) |
| `DW_AT_high_pc` | `DW_FORM_data4` or `DW_FORM_data8` | Function code size in bytes |
| `DW_AT_frame_base` | `DW_FORM_exprloc` | Frame base location expression |
| `DW_AT_external` | `DW_FORM_flag_present` | Present if the function has external linkage |

The `DW_AT_frame_base` expression is architecture-dependent:

- **x86-64**: Typically `DW_OP_reg6` (rbp) when frame pointer is used, or a `DW_OP_breg7` (rsp) + offset expression otherwise
- **i686**: Typically `DW_OP_reg5` (ebp) or `DW_OP_breg4` (esp) + offset
- **AArch64**: Typically `DW_OP_reg29` (x29/fp) or `DW_OP_breg31` (sp) + offset
- **RISC-V 64**: Typically `DW_OP_reg8` (s0/fp) or `DW_OP_breg2` (sp) + offset

Void functions omit the `DW_AT_type` attribute entirely.

#### `DW_TAG_variable`

Represents a variable (global or local).

| Attribute | Form | Description |
|---|---|---|
| `DW_AT_name` | `DW_FORM_strp` | Variable name |
| `DW_AT_decl_file` | `DW_FORM_data1` or `DW_FORM_data2` | Source file index |
| `DW_AT_decl_line` | `DW_FORM_data2` or `DW_FORM_data4` | Source line number |
| `DW_AT_type` | `DW_FORM_ref4` | Reference to type DIE |
| `DW_AT_location` | `DW_FORM_exprloc` or `DW_FORM_sec_offset` | Location expression or location list offset |

For variables with a single fixed location throughout their scope, `DW_FORM_exprloc` encodes the location inline. For variables whose location changes (e.g., a variable that moves from a register to the stack), `DW_FORM_sec_offset` points to an entry in `.debug_loc`.

Global variables use `DW_OP_addr` with the variable's absolute address (resolved via relocation). Local stack variables typically use `DW_OP_fbreg` + offset from the frame base.

#### `DW_TAG_formal_parameter`

Represents a function parameter. Children of `DW_TAG_subprogram` DIEs, ordered to match the function signature.

| Attribute | Form | Description |
|---|---|---|
| `DW_AT_name` | `DW_FORM_strp` | Parameter name |
| `DW_AT_decl_file` | `DW_FORM_data1` or `DW_FORM_data2` | Source file index |
| `DW_AT_decl_line` | `DW_FORM_data2` or `DW_FORM_data4` | Source line number |
| `DW_AT_type` | `DW_FORM_ref4` | Reference to parameter type DIE |
| `DW_AT_location` | `DW_FORM_exprloc` or `DW_FORM_sec_offset` | Parameter location |

Parameters follow the same location encoding rules as variables. On entry, parameters typically reside in ABI-defined registers; after the function prologue, they may be spilled to the stack. A location list in `.debug_loc` captures this transition.

#### `DW_TAG_base_type`

Represents a fundamental C type.

| Attribute | Form | Description |
|---|---|---|
| `DW_AT_name` | `DW_FORM_strp` | Type name (e.g., `"int"`, `"char"`, `"double"`) |
| `DW_AT_byte_size` | `DW_FORM_data1` | Size in bytes |
| `DW_AT_encoding` | `DW_FORM_data1` | Data encoding identifier |

Common encoding values:

| Encoding Constant | Value | Used For |
|---|---|---|
| `DW_ATE_signed` | `0x05` | `int`, `short`, `long`, `long long`, `signed char` |
| `DW_ATE_unsigned` | `0x07` | `unsigned int`, `unsigned short`, `unsigned long`, `unsigned long long`, `unsigned char` |
| `DW_ATE_float` | `0x04` | `float`, `double`, `long double` |
| `DW_ATE_boolean` | `0x02` | `_Bool` |
| `DW_ATE_signed_char` | `0x06` | `char` (when char is signed) |
| `DW_ATE_unsigned_char` | `0x08` | `char` (when char is unsigned) |

Note that `DW_AT_byte_size` is target-dependent for types like `long` (4 bytes on i686, 8 bytes on 64-bit targets) and pointers.

#### `DW_TAG_pointer_type`

Represents a pointer-to-T type.

| Attribute | Form | Description |
|---|---|---|
| `DW_AT_byte_size` | `DW_FORM_data1` | Pointer size: 4 (i686) or 8 (64-bit targets) |
| `DW_AT_type` | `DW_FORM_ref4` | Reference to the pointee type DIE |

A `void *` pointer omits the `DW_AT_type` attribute (the pointee type is unspecified).

#### `DW_TAG_structure_type`

Represents a `struct` definition. Child DIEs describe each member field.

| Attribute | Form | Description |
|---|---|---|
| `DW_AT_name` | `DW_FORM_strp` | Struct tag name (omitted for anonymous structs) |
| `DW_AT_byte_size` | `DW_FORM_data4` | Total struct size including padding |

Each child `DW_TAG_member` DIE has:

| Attribute | Form | Description |
|---|---|---|
| `DW_AT_name` | `DW_FORM_strp` | Member field name |
| `DW_AT_type` | `DW_FORM_ref4` | Reference to field type DIE |
| `DW_AT_data_member_location` | `DW_FORM_data1` or `DW_FORM_data2` | Byte offset of the member within the struct |

Struct layout (field offsets and padding) is computed by `src/sema/types.rs` following the target ABI's alignment rules.

#### `DW_TAG_array_type`

Represents an array type. Contains a child `DW_TAG_subrange_type` DIE describing bounds.

| Attribute | Form | Description |
|---|---|---|
| `DW_AT_type` | `DW_FORM_ref4` | Reference to element type DIE |

The child `DW_TAG_subrange_type` has:

| Attribute | Form | Description |
|---|---|---|
| `DW_AT_type` | `DW_FORM_ref4` | Index type (typically `DW_TAG_base_type` for `size_t` or `int`) |
| `DW_AT_upper_bound` | `DW_FORM_data4` | Upper bound (element count − 1) |

For a declaration like `int arr[10]`, the `DW_AT_upper_bound` is `9`. Variable-length arrays omit `DW_AT_upper_bound`.

#### `DW_TAG_typedef`

Represents a `typedef` alias.

| Attribute | Form | Description |
|---|---|---|
| `DW_AT_name` | `DW_FORM_strp` | Typedef name |
| `DW_AT_type` | `DW_FORM_ref4` | Reference to the underlying type DIE |

---

## Abbreviation Table Encoding

The `.debug_abbrev` section defines a lookup table of abbreviation codes. Each abbreviation describes the tag, children flag, and attribute schema for a class of DIEs. DIEs in `.debug_info` reference abbreviations by their code number rather than repeating the full schema, providing significant compression.

### Abbreviation Entry Structure

Each abbreviation entry is encoded as:

```
ULEB128     abbreviation_code       (1, 2, 3, ... — sequential)
ULEB128     tag                     (DW_TAG_compile_unit, DW_TAG_subprogram, etc.)
1 byte      children_flag           (DW_CHILDREN_yes = 1, DW_CHILDREN_no = 0)
  Repeated pairs:
    ULEB128   attribute_name        (DW_AT_name, DW_AT_type, etc.)
    ULEB128   attribute_form        (DW_FORM_strp, DW_FORM_ref4, etc.)
  Terminator:
    0, 0                            (two zero ULEB128 values end the attribute list)
```

The entire abbreviation table is terminated by a single `0` byte (an abbreviation code of zero).

### LEB128 Encoding

DWARF uses LEB128 (Little-Endian Base 128) encoding extensively for compact representation of integers:

**ULEB128 (Unsigned LEB128):**
- Each byte contributes 7 data bits (bits 0–6)
- Bit 7 (high bit) is the continuation flag: `1` means more bytes follow, `0` means this is the last byte
- Bytes are ordered least-significant first

Example: encoding the value `624485`:
```
Byte 0: 0xE5  (0b_1_1100101)  — bits 0-6 = 0x65, continuation = 1
Byte 1: 0x8E  (0b_1_0001110)  — bits 7-13 = 0x0E, continuation = 1
Byte 2: 0x26  (0b_0_0100110)  — bits 14-20 = 0x26, continuation = 0 (last byte)
```

**SLEB128 (Signed LEB128):**
- Same byte format as ULEB128
- The sign bit is the most significant bit of the final byte's 7-bit payload
- If the sign bit is set, the value is sign-extended to fill the target integer width

### Shared Abbreviation Table

In practice, `bcc` generates a single abbreviation table shared across all compilation units. Each CU header's `debug_abbrev_offset` field points to offset `0` within `.debug_abbrev`. This simplifies the implementation without measurable cost, since typical compilation units use the same set of DIE tag/attribute patterns.

---

## Line Number Program (`.debug_line`)

The `.debug_line` section encodes a mapping from machine code addresses to source file, line number, and column positions. This is the section that allows a debugger to display "you are on line 42 of main.c" when stepping through code.

The encoding uses a compact state-machine program: a sequence of opcodes that, when executed, produce a matrix of (address, file, line, column) rows. This is far more space-efficient than storing an explicit table of every address-to-line mapping.

### Line Number Program Header

Each compilation unit's line program begins with a header (implemented in `src/debug/line_program.rs`):

| Field | Size | Value / Description |
|---|---|---|
| `unit_length` | 4 bytes | Total length of the line number program (excluding this field) |
| `version` | 2 bytes | `4` (DWARF v4) |
| `header_length` | 4 bytes | Byte length of the header after this field |
| `minimum_instruction_length` | 1 byte | Minimum instruction size in bytes: `1` for x86-64/i686, `4` for AArch64/RISC-V 64 |
| `maximum_operations_per_instruction` | 1 byte | `1` (bcc targets are not VLIW architectures) |
| `default_is_stmt` | 1 byte | `1` (lines are statement boundaries by default) |
| `line_base` | 1 byte (signed) | `-5` (minimum line increment for special opcodes) |
| `line_range` | 1 byte | `14` (range of line increments for special opcodes) |
| `opcode_base` | 1 byte | `13` (first special opcode value; standard opcodes are 1–12) |
| `standard_opcode_lengths` | 12 bytes | Array specifying how many ULEB128 operands each standard opcode takes |

Following the fixed fields are:

- **Directory table**: A sequence of null-terminated directory path strings, terminated by an empty string. These are the include directories referenced by file entries.
- **File name table**: A sequence of file entries, each containing: file name (null-terminated string), directory index (ULEB128), last modification time (ULEB128, 0 if unknown), file size (ULEB128, 0 if unknown). Terminated by an empty file name.

### State Machine Registers

The line number program operates a virtual state machine with the following registers:

| Register | Type | Initial Value | Description |
|---|---|---|---|
| `address` | Target address | `0` | Current machine code address |
| `file` | Unsigned integer | `1` | Current source file index (1-based, referencing the file table) |
| `line` | Unsigned integer | `1` | Current source line number |
| `column` | Unsigned integer | `0` | Current source column (0 means unknown) |
| `is_stmt` | Boolean | `default_is_stmt` | Whether the current position is a recommended breakpoint |
| `basic_block` | Boolean | `false` | Whether the current position begins a basic block |
| `end_sequence` | Boolean | `false` | Whether the current address is past the end of the sequence |
| `prologue_end` | Boolean | `false` | Whether the current address is the end of the function prologue |
| `epilogue_begin` | Boolean | `false` | Whether the current address is the start of the function epilogue |
| `discriminator` | Unsigned integer | `0` | Identifies the block within a source line (for distinguishing code from the same line in different control paths) |

### Standard Opcodes

Standard opcodes (values 1 through `opcode_base - 1`) perform explicit state machine operations:

| Opcode | Value | Operands | Operation |
|---|---|---|---|
| `DW_LNS_copy` | 1 | None | Append current state as a row to the line matrix; reset `basic_block`, `prologue_end`, `epilogue_begin`; set `discriminator` to 0 |
| `DW_LNS_advance_pc` | 2 | 1 (ULEB128) | Increment `address` by operand × `minimum_instruction_length` |
| `DW_LNS_advance_line` | 3 | 1 (SLEB128) | Add signed operand to `line` |
| `DW_LNS_set_file` | 4 | 1 (ULEB128) | Set `file` to operand value |
| `DW_LNS_set_column` | 5 | 1 (ULEB128) | Set `column` to operand value |
| `DW_LNS_negate_stmt` | 6 | None | Toggle `is_stmt` |
| `DW_LNS_set_basic_block` | 7 | None | Set `basic_block` to `true` |
| `DW_LNS_const_add_pc` | 8 | None | Advance `address` by the increment corresponding to special opcode 255 (a computed constant) |
| `DW_LNS_fixed_advance_pc` | 9 | 1 (uhalf, 2 bytes) | Set `address` to `address + operand` (does NOT multiply by `minimum_instruction_length`) |
| `DW_LNS_set_prologue_end` | 10 | None | Set `prologue_end` to `true` |
| `DW_LNS_set_epilogue_begin` | 11 | None | Set `epilogue_begin` to `true` |
| `DW_LNS_set_isa` | 12 | 1 (ULEB128) | Set instruction set architecture register (unused by bcc) |

### Extended Opcodes

Extended opcodes are encoded as: a `0` byte (indicating extended), followed by the total length of the extended opcode (ULEB128), followed by the sub-opcode byte, followed by operands:

| Opcode | Sub-opcode | Operands | Operation |
|---|---|---|---|
| `DW_LNE_end_sequence` | 1 | None | Set `end_sequence` to `true`; append row to matrix; reset all registers to initial values |
| `DW_LNE_set_address` | 2 | Target-width address (4 or 8 bytes) | Set `address` to the operand value; this is the primary mechanism for establishing the initial address of a line program sequence |
| `DW_LNE_define_file` | 4 | File entry (name + dir_idx + time + size) | Add a new file entry to the file table at runtime |

Every line program for a compilation unit must end with `DW_LNE_end_sequence`. Failure to emit this terminator causes debuggers to report corrupt debug information.

### Special Opcodes

Special opcodes (values ≥ `opcode_base`, i.e., ≥ 13) provide a compact encoding for the most common operation: simultaneously advancing both the `address` and `line` registers and appending a row. A single byte encodes both increments:

```
adjusted_opcode = opcode - opcode_base
line_increment  = line_base + (adjusted_opcode % line_range)
address_advance = adjusted_opcode / line_range
```

The opcode that produces a given `(line_increment, address_advance)` pair is computed as:

```
opcode = (line_increment - line_base) + (line_range × address_advance) + opcode_base
```

With `line_base = -5` and `line_range = 14`, special opcodes can encode line increments from `-5` to `+8` and address advances from `0` to `(255 - 13) / 14 = 17` operations (where each operation is `minimum_instruction_length` bytes).

This is the **primary encoding mechanism** used by the line program generator in `src/debug/line_program.rs`. The vast majority of line-to-address transitions in typical C code fit within special opcode ranges, yielding extremely compact encoding (one byte per source-line transition in most cases).

---

## Call Frame Information (`.debug_frame`)

The `.debug_frame` section enables debuggers and exception handlers to unwind the call stack—walking from the current function back through the chain of callers—by describing how each function's stack frame is laid out and where callee-saved registers are preserved.

Implementation resides in `src/debug/frame.rs`.

### CIE (Common Information Entry)

A CIE defines the shared conventions for a set of functions. In practice, `bcc` emits one CIE per target architecture.

| Field | Size | Value / Description |
|---|---|---|
| `length` | 4 bytes | Total byte length of this CIE (excluding the length field itself) |
| `CIE_id` | 4 bytes | `0xFFFFFFFF` (distinguishes a CIE from an FDE) |
| `version` | 1 byte | `4` (DWARF v4 CIE version) |
| `augmentation` | Null-terminated string | `""` (empty string — bcc does not use augmentations) |
| `address_size` | 1 byte | Target pointer size: `4` (i686) or `8` (64-bit targets) |
| `segment_selector_size` | 1 byte | `0` (Linux does not use segmented addressing) |
| `code_alignment_factor` | ULEB128 | Instruction alignment in bytes: `1` for x86-64/i686, `4` for AArch64/RISC-V 64 |
| `data_alignment_factor` | SLEB128 | Stack slot alignment (negative by convention): `-8` for x86-64, `-4` for i686/AArch64/RISC-V 64 |
| `return_address_register` | ULEB128 | DWARF register number of the return address: `16` (x86-64), `8` (i686), `30` (AArch64 x30/lr), `1` (RISC-V ra/x1) |
| Initial instructions | Variable | CFI instructions defining the default CFA rule at function entry |

The `code_alignment_factor` allows CFI advance-location deltas to be divided by this factor, saving space. The `data_alignment_factor` similarly factors register save offsets.

### FDE (Frame Description Entry)

An FDE describes one function's call frame changes throughout its code range. Each FDE references a CIE.

| Field | Size | Value / Description |
|---|---|---|
| `length` | 4 bytes | Total byte length of this FDE (excluding the length field) |
| `CIE_pointer` | 4 bytes | Byte offset from this field to the associated CIE |
| `initial_location` | Address-size bytes | Function start address (subject to relocation) |
| `address_range` | Address-size bytes | Function code size in bytes |
| Call frame instructions | Variable | CFI instructions describing register save locations and CFA changes |

### CFI Instructions

Call Frame Instructions describe how the Canonical Frame Address (CFA) and register save locations change as execution progresses through a function:

| Instruction | Encoding | Operands | Description |
|---|---|---|---|
| `DW_CFA_def_cfa` | `0x0C` | register (ULEB128), offset (ULEB128) | Set CFA to `register + offset` |
| `DW_CFA_def_cfa_register` | `0x0D` | register (ULEB128) | Change CFA register (offset unchanged) |
| `DW_CFA_def_cfa_offset` | `0x0E` | offset (ULEB128) | Change CFA offset (register unchanged) |
| `DW_CFA_offset` | `0x80 \| reg` (high 2 bits = `10`) | factored offset (ULEB128) | Register `reg` is saved at `CFA + (offset × data_alignment_factor)` |
| `DW_CFA_advance_loc` | `0x40 \| delta` (high 2 bits = `01`) | (inline) | Advance location by `delta × code_alignment_factor` bytes |
| `DW_CFA_advance_loc1` | `0x02` | 1-byte delta | Advance location by `delta × code_alignment_factor` |
| `DW_CFA_advance_loc2` | `0x03` | 2-byte delta | Advance location by `delta × code_alignment_factor` |
| `DW_CFA_advance_loc4` | `0x04` | 4-byte delta | Advance location by `delta × code_alignment_factor` |
| `DW_CFA_restore` | `0xC0 \| reg` (high 2 bits = `11`) | (inline) | Restore register `reg` to its initial (CIE-defined) rule |
| `DW_CFA_nop` | `0x00` | None | No operation (padding to align entries) |

### Per-Architecture CFI Conventions

#### x86-64

- **CFA at function entry**: `rsp + 8` (the return address has been pushed by `call`)
- **After `push rbp`**: CFA becomes `rsp + 16`; `rbp` saved at `CFA - 16`
- **After `mov rbp, rsp`**: CFA redefines to `rbp + 16`
- **Return address**: Saved at `CFA - 8` (always, by the `call` instruction)
- **Callee-saved registers**: `rbx`, `r12`, `r13`, `r14`, `r15`, `rbp` — must be tracked if saved

#### i686

- **CFA at function entry**: `esp + 4` (return address pushed by `call`)
- **After `push ebp`**: CFA becomes `esp + 8`; `ebp` saved at `CFA - 8`
- **After `mov ebp, esp`**: CFA redefines to `ebp + 8`
- **Return address**: Saved at `CFA - 4`
- **Callee-saved registers**: `ebx`, `esi`, `edi`, `ebp`

#### AArch64

- **CFA at function entry**: `sp + 0`
- **After `stp x29, x30, [sp, #-N]!`**: CFA becomes `sp + N`; `x29` (fp) and `x30` (lr) saved
- **Return address**: In `x30` (lr) register, DWARF register 30
- **Callee-saved registers**: `x19`–`x28`, `x29` (fp), `x30` (lr)

#### RISC-V 64

- **CFA at function entry**: `sp + 0`
- **After `addi sp, sp, -N`**: CFA becomes `sp + N`
- **After `sd ra, offset(sp)`**: Return address (`ra`, DWARF register 1) saved at `CFA + offset - N`
- **Return address**: In `ra` (x1) register, DWARF register 1
- **Callee-saved registers**: `s0`–`s11` (x8–x9, x18–x27), `ra` (x1)

---

## String Table (`.debug_str`)

The `.debug_str` section provides deduplicated storage for all strings referenced by `DW_FORM_strp` attributes in `.debug_info`. Rather than embedding string data inline in each DIE (which would duplicate common strings like type names, file names, and the producer string), DIEs store a 4-byte offset into `.debug_str`.

### Structure

The section is a simple concatenation of null-terminated strings:

```
Offset 0x0000: "bcc 1.0.0\0"
Offset 0x000B: "main.c\0"
Offset 0x0012: "int\0"
Offset 0x0016: "char\0"
Offset 0x001B: "main\0"
...
```

A `DW_FORM_strp` attribute in `.debug_info` contains the byte offset (e.g., `0x0012` to reference the string `"int"`).

### Deduplication

During generation, `src/debug/dwarf.rs` maintains a `HashMap<String, u32>` mapping each unique string to its offset in the section. When a new string is encountered:

1. Check if it already exists in the map
2. If yes, reuse the existing offset
3. If no, append the string (with null terminator) to the section buffer and record its offset

This deduplication provides significant size reduction in practice, since type names like `"int"`, `"char"`, and `"unsigned int"` appear in many DIEs across a compilation unit.

---

## Address Ranges (`.debug_aranges`)

The `.debug_aranges` section provides a fast lookup table that maps machine code addresses to compilation units. Debuggers use this section to quickly determine which CU owns a given program counter value without needing to parse the entire `.debug_info` section.

### Structure Per Compilation Unit

Each CU's entry in `.debug_aranges` has the following layout:

| Field | Size | Description |
|---|---|---|
| `unit_length` | 4 bytes | Length of this CU's arange contribution (excluding this field) |
| `version` | 2 bytes | `2` (the aranges version is 2, even in DWARF v4) |
| `debug_info_offset` | 4 bytes | Byte offset of this CU's contribution in `.debug_info` |
| `address_size` | 1 byte | Target pointer size: `4` (i686) or `8` (64-bit targets) |
| `segment_size` | 1 byte | `0` (Linux does not use segmented addressing) |
| Padding | Variable | Pad to align the first tuple to a `2 × address_size` boundary |
| Address/length tuples | `2 × address_size` each | `(start_address, length)` pairs for each contiguous code range |
| Terminator | `2 × address_size` | `(0, 0)` pair marking the end of the list |

For a typical single-file compilation, there is one address/length tuple covering the entire `.text` section contribution of that CU.

The `debug_info_offset` field enables the debugger to jump directly to the right CU in `.debug_info` once it has identified which CU owns the target address.

---

## Location Lists (`.debug_loc`)

The `.debug_loc` section describes variable locations that change across a function's code range. When a variable cannot be described by a single location expression—for example, because it starts in a register and is later spilled to the stack—a **location list** encodes the variable's location at each point in the function's address range.

### When Location Lists Are Used

A variable DIE (`DW_TAG_variable` or `DW_TAG_formal_parameter`) uses a location list when:

- The variable resides in different registers at different points in the function
- The variable is in a register during part of the function and on the stack during another part
- The variable's stack offset changes (e.g., due to dynamic stack adjustments)
- The optimizer places the variable in different locations across optimized code paths

In the DIE, the `DW_AT_location` attribute uses `DW_FORM_sec_offset` to store a byte offset into `.debug_loc` (rather than `DW_FORM_exprloc` for an inline single-location expression).

### Location List Structure

A location list is a sequence of entries:

```
Repeated entries:
  start_address    (address_size bytes)   — Start of this location's validity range
  end_address      (address_size bytes)   — End of this location's validity range (exclusive)
  expression_length (2 bytes, uhalf)      — Byte length of the location expression
  location_expression (variable)          — DW_OP_* opcodes describing the location
Terminator:
  0                (address_size bytes)   — Zero start address
  0                (address_size bytes)   — Zero end address
```

Addresses in location list entries are relative to the compilation unit's base address (`DW_AT_low_pc` of the `DW_TAG_compile_unit` DIE). A base-address selection entry (where `start_address` is the maximum address value for the target) can override this base.

### Location Expression Opcodes

Location expressions use a stack-based virtual machine with `DW_OP_*` opcodes:

| Opcode | Value | Operands | Description |
|---|---|---|---|
| `DW_OP_reg0`–`DW_OP_reg31` | `0x50`–`0x6F` | None | Variable is in the specified DWARF register |
| `DW_OP_breg0`–`DW_OP_breg31` | `0x70`–`0x8F` | SLEB128 offset | Push `register + offset` onto the expression stack |
| `DW_OP_fbreg` | `0x91` | SLEB128 offset | Push `frame_base + offset` (frame-base-relative addressing for stack variables) |
| `DW_OP_addr` | `0x03` | Address-size value | Push an absolute address (for global variables; subject to relocation) |
| `DW_OP_stack_value` | `0x9F` | None | The computed value on the stack IS the variable's value (not a memory address) |
| `DW_OP_piece` | `0x93` | ULEB128 size | The preceding location describes `size` bytes of the variable |
| `DW_OP_plus_uconst` | `0x23` | ULEB128 value | Add an unsigned constant to the top of stack |
| `DW_OP_deref` | `0x06` | None | Dereference the address on top of stack |

Common patterns:

- **Stack variable**: `DW_OP_fbreg(-16)` — variable is at frame base minus 16 bytes
- **Register variable**: `DW_OP_reg0` — variable is in DWARF register 0 (rax on x86-64)
- **Global variable**: `DW_OP_addr(0x601040)` — variable is at absolute address 0x601040
- **Optimized-out variable**: Empty location expression (length 0) — variable value is not available

---

## Integration with the Integrated Linker

The debug info generator (`src/debug/mod.rs`) produces DWARF sections as raw byte buffers paired with relocation entries. These are consumed by the integrated linker (`src/linker/`) exactly like code and data sections.

### Output Format

Each DWARF section is passed to the linker as:

```rust
struct DebugSection {
    name: &'static str,          // e.g., ".debug_info"
    data: Vec<u8>,               // Raw section bytes
    relocations: Vec<Relocation>, // Relocation entries within this section
}
```

Relocations within DWARF sections reference symbols from the code sections—primarily function start addresses, global variable addresses, and section base addresses.

### Relocation Types in DWARF Sections

DWARF sections use a small subset of the target's relocation types:

| Usage in DWARF | x86-64 | i686 | AArch64 | RISC-V 64 |
|---|---|---|---|---|
| Absolute address (`DW_FORM_addr`) | `R_X86_64_64` | `R_386_32` | `R_AARCH64_ABS64` | `R_RISCV_64` |
| 32-bit section offset (`DW_FORM_ref_addr`) | `R_X86_64_32` | `R_386_32` | `R_AARCH64_ABS32` | `R_RISCV_32` |

Relocation entries in DWARF sections are processed during the linker's relocation application phase (see `src/linker/relocations.rs`). The linker patches the byte buffers with resolved addresses before writing them into the output ELF file.

### Section Flags in ELF Output

DWARF sections are emitted as **non-loadable** ELF sections:

- Section type: `SHT_PROGBITS` (except `.debug_str` which may use `SHT_PROGBITS` with the `SHF_MERGE | SHF_STRINGS` flags for deduplication)
- Section flags: `0` (no `SHF_ALLOC`, no `SHF_WRITE`, no `SHF_EXECINSTR`)
- Not mapped into any `PT_LOAD` program segment

This means debug sections consume disk space in the ELF file but not runtime memory when the program is loaded. The `strip` command can remove them entirely for production deployments.

### Behavior in `-c` (Relocatable Object) Mode

When compiling with `-c` (producing a relocatable `.o` file rather than a final executable), each object carries its own DWARF sections with unresolved relocations. The linker does **not** merge DWARF sections from multiple objects in this mode. Merging happens only during final linking to an executable or shared library.

---

## Architecture-Specific DWARF Register Mappings

DWARF register numbers are architecture-specific and differ from hardware register encoding. These mappings are used in location expressions (`DW_OP_reg*`, `DW_OP_breg*`), CFI instructions (`DW_CFA_offset`, `DW_CFA_def_cfa`), and the CIE's `return_address_register` field.

### x86-64 Register Mapping

| DWARF Number | Register | ABI Role |
|---|---|---|
| 0 | `rax` | Return value (integer) |
| 1 | `rdx` | 3rd argument / return value (high) |
| 2 | `rcx` | 4th argument |
| 3 | `rbx` | Callee-saved |
| 4 | `rsi` | 2nd argument |
| 5 | `rdi` | 1st argument |
| 6 | `rbp` | Frame pointer (callee-saved) |
| 7 | `rsp` | Stack pointer |
| 8 | `r8` | 5th argument |
| 9 | `r9` | 6th argument |
| 10 | `r10` | Caller-saved |
| 11 | `r11` | Caller-saved |
| 12 | `r12` | Callee-saved |
| 13 | `r13` | Callee-saved |
| 14 | `r14` | Callee-saved |
| 15 | `r15` | Callee-saved |
| 16 | Return address | Virtual register (return address column) |
| 17–32 | `xmm0`–`xmm15` | SSE registers |

### i686 Register Mapping

| DWARF Number | Register | ABI Role |
|---|---|---|
| 0 | `eax` | Return value |
| 1 | `ecx` | Caller-saved |
| 2 | `edx` | Caller-saved / return value (high) |
| 3 | `ebx` | Callee-saved |
| 4 | `esp` | Stack pointer |
| 5 | `ebp` | Frame pointer (callee-saved) |
| 6 | `esi` | Callee-saved |
| 7 | `edi` | Callee-saved |
| 8 | Return address | Virtual register (return address column) |

### AArch64 Register Mapping

| DWARF Number | Register | ABI Role |
|---|---|---|
| 0–7 | `x0`–`x7` | Argument / return value registers |
| 8–15 | `x8`–`x15` | Caller-saved (x8 = indirect result) |
| 16–18 | `x16`–`x18` | IP0, IP1, platform register |
| 19–28 | `x19`–`x28` | Callee-saved |
| 29 | `x29` (`fp`) | Frame pointer (callee-saved) |
| 30 | `x30` (`lr`) | Link register / return address |
| 31 | `sp` | Stack pointer |
| 64–95 | `v0`–`v31` | SIMD/FP registers |

### RISC-V 64 Register Mapping

| DWARF Number | Register | ABI Name | ABI Role |
|---|---|---|---|
| 0 | `x0` | `zero` | Hardwired zero |
| 1 | `x1` | `ra` | Return address |
| 2 | `x2` | `sp` | Stack pointer |
| 3 | `x3` | `gp` | Global pointer |
| 4 | `x4` | `tp` | Thread pointer |
| 5–7 | `x5`–`x7` | `t0`–`t2` | Temporaries (caller-saved) |
| 8 | `x8` | `s0`/`fp` | Callee-saved / frame pointer |
| 9 | `x9` | `s1` | Callee-saved |
| 10–17 | `x10`–`x17` | `a0`–`a7` | Arguments / return values |
| 18–27 | `x18`–`x27` | `s2`–`s11` | Callee-saved |
| 28–31 | `x28`–`x31` | `t3`–`t6` | Temporaries (caller-saved) |
| 32–63 | `f0`–`f31` | Various | Floating-point registers |

---

## GDB and LLDB Compatibility

The DWARF v4 output from `bcc` is designed to work correctly with both GDB and LLDB for source-level debugging. This section documents the minimum requirements and common pitfalls.

### Minimum Required Sections

For basic source-level debugging (setting breakpoints by file/line, stepping, inspecting variables), debuggers require:

| Section | GDB | LLDB | Purpose |
|---|---|---|---|
| `.debug_info` | Required | Required | Type and variable descriptions |
| `.debug_abbrev` | Required | Required | DIE schema lookup table |
| `.debug_line` | Required | Required | Source file/line ↔ address mapping |
| `.debug_str` | Optional (beneficial) | Optional (beneficial) | String deduplication |
| `.debug_frame` | Recommended | Recommended | Stack unwinding |
| `.debug_aranges` | Optional (performance) | Optional (performance) | Fast CU address lookup |
| `.debug_loc` | Optional (for optimized code) | Optional (for optimized code) | Multi-location variables |

GDB can fall back to `.eh_frame` (in the loadable portion of the ELF) if `.debug_frame` is absent, but `bcc` emits `.debug_frame` for completeness.

LLDB tends to perform stricter validation of DWARF structures than GDB, making it a useful compatibility target during development.

### Common Pitfalls

The following implementation errors are known to cause debugger failures and should be carefully avoided:

1. **Incorrect abbreviation table references** — If a CU header's `debug_abbrev_offset` does not point to a valid abbreviation table, debuggers report "corrupt DWARF" or "invalid abbreviation code" errors. Verify that the offset points to the correct position in `.debug_abbrev`.

2. **Missing null DIE terminator** — Every DIE that has `DW_CHILDREN_yes` in its abbreviation must have its children list terminated by a single null byte (abbreviation code 0). Missing this terminator causes debuggers to misparse the entire DIE tree.

3. **Unterminated line number program** — Every line program sequence must end with a `DW_LNE_end_sequence` extended opcode. Without this, the debugger cannot determine where the address sequence ends and may produce garbage mappings or crash.

4. **Wrong `address_size` in CU header** — The `address_size` field must match the target: `4` for i686, `8` for x86-64/AArch64/RISC-V 64. An incorrect value causes all `DW_FORM_addr` attributes to be read at the wrong width, corrupting the entire CU.

5. **Mismatched DWARF register numbers** — The register numbers used in `.debug_info` location expressions (e.g., `DW_OP_reg6` for rbp on x86-64) must match the register numbers used in `.debug_frame` CFI instructions. Using hardware encoding instead of DWARF encoding (or mixing up architectures) causes the debugger to display incorrect register values and fail unwinding.

6. **Incorrect `DW_FORM_ref4` offsets** — Type references within `.debug_info` use CU-relative byte offsets. Off-by-one errors in offset computation cause type lookups to fail silently (debugger shows `<incomplete type>`) or crash.

7. **`line_base` / `line_range` mismatch** — If the line program header's `line_base` and `line_range` values do not match the formula used to generate special opcodes, debuggers will compute wrong line numbers. Always ensure the encoder and decoder use identical parameters.

8. **Missing `DW_AT_stmt_list`** — If the `DW_TAG_compile_unit` DIE omits the `DW_AT_stmt_list` attribute, debuggers cannot correlate addresses with source lines. This attribute must always be present and must point to the correct offset in `.debug_line`.

### Testing DWARF Output

Verify correctness of generated DWARF with these tools:

```bash
# Dump all DWARF sections
readelf --debug-dump=info,abbrev,line,str,aranges,frame,loc program

# Verify DWARF structure integrity
dwarfdump --verify program

# Test with GDB
gdb ./program -ex "break main" -ex "run" -ex "list" -ex "info locals" -ex "bt" -ex "quit"

# Test with LLDB
lldb ./program -o "breakpoint set --name main" -o "run" -o "frame variable" -o "thread backtrace" -o "quit"
```

---

## References

- DWARF Debugging Information Format, Version 4 (June 10, 2010) — [dwarfstd.org](https://dwarfstd.org/)
- System V ABI (AMD64 Architecture Processor Supplement) — DWARF register number assignments for x86-64
- AAPCS64 (Arm Architecture Procedure Call Standard for 64-bit) — DWARF register number assignments for AArch64
- RISC-V ABIs Specification — DWARF register number assignments for RISC-V
- `docs/architecture.md` Section 3.8 — Debug Info Module overview
- `docs/targets.md` — Per-architecture ABI details and register files
- `docs/internals/linker.md` — Integrated linker internals (DWARF section consumption)
