// DWARF v4 .debug_line section — Line Number Program generator.
//
// This module generates the line number program that maps machine code
// addresses to source file locations (file, line, column). The program
// uses a state machine encoding with standard, extended, and special
// opcodes for compact representation. This data enables source-level
// stepping in gdb and lldb.
//
// The line number program is encoded per DWARF v4 §6.2. The header
// specifies file/directory tables and configuration parameters, followed
// by an opcode stream that drives a virtual state machine. Consumers
// (debuggers) replay the opcodes to reconstruct the address-to-source
// mapping matrix.
//
// CONSTRAINTS:
// - Zero external dependencies: only std and internal crate modules.
// - No unsafe code: pure data structure generation and byte encoding.
// - DWARF v4 compliance: 32-bit DWARF format, version 4 headers.
// - Cross-architecture: works for x86-64 (addr=8, min_insn=1),
//   i686 (addr=4, min_insn=1), AArch64 (addr=8, min_insn=4),
//   and RISC-V 64 (addr=8, min_insn=2).

use super::dwarf::{
    encode_sleb128_to, encode_uleb128_to, write_address, write_string, write_u16_le, write_u32_le,
    write_u8,
};
use crate::common::source_map::{FileId, SourceMap};

// ---------------------------------------------------------------------------
// DWARF v4 Line Number Standard Opcodes (§6.2.5.2)
// ---------------------------------------------------------------------------

/// Emit the current state machine registers as a row in the line matrix.
const DW_LNS_COPY: u8 = 0x01;
/// Advance the program counter by an unsigned LEB128 operand multiplied by
/// `minimum_instruction_length`.
const DW_LNS_ADVANCE_PC: u8 = 0x02;
/// Advance the line register by a signed LEB128 operand (may be negative).
const DW_LNS_ADVANCE_LINE: u8 = 0x03;
/// Set the file register to the unsigned LEB128 operand value.
const DW_LNS_SET_FILE: u8 = 0x04;
/// Set the column register to the unsigned LEB128 operand value.
const DW_LNS_SET_COLUMN: u8 = 0x05;
/// Toggle the `is_stmt` flag.
const DW_LNS_NEGATE_STMT: u8 = 0x06;
/// Mark the current address as the beginning of a basic block.
const DW_LNS_SET_BASIC_BLOCK: u8 = 0x07;
/// Advance the program counter by a fixed amount computed from the header
/// parameters: `((255 - opcode_base) / line_range) * minimum_instruction_length`.
const DW_LNS_CONST_ADVANCE_PC: u8 = 0x08;
/// Advance the program counter by an unsigned 16-bit value (not scaled).
const DW_LNS_FIXED_ADVANCE_PC: u8 = 0x09;
/// Mark the end of the function prologue (breakpoint-friendly position).
const DW_LNS_SET_PROLOGUE_END: u8 = 0x0a;
/// Mark the beginning of the function epilogue.
const DW_LNS_SET_EPILOGUE_BEGIN: u8 = 0x0b;
/// Set the ISA (Instruction Set Architecture) register.
const DW_LNS_SET_ISA: u8 = 0x0c;

// ---------------------------------------------------------------------------
// DWARF v4 Line Number Extended Opcodes (§6.2.5.3)
// ---------------------------------------------------------------------------

/// Mark the end of a contiguous address sequence. Resets the state machine.
const DW_LNE_END_SEQUENCE: u8 = 0x01;
/// Set the address register to a specific absolute address.
const DW_LNE_SET_ADDRESS: u8 = 0x02;
/// Define a new source file entry inline in the opcode stream.
const DW_LNE_DEFINE_FILE: u8 = 0x03;
/// Set the discriminator register for disambiguation of same-line different
/// basic blocks (DWARF v4 addition).
const DW_LNE_SET_DISCRIMINATOR: u8 = 0x04;

// ---------------------------------------------------------------------------
// Default Line Program Parameters
// ---------------------------------------------------------------------------

/// Default line_base for special opcode calculation. Covers negative line
/// deltas down to -5, which is common for backwards branches in loops.
const DEFAULT_LINE_BASE: i8 = -5;

/// Default line_range for special opcode calculation. With line_base=-5,
/// this covers line deltas in the range [-5, +8] (14 values).
const DEFAULT_LINE_RANGE: u8 = 14;

/// First special opcode value. Opcodes 0 through 12 are reserved for
/// standard opcodes (0 = extended opcode prefix, 1-12 = standard opcodes),
/// so the first available special opcode is 13.
const DEFAULT_OPCODE_BASE: u8 = 13;

/// Standard opcode operand counts, one entry per standard opcode (1 through 12).
/// Index 0 corresponds to DW_LNS_copy (opcode 1), index 11 to DW_LNS_set_isa
/// (opcode 12). The value at each index is the number of LEB128 operands that
/// follow the opcode byte.
const STANDARD_OPCODE_LENGTHS: [u8; 12] = [
    0, // DW_LNS_copy            (0x01) — 0 operands
    1, // DW_LNS_advance_pc      (0x02) — 1 ULEB128 operand
    1, // DW_LNS_advance_line    (0x03) — 1 SLEB128 operand
    1, // DW_LNS_set_file        (0x04) — 1 ULEB128 operand
    1, // DW_LNS_set_column      (0x05) — 1 ULEB128 operand
    0, // DW_LNS_negate_stmt     (0x06) — 0 operands
    0, // DW_LNS_set_basic_block (0x07) — 0 operands
    0, // DW_LNS_const_advance_pc(0x08) — 0 operands
    1, // DW_LNS_fixed_advance_pc(0x09) — 1 uhalf operand
    0, // DW_LNS_set_prologue_end(0x0a) — 0 operands
    0, // DW_LNS_set_epilogue_begin(0x0b) — 0 operands
    1, // DW_LNS_set_isa         (0x0c) — 1 ULEB128 operand
];

// ---------------------------------------------------------------------------
// FileEntry — source file entry in the line program header
// ---------------------------------------------------------------------------

/// A single source file entry in the line number program header's file table.
///
/// Each entry records the file name, its parent directory index into the
/// include directory table, and optional metadata (modification time and
/// file size). A value of 0 for `modification_time` or `file_size` means
/// "unknown" per the DWARF specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileEntry {
    /// File name (relative to the directory indicated by `directory_index`).
    pub name: String,
    /// Index into the include directory table. 0 means the compilation
    /// directory (the directory that was current when the compiler was invoked).
    pub directory_index: u32,
    /// Last modification time in an implementation-defined encoding.
    /// 0 means unknown.
    pub modification_time: u64,
    /// File size in bytes. 0 means unknown.
    pub file_size: u64,
}

// ---------------------------------------------------------------------------
// LineProgramHeader — line number program header structure
// ---------------------------------------------------------------------------

/// The DWARF v4 line number program header (§6.2.4).
///
/// This header appears at the start of each line number program in the
/// `.debug_line` section. It contains the configuration parameters that
/// govern the state machine, plus the file and directory tables needed
/// to map file indices to actual source paths.
#[derive(Debug, Clone)]
pub struct LineProgramHeader {
    /// DWARF version number (4 for DWARF v4).
    pub version: u16,
    /// Size of a target address in bytes (4 for i686, 8 for 64-bit targets).
    pub address_size: u8,
    /// Minimum length of an instruction in bytes on the target machine.
    /// 1 for x86/x86-64 (variable-length), 4 for AArch64 (fixed-width),
    /// 2 for RISC-V 64 (compressed instructions minimum).
    pub minimum_instruction_length: u8,
    /// Maximum number of individual operations that may be encoded in a
    /// single instruction. Always 1 for non-VLIW architectures.
    pub maximum_operations_per_instruction: u8,
    /// Initial value of the `is_stmt` register. `true` means addresses are
    /// recommended breakpoint positions by default.
    pub default_is_stmt: bool,
    /// Base value for the line increment in special opcodes. Typically
    /// negative to allow encoding backwards line moves (e.g., -5).
    pub line_base: i8,
    /// Range of line increments encodable by special opcodes. With
    /// `line_base = -5` and `line_range = 14`, line deltas -5 through +8
    /// can be encoded.
    pub line_range: u8,
    /// First special opcode value. Standard opcodes are numbered 1 through
    /// `opcode_base - 1`.
    pub opcode_base: u8,
    /// Number of LEB128 operands for each standard opcode. Has
    /// `opcode_base - 1` entries (one per standard opcode, indexed from 0).
    pub standard_opcode_lengths: Vec<u8>,
    /// Include directories table. Index 0 is implicitly the compilation
    /// directory; the entries here start at index 1.
    pub include_directories: Vec<String>,
    /// File name entries. Each entry references a directory by index.
    pub file_entries: Vec<FileEntry>,
}

impl LineProgramHeader {
    /// Creates a new header with architecture-appropriate defaults.
    ///
    /// # Arguments
    /// - `address_size` — Target address size in bytes (4 or 8).
    /// - `min_instruction_length` — Minimum instruction length for the target.
    fn new(address_size: u8, min_instruction_length: u8) -> Self {
        LineProgramHeader {
            version: 4,
            address_size,
            minimum_instruction_length: min_instruction_length,
            maximum_operations_per_instruction: 1,
            default_is_stmt: true,
            line_base: DEFAULT_LINE_BASE,
            line_range: DEFAULT_LINE_RANGE,
            opcode_base: DEFAULT_OPCODE_BASE,
            standard_opcode_lengths: STANDARD_OPCODE_LENGTHS.to_vec(),
            include_directories: Vec::new(),
            file_entries: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// LineNumberState — DWARF line number state machine registers
// ---------------------------------------------------------------------------

/// The complete set of registers maintained by the DWARF v4 line number
/// state machine (§6.2.2).
///
/// Debuggers replay the line number program opcodes against this state to
/// reconstruct the address-to-source mapping matrix. Each `copy` or
/// special opcode emits the current register values as a matrix row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineNumberState {
    /// Current machine code address.
    pub address: u64,
    /// Operation index within a VLIW instruction (always 0 for non-VLIW).
    pub op_index: u32,
    /// Current source file (1-based index into the file table).
    pub file: u32,
    /// Current source line number (1-based).
    pub line: u32,
    /// Current source column number (0 means unknown, 1+ means column).
    pub column: u32,
    /// Whether the current address is a recommended breakpoint location.
    pub is_stmt: bool,
    /// Whether the current address is the start of a basic block.
    pub basic_block: bool,
    /// Whether this row marks the first address past the end of a
    /// contiguous code sequence.
    pub end_sequence: bool,
    /// Whether the current address is where function prologue code ends.
    pub prologue_end: bool,
    /// Whether the current address is where function epilogue code begins.
    pub epilogue_begin: bool,
    /// Instruction Set Architecture identifier for the current instruction.
    pub isa: u32,
    /// Discriminator for disambiguating multiple blocks on the same source
    /// line (DWARF v4 addition).
    pub discriminator: u32,
}

impl LineNumberState {
    /// Creates a new state machine initialized to DWARF v4 defaults.
    ///
    /// Per §6.2.2, the initial register values are:
    /// - address = 0, op_index = 0, file = 1, line = 1, column = 0
    /// - is_stmt = default_is_stmt, all other flags = false, isa = 0,
    ///   discriminator = 0
    fn initial(default_is_stmt: bool) -> Self {
        LineNumberState {
            address: 0,
            op_index: 0,
            file: 1,
            line: 1,
            column: 0,
            is_stmt: default_is_stmt,
            basic_block: false,
            end_sequence: false,
            prologue_end: false,
            epilogue_begin: false,
            isa: 0,
            discriminator: 0,
        }
    }

    /// Resets flags that are cleared after a row-emitting operation (copy or
    /// special opcode). Per DWARF v4 §6.2.5.1, after appending a row:
    /// `basic_block`, `prologue_end`, `epilogue_begin` are set to false,
    /// and `discriminator` is set to 0.
    fn reset_row_flags(&mut self) {
        self.basic_block = false;
        self.prologue_end = false;
        self.epilogue_begin = false;
        self.discriminator = 0;
    }
}

// ---------------------------------------------------------------------------
// LineMappingEntry — a single address-to-source mapping
// ---------------------------------------------------------------------------

/// Represents a single mapping between a machine code address and a source
/// location, used as input to the line program builder.
///
/// The code generator produces a list of these entries for each function, and
/// the line program builder encodes them into the compact DWARF opcode stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineMappingEntry {
    /// Machine code address (absolute offset from the start of the .text
    /// section, or absolute virtual address).
    pub address: u64,
    /// File index in the line program's file table (1-based).
    pub file_id: u32,
    /// Source line number (1-based).
    pub line: u32,
    /// Source column number (0 means unknown, 1+ means column).
    pub column: u32,
    /// Whether this address is a recommended breakpoint position.
    pub is_stmt: bool,
    /// Whether this address marks the end of the function prologue.
    pub is_prologue_end: bool,
    /// Whether this entry marks the end of a contiguous code sequence.
    /// When true, the builder emits an end_sequence opcode after this entry.
    pub is_end_sequence: bool,
}

// ---------------------------------------------------------------------------
// LineProgramBuilder — low-level line program construction API
// ---------------------------------------------------------------------------

/// Builder for constructing a DWARF v4 line number program opcode stream.
///
/// The builder maintains the current state machine registers and provides
/// methods to emit standard, extended, and special opcodes. Each method
/// updates the internal state and appends the encoded bytes to the opcode
/// buffer.
///
/// # Usage
/// ```ignore
/// let mut builder = LineProgramBuilder::new(8, 1); // x86-64
/// let file = builder.add_file("main.c", 0);
/// builder.set_address(0x401000);
/// builder.add_line_entry(0x401000, 1, 1, file);
/// builder.add_line_entry(0x401008, 2, 1, file);
/// builder.end_sequence();
/// let bytes = builder.serialize();
/// ```
pub struct LineProgramBuilder {
    /// The line program header containing file/directory tables and config.
    header: LineProgramHeader,
    /// Encoded opcode byte stream accumulated by builder methods.
    opcodes: Vec<u8>,
    /// Current state machine register values.
    current_state: LineNumberState,
}

impl LineProgramBuilder {
    /// Creates a new line program builder with architecture-appropriate defaults.
    ///
    /// # Arguments
    /// - `address_size` — Target address size (4 for i686, 8 for x86-64/AArch64/RISC-V 64).
    /// - `min_instruction_length` — Minimum instruction length for the target
    ///   (1 for x86/x86-64, 4 for AArch64, 2 for RISC-V 64).
    pub fn new(address_size: u8, min_instruction_length: u8) -> Self {
        let header = LineProgramHeader::new(address_size, min_instruction_length);
        let default_is_stmt = header.default_is_stmt;
        LineProgramBuilder {
            header,
            opcodes: Vec::new(),
            current_state: LineNumberState::initial(default_is_stmt),
        }
    }

    // -- File and Directory Table Management ---------------------------------

    /// Adds a file entry to the line program's file table.
    ///
    /// # Arguments
    /// - `name` — File name (relative to the directory at `dir_index`).
    /// - `dir_index` — Index into the include directory table (0 = compilation dir).
    ///
    /// # Returns
    /// The 1-based file index that can be used with `set_file`.
    pub fn add_file(&mut self, name: &str, dir_index: u32) -> u32 {
        self.header.file_entries.push(FileEntry {
            name: name.to_owned(),
            directory_index: dir_index,
            modification_time: 0,
            file_size: 0,
        });
        // File indices are 1-based in DWARF.
        self.header.file_entries.len() as u32
    }

    /// Adds an include directory to the line program's directory table.
    ///
    /// # Arguments
    /// - `dir` — Directory path string.
    ///
    /// # Returns
    /// The 1-based directory index (0 is always the compilation directory).
    pub fn add_directory(&mut self, dir: &str) -> u32 {
        self.header.include_directories.push(dir.to_owned());
        // Directory indices are 1-based in DWARF (0 = compilation directory).
        self.header.include_directories.len() as u32
    }

    // -- Extended Opcodes ---------------------------------------------------

    /// Emits a `DW_LNE_set_address` extended opcode, setting the address
    /// register to an absolute value.
    ///
    /// Encoding: `0x00` (extended opcode prefix) + ULEB128(1 + address_size) +
    /// `0x02` (DW_LNE_set_address) + address bytes.
    pub fn set_address(&mut self, address: u64) {
        let addr_size = self.header.address_size;
        // Extended opcode prefix byte.
        write_u8(&mut self.opcodes, 0x00);
        // Total length of extended opcode data: 1 byte for the opcode itself
        // plus the address bytes.
        encode_uleb128_to(&mut self.opcodes, 1 + addr_size as u64);
        // Extended opcode type.
        write_u8(&mut self.opcodes, DW_LNE_SET_ADDRESS);
        // Absolute address in target byte order (little-endian).
        write_address(&mut self.opcodes, address, addr_size);
        // Update state machine register.
        self.current_state.address = address;
        self.current_state.op_index = 0;
    }

    /// Emits a `DW_LNE_end_sequence` extended opcode, marking the end of a
    /// contiguous code address range.
    ///
    /// After this opcode, the state machine is reset to its initial values.
    /// Each function or contiguous code region should be terminated with this.
    pub fn end_sequence(&mut self) {
        // Extended opcode prefix.
        write_u8(&mut self.opcodes, 0x00);
        // Length = 1 (just the opcode byte, no additional operands).
        encode_uleb128_to(&mut self.opcodes, 1);
        // Extended opcode type.
        write_u8(&mut self.opcodes, DW_LNE_END_SEQUENCE);
        // Reset state machine to initial values.
        self.current_state = LineNumberState::initial(self.header.default_is_stmt);
    }

    // -- Standard Opcodes ---------------------------------------------------

    /// Emits a `DW_LNS_advance_pc` opcode, advancing the address register by
    /// `delta` bytes.
    ///
    /// The delta is divided by `minimum_instruction_length` before encoding
    /// as a ULEB128 operand, per the DWARF specification.
    pub fn advance_pc(&mut self, delta: u64) {
        let min_len = self.header.minimum_instruction_length as u64;
        let operation_advance = if min_len > 0 { delta / min_len } else { delta };
        write_u8(&mut self.opcodes, DW_LNS_ADVANCE_PC);
        encode_uleb128_to(&mut self.opcodes, operation_advance);
        self.current_state.address += delta;
    }

    /// Emits a `DW_LNS_advance_line` opcode, advancing (or retreating) the
    /// line register by `delta`.
    ///
    /// The delta is encoded as a signed LEB128 value, allowing negative line
    /// changes (e.g., backwards jumps in loops or macros).
    pub fn advance_line(&mut self, delta: i32) {
        write_u8(&mut self.opcodes, DW_LNS_ADVANCE_LINE);
        encode_sleb128_to(&mut self.opcodes, delta as i64);
        self.current_state.line = (self.current_state.line as i64 + delta as i64) as u32;
    }

    /// Emits a `DW_LNS_set_file` opcode, setting the file register to the
    /// specified 1-based file index.
    pub fn set_file(&mut self, file_index: u32) {
        write_u8(&mut self.opcodes, DW_LNS_SET_FILE);
        encode_uleb128_to(&mut self.opcodes, file_index as u64);
        self.current_state.file = file_index;
    }

    /// Emits a `DW_LNS_set_column` opcode, setting the column register.
    pub fn set_column(&mut self, column: u32) {
        write_u8(&mut self.opcodes, DW_LNS_SET_COLUMN);
        encode_uleb128_to(&mut self.opcodes, column as u64);
        self.current_state.column = column;
    }

    /// Emits a `DW_LNS_copy` opcode, appending the current state as a row
    /// in the line number matrix.
    ///
    /// After the row is emitted, the `basic_block`, `prologue_end`,
    /// `epilogue_begin` flags are reset to false and `discriminator` is
    /// reset to 0.
    pub fn copy(&mut self) {
        write_u8(&mut self.opcodes, DW_LNS_COPY);
        self.current_state.reset_row_flags();
    }

    /// Emits a `DW_LNS_set_prologue_end` opcode, marking the next matrix
    /// row as the point where function prologue code ends.
    pub fn set_prologue_end(&mut self) {
        write_u8(&mut self.opcodes, DW_LNS_SET_PROLOGUE_END);
        self.current_state.prologue_end = true;
    }

    // -- Special Opcode Support ---------------------------------------------

    /// Attempts to encode a combined address advance + line advance + copy
    /// as a single special opcode byte.
    ///
    /// The DWARF v4 special opcode formula (§6.2.5.1):
    /// ```text
    /// special_opcode = (line_delta - line_base) + (line_range * address_advance) + opcode_base
    /// ```
    ///
    /// Returns `Some(opcode)` if the values fit, `None` otherwise.
    fn try_special_opcode(&self, addr_delta: u64, line_delta: i32) -> Option<u8> {
        let line_base = self.header.line_base as i32;
        let line_range = self.header.line_range as i32;
        let opcode_base = self.header.opcode_base as i32;
        let min_len = self.header.minimum_instruction_length as u64;

        // Check that the line delta is within the encodable range.
        if line_delta < line_base || line_delta >= line_base + line_range {
            return None;
        }

        // Compute operation advance (address delta divided by min instruction length).
        if min_len == 0 {
            return None;
        }
        // Address delta must be exactly divisible by minimum instruction length
        // for a special opcode (no remainder allowed).
        if addr_delta % min_len != 0 {
            return None;
        }
        let address_advance = addr_delta / min_len;

        // Compute the special opcode value.
        let opcode = (line_delta - line_base) + (line_range * address_advance as i32) + opcode_base;

        // The opcode must fit in a single byte [opcode_base, 255].
        if opcode >= opcode_base && opcode <= 255 {
            Some(opcode as u8)
        } else {
            None
        }
    }

    /// High-level method that emits the most compact encoding for a new
    /// line mapping entry.
    ///
    /// This method determines the optimal encoding strategy:
    /// 1. If the file changed, emit `set_file` first.
    /// 2. If the column changed, emit `set_column` first.
    /// 3. Try to encode the address and line advance as a single special opcode.
    /// 4. Fall back to standard opcodes (`advance_pc` + `advance_line` + `copy`).
    ///
    /// # Arguments
    /// - `address` — Absolute machine code address.
    /// - `line` — Source line number (1-based).
    /// - `column` — Source column number (0 = unknown).
    /// - `file` — File index in the line program (1-based).
    pub fn add_line_entry(&mut self, address: u64, line: u32, column: u32, file: u32) {
        // Step 1: If the file changed, emit a set_file opcode.
        if file != self.current_state.file {
            self.set_file(file);
        }

        // Step 2: If the column changed, emit a set_column opcode.
        if column != self.current_state.column {
            self.set_column(column);
        }

        // Step 3: Calculate deltas from the current state.
        let addr_delta = address.saturating_sub(self.current_state.address);
        let line_delta = line as i64 - self.current_state.line as i64;

        // Step 4: Try special opcode encoding (most compact — single byte).
        if let Some(opcode) = self.try_special_opcode(addr_delta, line_delta as i32) {
            write_u8(&mut self.opcodes, opcode);
            // A special opcode implicitly advances the address and line, then
            // emits a row (copy). Update state accordingly.
            self.current_state.address = address;
            self.current_state.line = line;
            self.current_state.reset_row_flags();
            return;
        }

        // Step 5: Fall back to standard opcodes.
        if addr_delta > 0 {
            self.advance_pc(addr_delta);
        }
        if line_delta != 0 {
            self.advance_line(line_delta as i32);
        }
        self.copy();
    }

    // -- Serialization ------------------------------------------------------

    /// Serializes the complete `.debug_line` section content for this line
    /// number program unit.
    ///
    /// The output includes:
    /// 1. Unit header with `unit_length` and `header_length` fields.
    /// 2. Line program header (configuration, directory table, file table).
    /// 3. The accumulated opcode byte stream.
    ///
    /// The `unit_length` and `header_length` fields are patched after
    /// serialization to reflect actual sizes.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // -- Unit length placeholder (4 bytes, patched below) ---------------
        let unit_length_pos = buf.len();
        write_u32_le(&mut buf, 0); // Placeholder for unit_length.

        // -- Version (2 bytes) ----------------------------------------------
        write_u16_le(&mut buf, self.header.version);

        // -- Header length placeholder (4 bytes, patched below) -------------
        let header_length_pos = buf.len();
        write_u32_le(&mut buf, 0); // Placeholder for header_length.

        // -- Header fields --------------------------------------------------
        let header_content_start = buf.len();

        write_u8(&mut buf, self.header.minimum_instruction_length);
        write_u8(&mut buf, self.header.maximum_operations_per_instruction);
        write_u8(&mut buf, if self.header.default_is_stmt { 1 } else { 0 });
        write_u8(&mut buf, self.header.line_base as u8);
        write_u8(&mut buf, self.header.line_range);
        write_u8(&mut buf, self.header.opcode_base);

        // Standard opcode lengths (opcode_base - 1 entries).
        for &length in &self.header.standard_opcode_lengths {
            write_u8(&mut buf, length);
        }

        // -- Include directory table ----------------------------------------
        // Each entry is a null-terminated string. The table is terminated by
        // a single null byte (empty string).
        for dir in &self.header.include_directories {
            write_string(&mut buf, dir);
        }
        write_u8(&mut buf, 0x00); // Directory table terminator.

        // -- File name table ------------------------------------------------
        // Each entry: null-terminated name + ULEB128 dir_index +
        // ULEB128 mod_time + ULEB128 file_size.
        // Terminated by a single null byte (empty entry).
        for entry in &self.header.file_entries {
            write_string(&mut buf, &entry.name);
            encode_uleb128_to(&mut buf, entry.directory_index as u64);
            encode_uleb128_to(&mut buf, entry.modification_time);
            encode_uleb128_to(&mut buf, entry.file_size);
        }
        write_u8(&mut buf, 0x00); // File table terminator.

        // -- Patch header_length -------------------------------------------
        // header_length measures from just after the header_length field to
        // the first byte of the opcode stream (i.e., the end of the file table).
        let header_length = (buf.len() - header_content_start) as u32;
        buf[header_length_pos..header_length_pos + 4].copy_from_slice(&header_length.to_le_bytes());

        // -- Opcode stream --------------------------------------------------
        buf.extend_from_slice(&self.opcodes);

        // -- Patch unit_length ---------------------------------------------
        // unit_length measures everything after the initial 4-byte length field.
        let unit_length = (buf.len() - (unit_length_pos + 4)) as u32;
        buf[unit_length_pos..unit_length_pos + 4].copy_from_slice(&unit_length.to_le_bytes());

        buf
    }

    // -- High-Level Construction from Source Mappings ------------------------

    /// Constructs a complete line program from a list of address-to-source
    /// mapping entries and a source map.
    ///
    /// This factory method:
    /// 1. Collects unique files referenced by the mappings from the source map.
    /// 2. Registers them in the line program's file/directory tables.
    /// 3. Groups mappings into contiguous sequences (split by `is_end_sequence`
    ///    entries or address gaps).
    /// 4. For each sequence: emits `set_address`, individual line entries, and
    ///    `end_sequence`.
    ///
    /// # Arguments
    /// - `mappings` — Address-to-source mapping entries (need not be sorted).
    /// - `source_map` — The compiler's source file registry for file path lookups.
    /// - `address_size` — Target address size in bytes.
    /// - `min_instruction_length` — Minimum instruction length for the target.
    pub fn from_source_mappings(
        mappings: &[LineMappingEntry],
        source_map: &SourceMap,
        address_size: u8,
        min_instruction_length: u8,
    ) -> Self {
        let mut builder = LineProgramBuilder::new(address_size, min_instruction_length);

        if mappings.is_empty() {
            return builder;
        }

        // Collect unique file IDs and register them in the line program.
        // We build a mapping from LineMappingEntry.file_id (which corresponds
        // to the SourceMap's FileId index) to the line program's 1-based file index.
        let mut file_id_map: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for entry in mappings {
            if !file_id_map.contains_key(&entry.file_id) {
                let source_file_id = FileId(entry.file_id);
                let path = source_map.get_file_path(source_file_id);
                let path_str = path.to_string_lossy();

                // Extract directory and file name components.
                let (dir_str, file_name) = if let Some(parent) = path.parent() {
                    let parent_str = parent.to_string_lossy().to_string();
                    let name = path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| path_str.to_string());
                    (parent_str, name)
                } else {
                    (String::new(), path_str.to_string())
                };

                // Register directory (if non-empty).
                let dir_index = if dir_str.is_empty() {
                    0 // Compilation directory.
                } else {
                    // Check if directory already registered.
                    let existing = builder
                        .header
                        .include_directories
                        .iter()
                        .position(|d| d == &dir_str);
                    match existing {
                        Some(idx) => (idx as u32) + 1,
                        None => builder.add_directory(&dir_str),
                    }
                };

                // Register file.
                let lp_file_id = builder.add_file(&file_name, dir_index);
                file_id_map.insert(entry.file_id, lp_file_id);
            }
        }

        // Sort mappings by address for sequential processing.
        let mut sorted_mappings: Vec<&LineMappingEntry> = mappings.iter().collect();
        sorted_mappings.sort_by_key(|m| m.address);

        // Process mappings, splitting into sequences.
        let mut in_sequence = false;

        for mapping in &sorted_mappings {
            let lp_file_id = file_id_map.get(&mapping.file_id).copied().unwrap_or(1);

            if !in_sequence {
                // Start a new sequence with set_address.
                builder.set_address(mapping.address);
                in_sequence = true;
            }

            // Set prologue_end if flagged.
            if mapping.is_prologue_end {
                builder.set_prologue_end();
            }

            // Emit the line entry.
            builder.add_line_entry(mapping.address, mapping.line, mapping.column, lp_file_id);

            // End the sequence if flagged.
            if mapping.is_end_sequence {
                builder.end_sequence();
                in_sequence = false;
            }
        }

        // If the last mapping didn't end the sequence, end it now.
        if in_sequence {
            builder.end_sequence();
        }

        builder
    }
}

// ---------------------------------------------------------------------------
// LineProgramEmitter — high-level emission API
// ---------------------------------------------------------------------------

/// High-level emitter that produces a complete `.debug_line` section from
/// address-to-source mappings.
///
/// This struct provides a simplified API for the common case where the
/// caller has a list of `LineMappingEntry` values and wants the serialized
/// DWARF bytes. It delegates to `LineProgramBuilder` internally.
pub struct LineProgramEmitter {
    /// Target address size in bytes (4 or 8).
    address_size: u8,
    /// Minimum instruction length for the target architecture.
    min_instruction_length: u8,
}

impl LineProgramEmitter {
    /// Creates a new emitter configured for the specified target architecture.
    ///
    /// # Arguments
    /// - `address_size` — 4 for i686, 8 for x86-64/AArch64/RISC-V 64.
    /// - `min_instruction_length` — 1 for x86/x86-64, 4 for AArch64,
    ///   2 for RISC-V 64.
    pub fn new(address_size: u8, min_instruction_length: u8) -> Self {
        LineProgramEmitter {
            address_size,
            min_instruction_length,
        }
    }

    /// Emits a complete `.debug_line` section from the given source mappings.
    ///
    /// This is the primary entry point for generating DWARF line information.
    /// It constructs a `LineProgramBuilder`, populates it from the mappings
    /// and source map, serializes the result, and returns the raw bytes ready
    /// for inclusion in the ELF output.
    ///
    /// # Arguments
    /// - `mappings` — Address-to-source mapping entries from code generation.
    /// - `source_map` — The compiler's source file registry.
    ///
    /// # Returns
    /// A `Vec<u8>` containing the complete `.debug_line` section bytes.
    pub fn emit(&self, mappings: &[LineMappingEntry], source_map: &SourceMap) -> Vec<u8> {
        let builder = LineProgramBuilder::from_source_mappings(
            mappings,
            source_map,
            self.address_size,
            self.min_instruction_length,
        );
        builder.serialize()
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helper to create a builder for each architecture -------------------

    fn x86_64_builder() -> LineProgramBuilder {
        LineProgramBuilder::new(8, 1)
    }

    fn i686_builder() -> LineProgramBuilder {
        LineProgramBuilder::new(4, 1)
    }

    fn aarch64_builder() -> LineProgramBuilder {
        LineProgramBuilder::new(8, 4)
    }

    fn riscv64_builder() -> LineProgramBuilder {
        LineProgramBuilder::new(8, 2)
    }

    // -----------------------------------------------------------------------
    // Header serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_header_version_is_4() {
        let builder = x86_64_builder();
        let data = builder.serialize();
        // Version is at offset 4 (after the 4-byte unit_length).
        let version = u16::from_le_bytes([data[4], data[5]]);
        assert_eq!(version, 4);
    }

    #[test]
    fn test_header_x86_64_parameters() {
        let builder = x86_64_builder();
        assert_eq!(builder.header.address_size, 8);
        assert_eq!(builder.header.minimum_instruction_length, 1);
        assert_eq!(builder.header.opcode_base, DEFAULT_OPCODE_BASE);
        assert_eq!(
            builder.header.standard_opcode_lengths.len(),
            (DEFAULT_OPCODE_BASE - 1) as usize
        );
    }

    #[test]
    fn test_header_i686_parameters() {
        let builder = i686_builder();
        assert_eq!(builder.header.address_size, 4);
        assert_eq!(builder.header.minimum_instruction_length, 1);
    }

    #[test]
    fn test_header_aarch64_parameters() {
        let builder = aarch64_builder();
        assert_eq!(builder.header.address_size, 8);
        assert_eq!(builder.header.minimum_instruction_length, 4);
    }

    #[test]
    fn test_header_riscv64_parameters() {
        let builder = riscv64_builder();
        assert_eq!(builder.header.address_size, 8);
        assert_eq!(builder.header.minimum_instruction_length, 2);
    }

    #[test]
    fn test_opcode_base_and_lengths_consistent() {
        let builder = x86_64_builder();
        assert_eq!(
            builder.header.standard_opcode_lengths.len(),
            (builder.header.opcode_base - 1) as usize
        );
    }

    // -----------------------------------------------------------------------
    // File and directory table tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_file_returns_one_based_index() {
        let mut builder = x86_64_builder();
        let idx1 = builder.add_file("main.c", 0);
        let idx2 = builder.add_file("util.c", 0);
        assert_eq!(idx1, 1);
        assert_eq!(idx2, 2);
    }

    #[test]
    fn test_add_directory_returns_one_based_index() {
        let mut builder = x86_64_builder();
        let idx1 = builder.add_directory("/usr/include");
        let idx2 = builder.add_directory("/home/user/src");
        assert_eq!(idx1, 1);
        assert_eq!(idx2, 2);
    }

    #[test]
    fn test_serialized_file_table_terminated() {
        let mut builder = x86_64_builder();
        builder.add_file("test.c", 0);
        let data = builder.serialize();
        // The serialized data should contain the file name "test.c\0" somewhere.
        let name_bytes = b"test.c\0";
        let pos = data.windows(name_bytes.len()).position(|w| w == name_bytes);
        assert!(
            pos.is_some(),
            "File name 'test.c' not found in serialized data"
        );
    }

    #[test]
    fn test_serialized_directory_table_terminated() {
        let mut builder = x86_64_builder();
        builder.add_directory("/src");
        let data = builder.serialize();
        let dir_bytes = b"/src\0";
        let pos = data.windows(dir_bytes.len()).position(|w| w == dir_bytes);
        assert!(
            pos.is_some(),
            "Directory '/src' not found in serialized data"
        );
    }

    // -----------------------------------------------------------------------
    // Standard opcode tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_address_encoding() {
        let mut builder = x86_64_builder();
        builder.set_address(0x401000);
        let opcodes = &builder.opcodes;

        // Extended opcode: 0x00, ULEB128(1+8=9), 0x02, then 8 address bytes.
        assert_eq!(opcodes[0], 0x00); // Extended prefix
        assert_eq!(opcodes[1], 9); // ULEB128(9) = 0x09
        assert_eq!(opcodes[2], DW_LNE_SET_ADDRESS); // 0x02
                                                    // Address 0x401000 in little-endian.
        let addr = u64::from_le_bytes([
            opcodes[3],
            opcodes[4],
            opcodes[5],
            opcodes[6],
            opcodes[7],
            opcodes[8],
            opcodes[9],
            opcodes[10],
        ]);
        assert_eq!(addr, 0x401000);
    }

    #[test]
    fn test_set_address_i686_encoding() {
        let mut builder = i686_builder();
        builder.set_address(0x08048000);
        let opcodes = &builder.opcodes;

        assert_eq!(opcodes[0], 0x00); // Extended prefix
        assert_eq!(opcodes[1], 5); // ULEB128(1+4=5)
        assert_eq!(opcodes[2], DW_LNE_SET_ADDRESS);
        let addr = u32::from_le_bytes([opcodes[3], opcodes[4], opcodes[5], opcodes[6]]);
        assert_eq!(addr, 0x08048000);
    }

    #[test]
    fn test_advance_pc_encoding() {
        let mut builder = x86_64_builder();
        builder.advance_pc(10);
        // DW_LNS_advance_pc (0x02) + ULEB128(10) since min_instruction_length=1.
        assert_eq!(builder.opcodes[0], DW_LNS_ADVANCE_PC);
        assert_eq!(builder.opcodes[1], 10); // ULEB128(10) = single byte
        assert_eq!(builder.current_state.address, 10);
    }

    #[test]
    fn test_advance_pc_scaled_aarch64() {
        let mut builder = aarch64_builder();
        builder.advance_pc(16); // 16 bytes = 4 instructions on AArch64
        assert_eq!(builder.opcodes[0], DW_LNS_ADVANCE_PC);
        assert_eq!(builder.opcodes[1], 4); // 16 / 4 = 4 operations
        assert_eq!(builder.current_state.address, 16);
    }

    #[test]
    fn test_advance_line_positive() {
        let mut builder = x86_64_builder();
        builder.advance_line(5);
        assert_eq!(builder.opcodes[0], DW_LNS_ADVANCE_LINE);
        assert_eq!(builder.opcodes[1], 5); // SLEB128(5)
        assert_eq!(builder.current_state.line, 6); // Started at 1, +5 = 6
    }

    #[test]
    fn test_advance_line_negative() {
        let mut builder = x86_64_builder();
        // Advance line to 10 first.
        builder.current_state.line = 10;
        builder.advance_line(-3);
        // SLEB128(-3): 0x7d.
        assert_eq!(builder.opcodes[0], DW_LNS_ADVANCE_LINE);
        assert_eq!(builder.opcodes[1], 0x7d); // SLEB128(-3)
        assert_eq!(builder.current_state.line, 7);
    }

    #[test]
    fn test_set_file_encoding() {
        let mut builder = x86_64_builder();
        builder.set_file(3);
        assert_eq!(builder.opcodes[0], DW_LNS_SET_FILE);
        assert_eq!(builder.opcodes[1], 3);
        assert_eq!(builder.current_state.file, 3);
    }

    #[test]
    fn test_set_column_encoding() {
        let mut builder = x86_64_builder();
        builder.set_column(42);
        assert_eq!(builder.opcodes[0], DW_LNS_SET_COLUMN);
        assert_eq!(builder.opcodes[1], 42);
        assert_eq!(builder.current_state.column, 42);
    }

    #[test]
    fn test_copy_encoding() {
        let mut builder = x86_64_builder();
        builder.current_state.prologue_end = true;
        builder.current_state.basic_block = true;
        builder.copy();
        assert_eq!(builder.opcodes[0], DW_LNS_COPY);
        // Flags should be reset after copy.
        assert!(!builder.current_state.prologue_end);
        assert!(!builder.current_state.basic_block);
        assert!(!builder.current_state.epilogue_begin);
        assert_eq!(builder.current_state.discriminator, 0);
    }

    #[test]
    fn test_set_prologue_end_encoding() {
        let mut builder = x86_64_builder();
        builder.set_prologue_end();
        assert_eq!(builder.opcodes[0], DW_LNS_SET_PROLOGUE_END);
        assert!(builder.current_state.prologue_end);
    }

    #[test]
    fn test_end_sequence_encoding() {
        let mut builder = x86_64_builder();
        builder.current_state.address = 0x1000;
        builder.current_state.line = 42;
        builder.end_sequence();
        // Extended opcode: 0x00, ULEB128(1), 0x01 (DW_LNE_end_sequence).
        assert_eq!(builder.opcodes[0], 0x00);
        assert_eq!(builder.opcodes[1], 1);
        assert_eq!(builder.opcodes[2], DW_LNE_END_SEQUENCE);
        // State should be reset.
        assert_eq!(builder.current_state.address, 0);
        assert_eq!(builder.current_state.line, 1);
        assert_eq!(builder.current_state.file, 1);
    }

    // -----------------------------------------------------------------------
    // Special opcode tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_special_opcode_line_plus1_addr_plus1() {
        let builder = x86_64_builder();
        // line_delta = +1, addr_delta = 1 (operation_advance = 1)
        // special = (1 - (-5)) + (14 * 1) + 13 = 6 + 14 + 13 = 33
        let result = builder.try_special_opcode(1, 1);
        assert_eq!(result, Some(33));
    }

    #[test]
    fn test_special_opcode_line_base_addr_0() {
        let builder = x86_64_builder();
        // line_delta = -5 (= line_base), addr_delta = 0
        // special = (-5 - (-5)) + (14 * 0) + 13 = 0 + 0 + 13 = 13
        let result = builder.try_special_opcode(0, -5);
        assert_eq!(result, Some(13)); // Exactly the opcode_base
    }

    #[test]
    fn test_special_opcode_line_too_large() {
        let builder = x86_64_builder();
        // line_delta = +9 exceeds line_base + line_range - 1 = -5 + 14 - 1 = +8
        let result = builder.try_special_opcode(0, 9);
        assert_eq!(result, None);
    }

    #[test]
    fn test_special_opcode_addr_too_large() {
        let builder = x86_64_builder();
        // Large address advance: max opcode is 255, so max operation advance is
        // (255 - opcode_base - (line_delta - line_base)) / line_range
        // For line_delta=0: (255 - 13 - 5) / 14 = 237/14 = 16.9 → max = 16
        // addr_delta = 18 operations is too large.
        let result = builder.try_special_opcode(18, 0);
        assert_eq!(result, None);
    }

    #[test]
    fn test_special_opcode_aarch64_alignment() {
        let builder = aarch64_builder();
        // AArch64 min_instruction_length = 4.
        // addr_delta = 8 (2 instructions), line_delta = 1
        // operation_advance = 8 / 4 = 2
        // special = (1 - (-5)) + (14 * 2) + 13 = 6 + 28 + 13 = 47
        let result = builder.try_special_opcode(8, 1);
        assert_eq!(result, Some(47));
    }

    #[test]
    fn test_special_opcode_aarch64_unaligned_rejected() {
        let builder = aarch64_builder();
        // addr_delta = 5 is not divisible by 4 (min_instruction_length).
        let result = builder.try_special_opcode(5, 1);
        assert_eq!(result, None);
    }

    #[test]
    fn test_add_line_entry_uses_special_opcode() {
        let mut builder = x86_64_builder();
        builder.add_file("test.c", 0);
        builder.set_address(0x1000);
        let opcodes_before = builder.opcodes.len();
        // Line 1→2, addr +1: should use a special opcode (single byte).
        builder.add_line_entry(0x1001, 2, 0, 1);
        // Should have added exactly 1 byte (the special opcode).
        assert_eq!(builder.opcodes.len(), opcodes_before + 1);
        assert_eq!(builder.current_state.address, 0x1001);
        assert_eq!(builder.current_state.line, 2);
    }

    #[test]
    fn test_add_line_entry_fallback_to_standard() {
        let mut builder = x86_64_builder();
        builder.add_file("test.c", 0);
        builder.set_address(0x1000);
        // Large line delta (+100) that won't fit in a special opcode.
        let opcodes_before = builder.opcodes.len();
        builder.add_line_entry(0x1001, 101, 0, 1);
        // Should have emitted more than 1 byte (advance_pc + advance_line + copy).
        assert!(builder.opcodes.len() - opcodes_before > 1);
        assert_eq!(builder.current_state.address, 0x1001);
        assert_eq!(builder.current_state.line, 101);
    }

    #[test]
    fn test_add_line_entry_file_change() {
        let mut builder = x86_64_builder();
        builder.add_file("main.c", 0);
        builder.add_file("util.c", 0);
        builder.set_address(0x1000);
        builder.add_line_entry(0x1000, 1, 0, 1);
        // Change file to util.c (file 2).
        builder.add_line_entry(0x1001, 2, 0, 2);
        // Check that the file state was updated.
        assert_eq!(builder.current_state.file, 2);
    }

    // -----------------------------------------------------------------------
    // Full line program serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_serialize_empty_program() {
        let builder = x86_64_builder();
        let data = builder.serialize();
        // Should have at least the header: unit_length(4) + version(2) +
        // header_length(4) + config bytes + tables.
        assert!(data.len() > 10);
        // Verify the unit_length matches.
        let unit_length = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert_eq!(unit_length as usize, data.len() - 4);
    }

    #[test]
    fn test_serialize_starts_with_set_address_ends_with_end_sequence() {
        let mut builder = x86_64_builder();
        builder.add_file("test.c", 0);
        builder.set_address(0x1000);
        builder.add_line_entry(0x1000, 1, 1, 1);
        builder.end_sequence();
        let data = builder.serialize();

        // The last 3 bytes should be 0x00, 0x01, 0x01 (end_sequence).
        let last_three = &data[data.len() - 3..];
        assert_eq!(last_three[0], 0x00); // Extended prefix
        assert_eq!(last_three[1], 0x01); // Length = 1
        assert_eq!(last_three[2], DW_LNE_END_SEQUENCE);
    }

    #[test]
    fn test_serialize_reasonable_size() {
        let mut builder = x86_64_builder();
        builder.add_file("test.c", 0);
        builder.set_address(0x1000);
        for i in 0..100u64 {
            builder.add_line_entry(0x1000 + i * 4, (i as u32) + 1, 1, 1);
        }
        builder.end_sequence();
        let data = builder.serialize();
        // 100 line entries with small deltas should mostly use special opcodes.
        assert!(
            data.len() < 500,
            "Serialized size {} is unexpectedly large",
            data.len()
        );
        assert!(
            data.len() > 50,
            "Serialized size {} is unexpectedly small",
            data.len()
        );
    }

    #[test]
    fn test_roundtrip_unit_length_header_length() {
        let mut builder = x86_64_builder();
        builder.add_file("main.c", 0);
        builder.add_directory("/home/user");
        builder.set_address(0x400000);
        builder.add_line_entry(0x400000, 1, 1, 1);
        builder.add_line_entry(0x400010, 5, 1, 1);
        builder.end_sequence();
        let data = builder.serialize();

        // Verify unit_length.
        let unit_length = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        assert_eq!(unit_length, data.len() - 4);

        // Verify header_length: at offset 6 (after unit_length + version).
        let header_length = u32::from_le_bytes([data[6], data[7], data[8], data[9]]) as usize;
        // The opcode stream starts at offset 10 + header_length.
        let opcode_start = 10 + header_length;
        assert!(opcode_start < data.len());
    }

    // -----------------------------------------------------------------------
    // Architecture-specific tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_arch_x86_64() {
        let builder = x86_64_builder();
        assert_eq!(builder.header.minimum_instruction_length, 1);
        assert_eq!(builder.header.address_size, 8);
        assert_eq!(builder.header.version, 4);
    }

    #[test]
    fn test_arch_i686() {
        let builder = i686_builder();
        assert_eq!(builder.header.minimum_instruction_length, 1);
        assert_eq!(builder.header.address_size, 4);
        assert_eq!(builder.header.version, 4);
    }

    #[test]
    fn test_arch_aarch64() {
        let builder = aarch64_builder();
        assert_eq!(builder.header.minimum_instruction_length, 4);
        assert_eq!(builder.header.address_size, 8);
        assert_eq!(builder.header.version, 4);
    }

    #[test]
    fn test_arch_riscv64() {
        let builder = riscv64_builder();
        assert_eq!(builder.header.minimum_instruction_length, 2);
        assert_eq!(builder.header.address_size, 8);
        assert_eq!(builder.header.version, 4);
    }

    // -----------------------------------------------------------------------
    // LineProgramEmitter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_emitter_new() {
        let emitter = LineProgramEmitter::new(8, 1);
        assert_eq!(emitter.address_size, 8);
        assert_eq!(emitter.min_instruction_length, 1);
    }

    #[test]
    fn test_emitter_emit_empty() {
        let emitter = LineProgramEmitter::new(8, 1);
        let source_map = SourceMap::new();
        let data = emitter.emit(&[], &source_map);
        // Should produce a valid header with no opcodes.
        assert!(data.len() > 10);
        let version = u16::from_le_bytes([data[4], data[5]]);
        assert_eq!(version, 4);
    }

    // -----------------------------------------------------------------------
    // LineMappingEntry tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_line_mapping_entry_default_flags() {
        let entry = LineMappingEntry {
            address: 0x1000,
            file_id: 1,
            line: 10,
            column: 5,
            is_stmt: true,
            is_prologue_end: false,
            is_end_sequence: false,
        };
        assert_eq!(entry.address, 0x1000);
        assert_eq!(entry.file_id, 1);
        assert_eq!(entry.line, 10);
        assert_eq!(entry.column, 5);
        assert!(entry.is_stmt);
        assert!(!entry.is_prologue_end);
        assert!(!entry.is_end_sequence);
    }

    // -----------------------------------------------------------------------
    // from_source_mappings integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_from_source_mappings_basic() {
        use std::path::PathBuf;

        let mut source_map = SourceMap::new();
        let _file_id = source_map.add_file(
            PathBuf::from("src/main.c"),
            "int main() {\n    return 0;\n}\n".to_string(),
        );

        let mappings = vec![
            LineMappingEntry {
                address: 0x1000,
                file_id: 0,
                line: 1,
                column: 1,
                is_stmt: true,
                is_prologue_end: false,
                is_end_sequence: false,
            },
            LineMappingEntry {
                address: 0x1008,
                file_id: 0,
                line: 2,
                column: 5,
                is_stmt: true,
                is_prologue_end: false,
                is_end_sequence: false,
            },
            LineMappingEntry {
                address: 0x1010,
                file_id: 0,
                line: 3,
                column: 1,
                is_stmt: true,
                is_prologue_end: false,
                is_end_sequence: true,
            },
        ];

        let builder = LineProgramBuilder::from_source_mappings(&mappings, &source_map, 8, 1);
        let data = builder.serialize();

        // Should be a valid DWARF line program.
        assert!(data.len() > 20);
        let version = u16::from_le_bytes([data[4], data[5]]);
        assert_eq!(version, 4);

        // Should have at least one file registered.
        assert!(!builder.header.file_entries.is_empty());
    }

    #[test]
    fn test_from_source_mappings_empty() {
        let source_map = SourceMap::new();
        let builder = LineProgramBuilder::from_source_mappings(&[], &source_map, 8, 1);
        let data = builder.serialize();
        // Should produce valid header-only output.
        assert!(data.len() > 10);
    }

    // -----------------------------------------------------------------------
    // LineNumberState tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_initial_state() {
        let state = LineNumberState::initial(true);
        assert_eq!(state.address, 0);
        assert_eq!(state.op_index, 0);
        assert_eq!(state.file, 1);
        assert_eq!(state.line, 1);
        assert_eq!(state.column, 0);
        assert!(state.is_stmt);
        assert!(!state.basic_block);
        assert!(!state.end_sequence);
        assert!(!state.prologue_end);
        assert!(!state.epilogue_begin);
        assert_eq!(state.isa, 0);
        assert_eq!(state.discriminator, 0);
    }

    #[test]
    fn test_initial_state_is_stmt_false() {
        let state = LineNumberState::initial(false);
        assert!(!state.is_stmt);
    }

    #[test]
    fn test_reset_row_flags() {
        let mut state = LineNumberState::initial(true);
        state.basic_block = true;
        state.prologue_end = true;
        state.epilogue_begin = true;
        state.discriminator = 42;
        state.reset_row_flags();
        assert!(!state.basic_block);
        assert!(!state.prologue_end);
        assert!(!state.epilogue_begin);
        assert_eq!(state.discriminator, 0);
        // is_stmt should not be affected.
        assert!(state.is_stmt);
    }
}
