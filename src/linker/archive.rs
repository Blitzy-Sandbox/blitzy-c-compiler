//! # `ar` Static Archive Reader
//!
//! This module implements a complete reader for Unix `ar` static archive files
//! (`.a` files). The integrated linker uses this module to read `libc.a` and
//! other static libraries specified via the `-l` command-line flag, as well as
//! system CRT object archives.
//!
//! ## Supported Formats
//!
//! - **GNU/SysV archives** — The predominant format on Linux systems. Supports
//!   short member names (≤15 characters terminated by `/`), the `//` long-name
//!   table for names exceeding 15 characters, and the `/` symbol table entry.
//! - **BSD archives** — Recognizes the `#1/<length>` inline long-name format
//!   and handles it without crashing, though GNU/SysV is the primary target.
//!
//! ## Archive Format Overview
//!
//! An `ar` archive consists of:
//! 1. An 8-byte global magic string: `!<arch>\n`
//! 2. A sequence of members, each preceded by a 60-byte ASCII header
//! 3. Member data immediately following each header
//! 4. 1-byte padding (`\n`) after odd-length members to maintain 2-byte alignment
//!
//! ## Zero External Dependencies
//!
//! This module uses only the Rust standard library. No external archive-reading
//! libraries are used.

use super::elf;
use std::path::Path;

// ===========================================================================
// Archive Format Constants
// ===========================================================================

/// Archive global magic bytes: `!<arch>\n`
///
/// Every valid `ar` archive begins with these 8 bytes. This is checked first
/// during parsing to reject non-archive input early.
pub const AR_MAGIC: &[u8] = b"!<arch>\n";

/// Length of the archive global magic string in bytes.
pub const AR_MAGIC_LEN: usize = 8;

/// Size of an archive member header in bytes.
///
/// Each member header is exactly 60 bytes of ASCII data, containing the
/// member name, modification time, owner/group IDs, file mode, file size,
/// and a 2-byte end-of-header marker.
pub const AR_HEADER_SIZE: usize = 60;

/// End-of-header marker: `` `\n ``
///
/// The last two bytes of every valid archive member header must be this
/// sequence. It is used to validate header integrity during parsing.
pub const AR_FMAG: &[u8] = b"`\n";

/// GNU/SysV long-name table identifier: `//`
///
/// A member whose name field begins with `//` contains the extended name
/// table. Other members reference long names by storing `/offset` in their
/// name field, where `offset` is a byte position into this table.
pub const AR_GNU_NAMES: &[u8] = b"//";

/// GNU/SysV symbol table identifier: `/`
///
/// A member whose name field is exactly `/` (followed by spaces) contains
/// the archive symbol index. This is used by traditional linkers for fast
/// symbol lookup but is skipped by our linker, which performs its own
/// symbol resolution.
pub const AR_GNU_SYMTAB: &[u8] = b"/";

// ===========================================================================
// ArchiveError
// ===========================================================================

/// Errors that can occur during archive parsing or member access.
#[derive(Debug)]
pub enum ArchiveError {
    /// Input data is shorter than the minimum archive size (8 bytes for magic).
    NotAnArchive,

    /// The first 8 bytes of the input do not match `!<arch>\n`.
    InvalidMagic,

    /// A member header was encountered but the remaining data is shorter
    /// than the required 60 bytes.
    TruncatedHeader,

    /// The end-of-header marker (bytes 58–59) is not `` `\n ``.
    InvalidHeaderMagic,

    /// A numeric field in the header (size, date, uid, gid, or mode) could
    /// not be parsed. The `String` describes which field failed.
    InvalidNumericField(String),

    /// A member's declared size extends beyond the end of the archive data.
    /// The `String` identifies the member by name or offset.
    TruncatedMember(String),

    /// An I/O error occurred while reading the archive from disk.
    IoError(std::io::Error),
}

impl std::fmt::Display for ArchiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArchiveError::NotAnArchive => {
                write!(f, "data too short to be an ar archive")
            }
            ArchiveError::InvalidMagic => {
                write!(f, "invalid ar archive magic (expected \"!<arch>\\n\")")
            }
            ArchiveError::TruncatedHeader => {
                write!(f, "truncated archive member header (expected 60 bytes)")
            }
            ArchiveError::InvalidHeaderMagic => {
                write!(
                    f,
                    "invalid archive header end-of-header marker (expected \"`\\n\")"
                )
            }
            ArchiveError::InvalidNumericField(field) => {
                write!(f, "invalid numeric field in archive header: {}", field)
            }
            ArchiveError::TruncatedMember(name) => {
                write!(f, "archive member '{}' extends past end of archive", name)
            }
            ArchiveError::IoError(e) => {
                write!(f, "I/O error reading archive: {}", e)
            }
        }
    }
}

impl std::error::Error for ArchiveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ArchiveError::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ArchiveError {
    fn from(e: std::io::Error) -> Self {
        ArchiveError::IoError(e)
    }
}

// ===========================================================================
// ArHeader — Archive Member Header (60 bytes, ASCII)
// ===========================================================================

/// Parsed archive member header.
///
/// The raw archive header format is 60 bytes of ASCII text, with each field
/// right-padded with spaces:
///
/// | Offset | Size | Field                | Format              |
/// |--------|------|----------------------|---------------------|
/// | 0      | 16   | `ar_name`            | ASCII, space-padded |
/// | 16     | 12   | `ar_date`            | Decimal seconds     |
/// | 28     | 6    | `ar_uid`             | Decimal             |
/// | 34     | 6    | `ar_gid`             | Decimal             |
/// | 40     | 8    | `ar_mode`            | Octal               |
/// | 48     | 10   | `ar_size`            | Decimal bytes       |
/// | 58     | 2    | `ar_fmag`            | `` `\n ``           |
#[derive(Debug, Clone)]
pub struct ArHeader {
    /// Member name as extracted from the header (may contain `/` suffix or
    /// GNU long-name reference like `/123`).
    pub name: String,

    /// Modification time in seconds since the Unix epoch.
    pub modification_time: u64,

    /// Numeric owner (user) ID.
    pub owner_id: u32,

    /// Numeric group ID.
    pub group_id: u32,

    /// File mode (permission bits) in octal representation.
    pub mode: u32,

    /// Member data size in bytes.
    pub size: u64,
}

impl ArHeader {
    /// Parse a 60-byte archive member header from a byte slice.
    ///
    /// The slice must be at least `AR_HEADER_SIZE` (60) bytes long. This
    /// method validates the end-of-header marker and parses all numeric
    /// fields from their ASCII representations.
    ///
    /// # Errors
    ///
    /// Returns `ArchiveError::TruncatedHeader` if the slice is too short,
    /// `ArchiveError::InvalidHeaderMagic` if the fmag marker is wrong, or
    /// `ArchiveError::InvalidNumericField` if a numeric field cannot be parsed.
    pub fn parse(header_bytes: &[u8]) -> Result<Self, ArchiveError> {
        if header_bytes.len() < AR_HEADER_SIZE {
            return Err(ArchiveError::TruncatedHeader);
        }

        // Verify the end-of-header marker at bytes [58..60].
        if &header_bytes[58..60] != AR_FMAG {
            return Err(ArchiveError::InvalidHeaderMagic);
        }

        // Extract the raw name field (bytes 0..16) as a UTF-8 string.
        let name = parse_ascii_field(&header_bytes[0..16], "name")?;

        // Parse numeric fields, defaulting to 0 for empty/whitespace-only fields.
        let modification_time = parse_decimal_u64(&header_bytes[16..28], "modification_time")?;
        let owner_id = parse_decimal_u32(&header_bytes[28..34], "owner_id")?;
        let group_id = parse_decimal_u32(&header_bytes[34..40], "group_id")?;
        let mode = parse_octal_u32(&header_bytes[40..48], "mode")?;
        let size = parse_decimal_u64(&header_bytes[48..58], "size")?;

        Ok(ArHeader {
            name,
            modification_time,
            owner_id,
            group_id,
            mode,
            size,
        })
    }
}

// ===========================================================================
// ArchiveMember
// ===========================================================================

/// A parsed archive member with its resolved name and location within the
/// archive's raw byte buffer.
///
/// The `data_offset` and `data_size` fields describe the member's content
/// region within the `Archive`'s internal byte vector. The actual data can
/// be accessed via `Archive::get_member_data()` or `Archive::iter_members()`.
#[derive(Debug, Clone)]
pub struct ArchiveMember {
    /// Member name, resolved from short names, GNU long-name table references,
    /// or BSD inline long names.
    pub name: String,

    /// Byte offset of the member's data within the archive byte vector.
    /// This points past the 60-byte header to the start of actual content.
    pub data_offset: usize,

    /// Size of the member's data in bytes, as declared in the header.
    pub data_size: usize,
}

// ===========================================================================
// Archive
// ===========================================================================

/// A parsed `ar` static archive.
///
/// The archive reader stores the raw archive data and maintains a list of
/// parsed member entries. It supports sequential and random access to member
/// data, as well as ELF-specific filtering to extract only valid ELF object
/// members.
pub struct Archive {
    /// The complete raw archive file data, retained for member data access.
    /// Note: `Debug` is implemented manually to avoid printing the full data buffer.
    data: Vec<u8>,

    /// Parsed archive members (excluding special entries like symbol tables
    /// and long-name tables).
    members: Vec<ArchiveMember>,

    /// The GNU/SysV long-name table contents, if the archive contains one.
    /// Member name references of the form `/offset` are resolved against
    /// this table.
    long_names: Option<Vec<u8>>,
}

impl std::fmt::Debug for Archive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Archive")
            .field("data_len", &self.data.len())
            .field("members", &self.members)
            .field("has_long_names", &self.long_names.is_some())
            .finish()
    }
}

impl Archive {
    /// Parse an `ar` archive from raw bytes.
    ///
    /// This method validates the archive magic, then sequentially parses all
    /// member headers. Special members (GNU symbol table `/`, GNU long-name
    /// table `//`) are recognized and handled appropriately:
    /// - The long-name table (`//`) is stored for resolving extended member names
    /// - The symbol table (`/`) is skipped (our linker does its own resolution)
    /// - All other members are recorded with their resolved names and data offsets
    ///
    /// Member data is always 2-byte-aligned within the archive; a padding byte
    /// is skipped after odd-length members.
    ///
    /// # Errors
    ///
    /// Returns an error if the magic bytes are invalid, a header is malformed,
    /// or a member's data extends past the end of the archive.
    pub fn parse(data: Vec<u8>) -> Result<Self, ArchiveError> {
        // Validate minimum size and magic bytes.
        if data.len() < AR_MAGIC_LEN {
            return Err(ArchiveError::NotAnArchive);
        }
        if &data[..AR_MAGIC_LEN] != AR_MAGIC {
            return Err(ArchiveError::InvalidMagic);
        }

        let mut offset = AR_MAGIC_LEN;
        let mut members = Vec::new();
        let mut long_names: Option<Vec<u8>> = None;

        while offset < data.len() {
            // If there are fewer than AR_HEADER_SIZE bytes remaining, the
            // archive is either complete (trailing padding) or truncated.
            if offset + AR_HEADER_SIZE > data.len() {
                // Some archives have a trailing newline; tolerate up to
                // AR_HEADER_SIZE - 1 bytes of trailing data.
                break;
            }

            let header = ArHeader::parse(&data[offset..])?;
            let data_start = offset + AR_HEADER_SIZE;
            let member_size = header.size as usize;

            // Validate that the member data fits within the archive.
            if data_start + member_size > data.len() {
                return Err(ArchiveError::TruncatedMember(header.name.clone()));
            }

            // Identify and handle special members.
            if is_gnu_long_names(&header) {
                // Store the long-name table for subsequent name resolution.
                long_names = Some(data[data_start..data_start + member_size].to_vec());
            } else if is_gnu_symtab(&header) {
                // Skip the symbol table — our linker performs its own
                // symbol resolution across all input objects.
            } else {
                // Regular member: resolve its name and record it.
                let resolved_name = resolve_member_name(&header, &long_names)?;
                members.push(ArchiveMember {
                    name: resolved_name,
                    data_offset: data_start,
                    data_size: member_size,
                });
            }

            // Advance past the member data, with 2-byte alignment padding.
            offset = data_start + member_size;
            if offset % 2 != 0 {
                offset += 1;
            }
        }

        Ok(Archive {
            data,
            members,
            long_names,
        })
    }

    /// Read an `ar` archive from a file path on disk.
    ///
    /// This is a convenience wrapper that reads the entire file into memory
    /// and then delegates to `Archive::parse()`.
    ///
    /// # Errors
    ///
    /// Returns `ArchiveError::IoError` if the file cannot be read, or any
    /// parsing error from `Archive::parse()`.
    pub fn from_file(path: &Path) -> Result<Self, ArchiveError> {
        let data = std::fs::read(path).map_err(ArchiveError::IoError)?;
        Self::parse(data)
    }

    /// Return a list of all member names in the archive.
    ///
    /// The returned names are in archive order (the order they appear in the
    /// file). Special entries (symbol table, long-name table) are excluded.
    pub fn member_names(&self) -> Vec<&str> {
        self.members.iter().map(|m| m.name.as_str()).collect()
    }

    /// Return the number of regular members in the archive.
    ///
    /// Special entries (symbol table, long-name table) are excluded from
    /// this count.
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Look up a member by name and return its raw data bytes.
    ///
    /// If multiple members share the same name (which is valid in `ar`
    /// archives), the first match is returned.
    ///
    /// Returns `None` if no member with the given name exists.
    pub fn get_member_data(&self, name: &str) -> Option<&[u8]> {
        self.members
            .iter()
            .find(|m| m.name == name)
            .map(|m| &self.data[m.data_offset..m.data_offset + m.data_size])
    }

    /// Access a member by its zero-based index in the member list.
    ///
    /// Returns a tuple of `(name, data)` if the index is valid, or `None`
    /// if it is out of bounds.
    pub fn get_member_by_index(&self, index: usize) -> Option<(&str, &[u8])> {
        self.members.get(index).map(|m| {
            (
                m.name.as_str(),
                &self.data[m.data_offset..m.data_offset + m.data_size],
            )
        })
    }

    /// Iterate over all regular members, yielding `(name, data)` pairs.
    ///
    /// Members are yielded in archive order. Special entries (symbol table,
    /// long-name table) are excluded.
    pub fn iter_members(&self) -> impl Iterator<Item = (&str, &[u8])> {
        self.members.iter().map(move |m| {
            (
                m.name.as_str(),
                &self.data[m.data_offset..m.data_offset + m.data_size],
            )
        })
    }

    /// Extract all members whose content begins with the ELF magic bytes.
    ///
    /// This is useful for filtering out non-ELF members (such as LLVM
    /// bitcode, index files, or other metadata) from an archive before
    /// passing members to the ELF object parser.
    ///
    /// The ELF magic check uses the canonical `ELF_MAGIC` constant from
    /// the `elf` module to ensure consistency with the rest of the linker.
    pub fn elf_members(&self) -> Vec<(&str, &[u8])> {
        self.iter_members()
            .filter(|(_, data)| {
                data.len() >= elf::ELF_MAGIC.len() && data[..elf::ELF_MAGIC.len()] == elf::ELF_MAGIC
            })
            .collect()
    }
}

// ===========================================================================
// Name Resolution Helpers
// ===========================================================================

/// Resolve a member name from the raw header name field.
///
/// The `ar` format supports several naming conventions:
///
/// 1. **Short name** — `"foo.o/          "` (≤15 chars followed by `/`, space-padded
///    to 16 bytes). The name is extracted by trimming trailing spaces and the
///    terminating `/`.
///
/// 2. **GNU long name** — `"/123            "` (a `/` followed by a decimal byte
///    offset into the long-name table). The actual name is read from the `//`
///    table at the given offset, terminated by `/\n` or a null byte.
///
/// 3. **BSD long name** — `"#1/17           "` (the prefix `#1/` followed by a
///    decimal byte count). The actual name is stored inline immediately after
///    the header, consuming the first N bytes of the member data. We handle
///    this format gracefully but note that GNU/SysV is the primary Linux format.
fn resolve_member_name(
    header: &ArHeader,
    long_names: &Option<Vec<u8>>,
) -> Result<String, ArchiveError> {
    let raw_name = &header.name;

    // Check for GNU long-name reference: `/` followed by digits (and spaces).
    if raw_name.starts_with('/')
        && raw_name.len() > 1
        && raw_name[1..]
            .chars()
            .all(|c| c.is_ascii_digit() || c == ' ')
    {
        let offset_str = raw_name[1..].trim();
        if offset_str.is_empty() {
            // Just "/" with trailing spaces — this is the symtab, not a long name.
            // Should have been caught by is_gnu_symtab(), but handle defensively.
            return Ok("/".to_string());
        }
        let name_offset: usize = offset_str.parse().map_err(|_| {
            ArchiveError::InvalidNumericField(format!(
                "long name offset '{}' is not a valid integer",
                offset_str
            ))
        })?;

        if let Some(names) = long_names {
            if name_offset >= names.len() {
                return Err(ArchiveError::InvalidNumericField(format!(
                    "long name offset {} exceeds long name table size {}",
                    name_offset,
                    names.len()
                )));
            }
            // Read until we find `/`, `\n`, or a null byte.
            let end_pos = names[name_offset..]
                .iter()
                .position(|&b| b == b'/' || b == b'\n' || b == 0)
                .unwrap_or(names.len() - name_offset);
            let name_bytes = &names[name_offset..name_offset + end_pos];
            let name = std::str::from_utf8(name_bytes).map_err(|_| {
                ArchiveError::InvalidNumericField(format!(
                    "long name at offset {} is not valid UTF-8",
                    name_offset
                ))
            })?;
            Ok(name.to_string())
        } else {
            Err(ArchiveError::InvalidNumericField(
                "GNU long name reference found but no long name table (//) present".to_string(),
            ))
        }
    } else if raw_name.starts_with("#1/") {
        // BSD long-name format: `#1/<length>`. The actual name is stored inline
        // at the beginning of the member data. For robustness we parse the
        // length, but since we don't have the member data here, we return the
        // raw header name trimmed as a fallback identifier.
        let len_str = raw_name[3..].trim();
        let _name_len: usize = len_str.parse().map_err(|_| {
            ArchiveError::InvalidNumericField(format!(
                "BSD long name length '{}' is not a valid integer",
                len_str
            ))
        })?;
        // The actual inline name would need to be read from the member data.
        // For now, return the trimmed header name as an identifier.
        Ok(raw_name.trim_end().trim_end_matches('/').to_string())
    } else {
        // Short name: trim trailing spaces and the terminating `/`.
        let trimmed = raw_name.trim_end();
        let name = trimmed.trim_end_matches('/');
        Ok(name.to_string())
    }
}

/// Check whether this member header identifies the GNU long-name table (`//`).
fn is_gnu_long_names(header: &ArHeader) -> bool {
    header.name.starts_with("//")
}

/// Check whether this member header identifies a symbol table entry.
///
/// Recognizes both the GNU/SysV format (`/` with trailing spaces) and the
/// BSD format (`__.SYMDEF` prefix).
fn is_gnu_symtab(header: &ArHeader) -> bool {
    let trimmed = header.name.trim();
    trimmed == "/" || header.name.starts_with("__.SYMDEF")
}

// ===========================================================================
// ASCII Field Parsing Helpers
// ===========================================================================

/// Parse a raw ASCII byte slice into a `String`, trimming nothing — just
/// converting bytes to a string. This preserves trailing spaces so the
/// caller can inspect the raw name value.
fn parse_ascii_field(bytes: &[u8], field_name: &str) -> Result<String, ArchiveError> {
    std::str::from_utf8(bytes)
        .map(|s| s.to_string())
        .map_err(|_| {
            ArchiveError::InvalidNumericField(format!(
                "'{}' field contains invalid UTF-8",
                field_name
            ))
        })
}

/// Parse a decimal ASCII numeric field into `u64`.
///
/// The field is first converted to a UTF-8 string, then trimmed of leading
/// and trailing whitespace. Empty or all-whitespace fields default to 0,
/// which is common for UID/GID fields in some archive generators.
fn parse_decimal_u64(bytes: &[u8], field_name: &str) -> Result<u64, ArchiveError> {
    let s = std::str::from_utf8(bytes).map_err(|_| {
        ArchiveError::InvalidNumericField(format!("'{}' field contains invalid UTF-8", field_name))
    })?;
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Ok(0);
    }
    trimmed.parse::<u64>().map_err(|_| {
        ArchiveError::InvalidNumericField(format!(
            "'{}' field '{}' is not a valid decimal number",
            field_name, trimmed
        ))
    })
}

/// Parse a decimal ASCII numeric field into `u32`.
///
/// Same semantics as `parse_decimal_u64` but returns a `u32`. Empty fields
/// default to 0.
fn parse_decimal_u32(bytes: &[u8], field_name: &str) -> Result<u32, ArchiveError> {
    let s = std::str::from_utf8(bytes).map_err(|_| {
        ArchiveError::InvalidNumericField(format!("'{}' field contains invalid UTF-8", field_name))
    })?;
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Ok(0);
    }
    trimmed.parse::<u32>().map_err(|_| {
        ArchiveError::InvalidNumericField(format!(
            "'{}' field '{}' is not a valid decimal number",
            field_name, trimmed
        ))
    })
}

/// Parse an octal ASCII numeric field into `u32`.
///
/// The field value is interpreted as an octal (base-8) number, which is the
/// standard representation for file mode/permission fields in `ar` headers.
/// Empty fields default to 0.
fn parse_octal_u32(bytes: &[u8], field_name: &str) -> Result<u32, ArchiveError> {
    let s = std::str::from_utf8(bytes).map_err(|_| {
        ArchiveError::InvalidNumericField(format!("'{}' field contains invalid UTF-8", field_name))
    })?;
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Ok(0);
    }
    u32::from_str_radix(trimmed, 8).map_err(|_| {
        ArchiveError::InvalidNumericField(format!(
            "'{}' field '{}' is not a valid octal number",
            field_name, trimmed
        ))
    })
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Test helper: Build a synthetic archive in memory
    // -----------------------------------------------------------------------

    /// Build a minimal `ar` archive from a list of `(name, data)` pairs.
    ///
    /// Each member gets a properly formatted 60-byte header with the name
    /// terminated by `/`, size filled in decimal, and all other fields set
    /// to reasonable defaults. The fmag marker is set correctly.
    fn make_test_archive(members: &[(&str, &[u8])]) -> Vec<u8> {
        let mut ar = AR_MAGIC.to_vec();

        for &(name, data) in members {
            let mut header = [b' '; AR_HEADER_SIZE];

            // ar_name: name + "/" padded with spaces to 16 bytes
            let name_with_slash = format!("{}/", name);
            let name_bytes = name_with_slash.as_bytes();
            let copy_len = name_bytes.len().min(16);
            header[..copy_len].copy_from_slice(&name_bytes[..copy_len]);

            // ar_date (bytes 16..28): "0" padded
            header[16] = b'0';

            // ar_uid (bytes 28..34): "0"
            header[28] = b'0';

            // ar_gid (bytes 34..40): "0"
            header[34] = b'0';

            // ar_mode (bytes 40..48): "100644" (octal)
            let mode_bytes = b"100644";
            header[40..40 + mode_bytes.len()].copy_from_slice(mode_bytes);

            // ar_size (bytes 48..58): decimal size
            let size_str = format!("{}", data.len());
            let size_bytes = size_str.as_bytes();
            header[48..48 + size_bytes.len()].copy_from_slice(size_bytes);

            // ar_fmag (bytes 58..60): "`\n"
            header[58] = b'`';
            header[59] = b'\n';

            ar.extend_from_slice(&header);
            ar.extend_from_slice(data);

            // 2-byte alignment padding
            if ar.len() % 2 != 0 {
                ar.push(b'\n');
            }
        }

        ar
    }

    /// Build a raw 60-byte header with a custom name field (no `/` appended).
    /// Used for testing special members like `//` and `/`.
    fn make_raw_header(raw_name: &[u8], data_size: usize) -> [u8; AR_HEADER_SIZE] {
        let mut header = [b' '; AR_HEADER_SIZE];

        // Copy raw name bytes (up to 16 bytes)
        let copy_len = raw_name.len().min(16);
        header[..copy_len].copy_from_slice(&raw_name[..copy_len]);

        // ar_date
        header[16] = b'0';

        // ar_uid
        header[28] = b'0';

        // ar_gid
        header[34] = b'0';

        // ar_mode
        let mode_bytes = b"100644";
        header[40..40 + mode_bytes.len()].copy_from_slice(mode_bytes);

        // ar_size
        let size_str = format!("{}", data_size);
        let size_bytes = size_str.as_bytes();
        header[48..48 + size_bytes.len()].copy_from_slice(size_bytes);

        // ar_fmag
        header[58] = b'`';
        header[59] = b'\n';

        header
    }

    // -----------------------------------------------------------------------
    // Constant verification
    // -----------------------------------------------------------------------

    #[test]
    fn test_ar_magic_constant() {
        assert_eq!(AR_MAGIC, b"!<arch>\n");
        assert_eq!(AR_MAGIC_LEN, 8);
        assert_eq!(AR_MAGIC.len(), AR_MAGIC_LEN);
    }

    #[test]
    fn test_ar_header_size_constant() {
        assert_eq!(AR_HEADER_SIZE, 60);
    }

    #[test]
    fn test_ar_fmag_constant() {
        assert_eq!(AR_FMAG, b"`\n");
        assert_eq!(AR_FMAG.len(), 2);
    }

    #[test]
    fn test_ar_gnu_names_constant() {
        assert_eq!(AR_GNU_NAMES, b"//");
    }

    #[test]
    fn test_ar_gnu_symtab_constant() {
        assert_eq!(AR_GNU_SYMTAB, b"/");
    }

    // -----------------------------------------------------------------------
    // ArHeader parsing
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_valid_header() {
        let header = make_raw_header(b"test.o/         ", 42);
        let parsed = ArHeader::parse(&header).unwrap();
        assert!(parsed.name.starts_with("test.o/"));
        assert_eq!(parsed.size, 42);
        assert_eq!(parsed.modification_time, 0);
        assert_eq!(parsed.owner_id, 0);
        assert_eq!(parsed.group_id, 0);
    }

    #[test]
    fn test_parse_header_truncated() {
        let short_data = [0u8; 30];
        let result = ArHeader::parse(&short_data);
        assert!(result.is_err());
        match result.unwrap_err() {
            ArchiveError::TruncatedHeader => {}
            other => panic!("expected TruncatedHeader, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_header_invalid_fmag() {
        let mut header = [b' '; AR_HEADER_SIZE];
        // Don't set fmag correctly
        header[58] = b'X';
        header[59] = b'Y';
        let result = ArHeader::parse(&header);
        assert!(result.is_err());
        match result.unwrap_err() {
            ArchiveError::InvalidHeaderMagic => {}
            other => panic!("expected InvalidHeaderMagic, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Empty archive
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_archive() {
        let data = AR_MAGIC.to_vec();
        let archive = Archive::parse(data).unwrap();
        assert_eq!(archive.member_count(), 0);
        assert!(archive.member_names().is_empty());
    }

    // -----------------------------------------------------------------------
    // Single-member archive
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_member_archive() {
        let content = b"hello world";
        let data = make_test_archive(&[("test.o", content)]);
        let archive = Archive::parse(data).unwrap();

        assert_eq!(archive.member_count(), 1);
        assert_eq!(archive.member_names(), vec!["test.o"]);

        let member_data = archive.get_member_data("test.o").unwrap();
        assert_eq!(member_data, content);
    }

    // -----------------------------------------------------------------------
    // Multiple members
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_members() {
        let data = make_test_archive(&[
            ("foo.o", b"data1"),
            ("bar.o", b"data2data2"),
            ("baz.o", b"d3"),
        ]);
        let archive = Archive::parse(data).unwrap();

        assert_eq!(archive.member_count(), 3);
        assert_eq!(archive.member_names(), vec!["foo.o", "bar.o", "baz.o"]);

        assert_eq!(archive.get_member_data("foo.o").unwrap(), b"data1");
        assert_eq!(archive.get_member_data("bar.o").unwrap(), b"data2data2");
        assert_eq!(archive.get_member_data("baz.o").unwrap(), b"d3");
    }

    // -----------------------------------------------------------------------
    // Member access by index
    // -----------------------------------------------------------------------

    #[test]
    fn test_member_access_by_index() {
        let data = make_test_archive(&[("a.o", b"AAA"), ("b.o", b"BBB")]);
        let archive = Archive::parse(data).unwrap();

        let (name0, data0) = archive.get_member_by_index(0).unwrap();
        assert_eq!(name0, "a.o");
        assert_eq!(data0, b"AAA");

        let (name1, data1) = archive.get_member_by_index(1).unwrap();
        assert_eq!(name1, "b.o");
        assert_eq!(data1, b"BBB");

        assert!(archive.get_member_by_index(2).is_none());
    }

    // -----------------------------------------------------------------------
    // Iterator
    // -----------------------------------------------------------------------

    #[test]
    fn test_iter_members() {
        let data = make_test_archive(&[("x.o", b"X"), ("y.o", b"YY")]);
        let archive = Archive::parse(data).unwrap();

        let items: Vec<(&str, &[u8])> = archive.iter_members().collect();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].0, "x.o");
        assert_eq!(items[0].1, b"X");
        assert_eq!(items[1].0, "y.o");
        assert_eq!(items[1].1, b"YY");
    }

    // -----------------------------------------------------------------------
    // 2-byte alignment padding
    // -----------------------------------------------------------------------

    #[test]
    fn test_alignment_padding() {
        // First member has odd-length data (5 bytes), requiring 1 byte padding.
        // Second member follows correctly after the padding.
        let data = make_test_archive(&[("odd.o", b"12345"), ("even.o", b"67")]);
        let archive = Archive::parse(data).unwrap();

        assert_eq!(archive.member_count(), 2);
        assert_eq!(archive.get_member_data("odd.o").unwrap(), b"12345");
        assert_eq!(archive.get_member_data("even.o").unwrap(), b"67");
    }

    // -----------------------------------------------------------------------
    // Error: invalid magic
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalid_magic() {
        let data = b"not_an_archive_at_all".to_vec();
        let result = Archive::parse(data);
        assert!(result.is_err());
        match result.unwrap_err() {
            ArchiveError::InvalidMagic => {}
            other => panic!("expected InvalidMagic, got {:?}", other),
        }
    }

    #[test]
    fn test_too_short_for_magic() {
        let data = b"!<ar".to_vec();
        let result = Archive::parse(data);
        assert!(result.is_err());
        match result.unwrap_err() {
            ArchiveError::NotAnArchive => {}
            other => panic!("expected NotAnArchive, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Error: truncated header
    // -----------------------------------------------------------------------

    #[test]
    fn test_truncated_header_in_archive() {
        // Archive magic followed by less than 60 bytes of header.
        let mut data = AR_MAGIC.to_vec();
        data.extend_from_slice(&[b' '; 30]); // Not enough for a header
                                             // This should not be treated as a valid member — our parser
                                             // tolerates trailing bytes shorter than AR_HEADER_SIZE.
        let archive = Archive::parse(data).unwrap();
        assert_eq!(archive.member_count(), 0);
    }

    // -----------------------------------------------------------------------
    // Error: invalid fmag in archive member
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalid_fmag_in_archive() {
        let mut data = AR_MAGIC.to_vec();
        let mut header = [b' '; AR_HEADER_SIZE];
        // size = 0
        header[48] = b'0';
        // Wrong fmag
        header[58] = b'X';
        header[59] = b'X';
        data.extend_from_slice(&header);

        let result = Archive::parse(data);
        assert!(result.is_err());
        match result.unwrap_err() {
            ArchiveError::InvalidHeaderMagic => {}
            other => panic!("expected InvalidHeaderMagic, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Symbol table entry is skipped
    // -----------------------------------------------------------------------

    #[test]
    fn test_symtab_skipped() {
        // Build an archive with a symbol table entry (`/`) followed by a
        // regular member.
        let mut data = AR_MAGIC.to_vec();

        // Symbol table entry: name = "/" (16 bytes)
        let symtab_data = [0u8; 8]; // Fake symbol table data
        let symtab_header = make_raw_header(b"/               ", symtab_data.len());
        data.extend_from_slice(&symtab_header);
        data.extend_from_slice(&symtab_data);

        // Regular member
        let member_data = b"real content";
        let member_header = make_raw_header(b"real.o/         ", member_data.len());
        data.extend_from_slice(&member_header);
        data.extend_from_slice(member_data);

        let archive = Archive::parse(data).unwrap();
        assert_eq!(archive.member_count(), 1);
        assert_eq!(archive.member_names(), vec!["real.o"]);
        assert_eq!(archive.get_member_data("real.o").unwrap(), member_data);
    }

    // -----------------------------------------------------------------------
    // GNU long name resolution
    // -----------------------------------------------------------------------

    #[test]
    fn test_gnu_long_name_resolution() {
        // Build an archive with a long-name table and a member that
        // references it.
        let mut data = AR_MAGIC.to_vec();

        // Long-name table entry: name = "//" (16 bytes)
        // Table content: "very_long_member_name.o/\n"
        let long_names_content = b"very_long_member_name.o/\n";
        let ln_header = make_raw_header(b"//              ", long_names_content.len());
        data.extend_from_slice(&ln_header);
        data.extend_from_slice(long_names_content);
        // Padding to even boundary
        if data.len() % 2 != 0 {
            data.push(b'\n');
        }

        // Member referencing offset 0 in the long-name table
        let member_data = b"elf content here";
        let member_header = make_raw_header(b"/0              ", member_data.len());
        data.extend_from_slice(&member_header);
        data.extend_from_slice(member_data);

        let archive = Archive::parse(data).unwrap();
        assert_eq!(archive.member_count(), 1);
        assert_eq!(archive.member_names(), vec!["very_long_member_name.o"]);
        assert_eq!(
            archive.get_member_data("very_long_member_name.o").unwrap(),
            member_data
        );
    }

    #[test]
    fn test_gnu_long_name_multiple_entries() {
        let mut data = AR_MAGIC.to_vec();

        // Long-name table with two entries separated by "/\n"
        let long_names_content = b"first_long_name.o/\nsecond_long_name.o/\n";
        let ln_header = make_raw_header(b"//              ", long_names_content.len());
        data.extend_from_slice(&ln_header);
        data.extend_from_slice(long_names_content);
        if data.len() % 2 != 0 {
            data.push(b'\n');
        }

        // First member: offset 0 → "first_long_name.o"
        let data1 = b"content1";
        let h1 = make_raw_header(b"/0              ", data1.len());
        data.extend_from_slice(&h1);
        data.extend_from_slice(data1);
        if data.len() % 2 != 0 {
            data.push(b'\n');
        }

        // Second member: offset 19 → "second_long_name.o"
        let data2 = b"content2";
        let h2 = make_raw_header(b"/19             ", data2.len());
        data.extend_from_slice(&h2);
        data.extend_from_slice(data2);

        let archive = Archive::parse(data).unwrap();
        assert_eq!(archive.member_count(), 2);
        assert_eq!(
            archive.member_names(),
            vec!["first_long_name.o", "second_long_name.o"]
        );
    }

    // -----------------------------------------------------------------------
    // Short name parsing
    // -----------------------------------------------------------------------

    #[test]
    fn test_short_name_trimming() {
        let data = make_test_archive(&[("a.o", b"A")]);
        let archive = Archive::parse(data).unwrap();
        // The make_test_archive helper adds a "/" after the name, and
        // resolve_member_name trims it off.
        assert_eq!(archive.member_names(), vec!["a.o"]);
    }

    // -----------------------------------------------------------------------
    // ELF member filtering
    // -----------------------------------------------------------------------

    #[test]
    fn test_elf_members_filters_non_elf() {
        // Create an archive with one ELF member and one non-ELF member.
        let mut elf_data = vec![0x7f, b'E', b'L', b'F']; // ELF magic
        elf_data.extend_from_slice(&[0u8; 12]); // Padding to make it look like ELF
        let non_elf_data = b"not an elf object";

        let data = make_test_archive(&[("elf_obj.o", &elf_data), ("text_file.txt", non_elf_data)]);
        let archive = Archive::parse(data).unwrap();

        let elf = archive.elf_members();
        assert_eq!(elf.len(), 1);
        assert_eq!(elf[0].0, "elf_obj.o");
        assert_eq!(&elf[0].1[..4], &elf::ELF_MAGIC);
    }

    #[test]
    fn test_elf_members_empty_for_non_elf_archive() {
        let data = make_test_archive(&[("readme.txt", b"hello"), ("notes.txt", b"world")]);
        let archive = Archive::parse(data).unwrap();
        assert!(archive.elf_members().is_empty());
    }

    #[test]
    fn test_elf_members_skips_too_short_data() {
        // A member with only 3 bytes cannot match the 4-byte ELF magic.
        let data = make_test_archive(&[("tiny.o", &[0x7f, b'E', b'L'])]);
        let archive = Archive::parse(data).unwrap();
        assert!(archive.elf_members().is_empty());
    }

    // -----------------------------------------------------------------------
    // Error: member data truncated
    // -----------------------------------------------------------------------

    #[test]
    fn test_truncated_member_data() {
        // Build an archive where the header claims 100 bytes of data but
        // only 10 bytes follow.
        let mut data = AR_MAGIC.to_vec();
        let header = make_raw_header(b"trunc.o/        ", 100);
        data.extend_from_slice(&header);
        data.extend_from_slice(&[0u8; 10]); // Only 10 bytes instead of 100

        let result = Archive::parse(data);
        assert!(result.is_err());
        match result.unwrap_err() {
            ArchiveError::TruncatedMember(name) => {
                assert!(name.contains("trunc.o"), "got: {}", name);
            }
            other => panic!("expected TruncatedMember, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Nonexistent member lookup
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_member_data_nonexistent() {
        let data = make_test_archive(&[("exists.o", b"yes")]);
        let archive = Archive::parse(data).unwrap();
        assert!(archive.get_member_data("nope.o").is_none());
    }

    // -----------------------------------------------------------------------
    // ArchiveError Display
    // -----------------------------------------------------------------------

    #[test]
    fn test_archive_error_display() {
        let err = ArchiveError::InvalidMagic;
        let msg = format!("{}", err);
        assert!(msg.contains("invalid ar archive magic"));

        let err2 = ArchiveError::TruncatedMember("foo.o".into());
        let msg2 = format!("{}", err2);
        assert!(msg2.contains("foo.o"));
        assert!(msg2.contains("extends past"));
    }

    // -----------------------------------------------------------------------
    // ArchiveError is std::error::Error
    // -----------------------------------------------------------------------

    #[test]
    fn test_archive_error_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(ArchiveError::NotAnArchive);
        assert!(err.source().is_none());

        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err2: Box<dyn std::error::Error> = Box::new(ArchiveError::IoError(io_err));
        assert!(err2.source().is_some());
    }

    // -----------------------------------------------------------------------
    // BSD long name format (graceful handling)
    // -----------------------------------------------------------------------

    #[test]
    fn test_bsd_long_name_handled() {
        // Build a header with BSD long-name format: #1/20
        let header_result = ArHeader {
            name: "#1/20           ".to_string(),
            modification_time: 0,
            owner_id: 0,
            group_id: 0,
            mode: 0o100644,
            size: 40,
        };
        // The resolve function should not panic on BSD format.
        let result = resolve_member_name(&header_result, &None);
        assert!(result.is_ok());
        let name = result.unwrap();
        // Should contain the BSD prefix as a fallback name
        assert!(name.contains("#1"));
    }

    // -----------------------------------------------------------------------
    // Numeric field parsing edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_decimal_u64_empty() {
        let result = parse_decimal_u64(b"          ", "test");
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn test_parse_decimal_u64_valid() {
        let result = parse_decimal_u64(b"42        ", "test");
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_parse_octal_u32_valid() {
        let result = parse_octal_u32(b"100644  ", "mode");
        assert_eq!(result.unwrap(), 0o100644);
    }

    #[test]
    fn test_parse_octal_u32_empty() {
        let result = parse_octal_u32(b"        ", "mode");
        assert_eq!(result.unwrap(), 0);
    }

    // -----------------------------------------------------------------------
    // Archive with zero-length member
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_length_member() {
        let data = make_test_archive(&[("empty.o", b"")]);
        let archive = Archive::parse(data).unwrap();
        assert_eq!(archive.member_count(), 1);
        assert_eq!(archive.get_member_data("empty.o").unwrap().len(), 0);
    }
}
