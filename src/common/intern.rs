//! String Interning for Identifiers and Literals
//!
//! This module provides a string interner that deduplicates all identifier and
//! string literal values at lexing time, returning compact [`InternId`] handles
//! (u32 indices) for O(1) equality comparison and reduced memory usage. It is a
//! performance-critical component for meeting the SQLite compilation constraints
//! (under 60 seconds, under 2 GB RSS at `-O0` for ~230K LOC).
//!
//! # Design
//!
//! The interner uses a two-field architecture:
//!
//! - A [`HashMap<String, InternId>`] for O(1) amortized deduplication lookup on
//!   the `intern()` hot path.
//! - A [`Vec<String>`] for O(1) reverse resolution from [`InternId`] back to the
//!   interned string slice.
//!
//! This design stores each interned string twice (once as a HashMap key and once
//! in the Vec), which is a deliberate trade-off: it avoids all `unsafe` code and
//! lifetime complexity that an arena-backed approach would require, while keeping
//! the public API simple and fully safe.
//!
//! # Performance
//!
//! - **`intern()`**: O(1) amortized — HashMap lookup, then optional insert.
//!   This is the hot-path method called by the lexer for every identifier token.
//! - **`resolve()`**: O(1) — direct Vec index access.
//! - **`get()`**: O(1) amortized — HashMap lookup without insertion.
//! - **`InternId`**: 4 bytes (u32), far smaller than `String` (24 bytes on
//!   64-bit systems), enabling compact storage in tokens, AST nodes, symbol
//!   tables, and IR values.
//!
//! For the SQLite amalgamation (~230K LOC), expect approximately 10K–50K unique
//! identifiers. The HashMap is pre-sized with `with_capacity()` when using
//! `with_keywords()` to minimize early reallocations.
//!
//! # Integration Points
//!
//! - **`frontend::lexer`** — Every identifier and string literal token carries
//!   an `InternId` instead of a `String`.
//! - **`sema::symbol_table`** — Symbol table keys use `InternId` for O(1)
//!   equality comparison during scope lookup.
//! - **`frontend::parser`** — AST nodes reference identifiers via `InternId`.
//!
//! Per AAP §0.4.1: "All identifier and string literal values interned at lexing
//! time; subsequent phases reference interned handles."
//!
//! # Example
//!
//! ```
//! use bcc::common::intern::{Interner, InternId};
//!
//! let mut interner = Interner::new();
//!
//! let id1 = interner.intern("hello");
//! let id2 = interner.intern("hello"); // Same string → same InternId
//! let id3 = interner.intern("world"); // Different string → different InternId
//!
//! assert_eq!(id1, id2);
//! assert_ne!(id1, id3);
//! assert_eq!(interner.resolve(id1), "hello");
//! assert_eq!(interner.resolve(id3), "world");
//! ```

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// InternId — compact handle type
// ---------------------------------------------------------------------------

/// A compact handle representing an interned string.
///
/// `InternId` is a lightweight 4-byte wrapper around a `u32` index. It serves
/// as the canonical reference to a string that has been deduplicated through the
/// [`Interner`]. Comparing two `InternId` values is a single integer comparison
/// (O(1)), regardless of the length of the underlying strings.
///
/// `InternId` derives `Copy` for cheap value passing, `Hash` for use as
/// `HashMap` keys (e.g., in symbol tables), and `Ord` for deterministic
/// ordering in sorted data structures.
///
/// To retrieve the original string, pass the `InternId` to
/// [`Interner::resolve()`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InternId(u32);

impl InternId {
    /// Returns the raw underlying index value.
    ///
    /// This is primarily useful for serialization, debug logging, and
    /// diagnostics. The returned value is the zero-based index into the
    /// interner's string storage.
    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0
    }

    /// Creates an `InternId` from a raw index value.
    ///
    /// # Safety Contract (Logical)
    ///
    /// The caller must ensure that `raw` is a valid index previously returned
    /// by [`Interner::intern()`] on the same interner instance. Using an
    /// out-of-range index with [`Interner::resolve()`] will cause a panic.
    /// This method is safe in the Rust memory-safety sense (no `unsafe`), but
    /// misuse leads to logic errors.
    #[inline]
    pub fn from_raw(raw: u32) -> Self {
        InternId(raw)
    }
}

/// Displays the `InternId` as its underlying numeric index.
///
/// This is useful for debug output and diagnostics where the actual string
/// content is not readily available. For example, `InternId(42)` displays
/// as `42`.
impl fmt::Display for InternId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Interner — string deduplication data structure
// ---------------------------------------------------------------------------

/// A string interner that deduplicates strings and returns compact handles.
///
/// The `Interner` maps string content to [`InternId`] handles, ensuring that
/// each unique string is stored exactly once (in the interner's internal
/// storage). Subsequent calls to [`intern()`](Interner::intern) with the same
/// string content return the same `InternId`, enabling O(1) equality checks
/// via integer comparison.
///
/// # Thread Safety
///
/// `Interner` is **not** `Sync` or `Send` by default (it uses owned `String`
/// values with no internal synchronization). In the `bcc` compiler pipeline,
/// the interner is created once in the driver and passed through the sequential
/// compilation phases by mutable reference.
pub struct Interner {
    /// Map from string content to its interned ID for O(1) deduplication lookup.
    ///
    /// The HashMap key is an owned `String` matching the content in `strings`.
    /// This duplication (key in map + entry in Vec) is a deliberate trade-off
    /// to avoid `unsafe` code and lifetime complexity.
    map: HashMap<String, InternId>,

    /// Indexed storage: `InternId.0` indexes into this `Vec` to retrieve the
    /// interned string. The Vec preserves insertion order, so `InternId` values
    /// are sequential starting from 0.
    strings: Vec<String>,
}

impl Interner {
    /// Creates a new, empty `Interner`.
    ///
    /// The internal HashMap and Vec start with default capacity. For workloads
    /// where the approximate number of unique strings is known in advance,
    /// consider using [`with_keywords()`](Interner::with_keywords) which
    /// pre-sizes the storage and pre-interns C11 keywords.
    ///
    /// # Examples
    ///
    /// ```
    /// let interner = bcc::common::intern::Interner::new();
    /// assert!(interner.is_empty());
    /// assert_eq!(interner.len(), 0);
    /// ```
    #[inline]
    pub fn new() -> Self {
        Interner {
            map: HashMap::new(),
            strings: Vec::new(),
        }
    }

    /// Creates a new `Interner` pre-populated with all C11 keywords and
    /// common GCC extension keywords.
    ///
    /// Pre-interning keywords serves two purposes:
    ///
    /// 1. **Stable InternId assignment** — Keywords receive deterministic,
    ///    low-valued `InternId` handles, enabling fast keyword identification
    ///    via simple integer range checks instead of string comparisons.
    /// 2. **Reduced reallocation** — The HashMap is pre-sized with
    ///    `with_capacity()` to accommodate the keyword count plus headroom
    ///    for user identifiers, avoiding early rehashing during lexing.
    ///
    /// The following keyword categories are pre-interned:
    ///
    /// - **C11 standard keywords** (44 keywords): `auto`, `break`, `case`,
    ///   `char`, `const`, ... `while`, `_Alignas`, `_Alignof`, `_Atomic`,
    ///   `_Bool`, `_Complex`, `_Generic`, `_Imaginary`, `_Noreturn`,
    ///   `_Static_assert`, `_Thread_local`, `inline`, `restrict`
    /// - **GCC extension keywords**: `__attribute__`, `__asm__`, `__typeof__`,
    ///   `typeof`, `__extension__`, `__inline__`, `__volatile__`, `asm`,
    ///   `__builtin_va_start`, `__builtin_va_end`, `__builtin_va_arg`,
    ///   `__builtin_va_copy`, `__builtin_offsetof`, `__builtin_types_compatible_p`,
    ///   `__int128`, `__label__`, `__auto_type`
    ///
    /// # Examples
    ///
    /// ```
    /// let interner = bcc::common::intern::Interner::with_keywords();
    /// assert!(!interner.is_empty());
    /// // "int" is pre-interned, so get() finds it immediately
    /// assert!(interner.get("int").is_some());
    /// ```
    pub fn with_keywords() -> Self {
        // C11 standard keywords (44 keywords)
        const C11_KEYWORDS: &[&str] = &[
            "auto",
            "break",
            "case",
            "char",
            "const",
            "continue",
            "default",
            "do",
            "double",
            "else",
            "enum",
            "extern",
            "float",
            "for",
            "goto",
            "if",
            "inline",
            "int",
            "long",
            "register",
            "restrict",
            "return",
            "short",
            "signed",
            "sizeof",
            "static",
            "struct",
            "switch",
            "typedef",
            "union",
            "unsigned",
            "void",
            "volatile",
            "while",
            "_Alignas",
            "_Alignof",
            "_Atomic",
            "_Bool",
            "_Complex",
            "_Generic",
            "_Imaginary",
            "_Noreturn",
            "_Static_assert",
            "_Thread_local",
        ];

        // GCC extension keywords
        const GCC_KEYWORDS: &[&str] = &[
            "__attribute__",
            "__attribute",
            "__asm__",
            "__asm",
            "asm",
            "__typeof__",
            "__typeof",
            "typeof",
            "__extension__",
            "__inline__",
            "__inline",
            "__volatile__",
            "__volatile",
            "__const__",
            "__const",
            "__restrict__",
            "__restrict",
            "__signed__",
            "__signed",
            "__builtin_va_start",
            "__builtin_va_end",
            "__builtin_va_arg",
            "__builtin_va_copy",
            "__builtin_va_list",
            "__builtin_offsetof",
            "__builtin_types_compatible_p",
            "__builtin_expect",
            "__builtin_unreachable",
            "__builtin_constant_p",
            "__builtin_choose_expr",
            "__builtin_bswap16",
            "__builtin_bswap32",
            "__builtin_bswap64",
            "__builtin_clz",
            "__builtin_ctz",
            "__builtin_popcount",
            "__builtin_ffs",
            "__builtin_abs",
            "__builtin_fabsf",
            "__builtin_fabs",
            "__builtin_inf",
            "__builtin_inff",
            "__builtin_huge_val",
            "__builtin_huge_valf",
            "__builtin_nan",
            "__builtin_nanf",
            "__builtin_trap",
            "__int128",
            "__label__",
            "__auto_type",
        ];

        let total_keywords = C11_KEYWORDS.len() + GCC_KEYWORDS.len();
        // Pre-size with extra capacity for user identifiers. For a codebase
        // like SQLite (~230K LOC) we expect ~10K–50K unique identifiers.
        let initial_capacity = total_keywords + 4096;

        let mut interner = Interner {
            map: HashMap::with_capacity(initial_capacity),
            strings: Vec::with_capacity(initial_capacity),
        };

        for &keyword in C11_KEYWORDS.iter().chain(GCC_KEYWORDS.iter()) {
            interner.intern(keyword);
        }

        interner
    }

    /// Interns a string, returning its unique [`InternId`] handle.
    ///
    /// If `s` has already been interned, the existing `InternId` is returned
    /// immediately (O(1) HashMap lookup). Otherwise, a new `InternId` is
    /// allocated (the next sequential index), the string is stored, and the
    /// new handle is returned.
    ///
    /// This is the **primary hot-path method** called by the lexer for every
    /// identifier and string literal token. Its performance is critical for
    /// meeting the SQLite compilation time constraint.
    ///
    /// # Arguments
    ///
    /// * `s` — The string slice to intern. The interner takes ownership of a
    ///   copy; the caller's reference is not retained.
    ///
    /// # Returns
    ///
    /// The [`InternId`] handle uniquely identifying the interned string.
    ///
    /// # Panics
    ///
    /// Panics if the number of interned strings exceeds `u32::MAX` (over 4
    /// billion unique strings). This limit is effectively unreachable for any
    /// realistic C compilation workload.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut interner = bcc::common::intern::Interner::new();
    /// let id1 = interner.intern("foo");
    /// let id2 = interner.intern("foo");
    /// assert_eq!(id1, id2); // Same string → same InternId
    /// ```
    #[inline]
    pub fn intern(&mut self, s: &str) -> InternId {
        // Fast path: check if the string is already interned.
        // HashMap::get accepts &str directly thanks to Borrow<str> on String.
        if let Some(&id) = self.map.get(s) {
            return id;
        }

        // Slow path: allocate a new InternId and store the string.
        let index = self.strings.len();
        assert!(
            index <= u32::MAX as usize,
            "intern: exceeded maximum InternId capacity (u32::MAX)"
        );
        let id = InternId(index as u32);

        // Allocate the owned string once, clone for HashMap key.
        // This is the minimal allocation strategy without unsafe code:
        // one heap allocation for the owned String, one clone for the
        // HashMap key. The Vec and HashMap both need ownership.
        let owned = s.to_string();
        self.map.insert(owned.clone(), id);
        self.strings.push(owned);

        id
    }

    /// Resolves an [`InternId`] back to its interned string slice.
    ///
    /// This is an O(1) operation via direct Vec index access.
    ///
    /// # Arguments
    ///
    /// * `id` — The `InternId` handle to resolve, as previously returned by
    ///   [`intern()`](Interner::intern).
    ///
    /// # Returns
    ///
    /// A string slice reference with the lifetime of the `Interner`.
    ///
    /// # Panics
    ///
    /// Panics if `id` was not produced by this interner instance (i.e., if
    /// `id.as_u32()` is out of bounds). In correct usage, this should never
    /// occur because `InternId` values are only created by `intern()`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut interner = bcc::common::intern::Interner::new();
    /// let id = interner.intern("bar");
    /// assert_eq!(interner.resolve(id), "bar");
    /// ```
    #[inline]
    pub fn resolve(&self, id: InternId) -> &str {
        &self.strings[id.0 as usize]
    }

    /// Checks whether a string has already been interned, without inserting it.
    ///
    /// Returns `Some(id)` if the string is found in the interner, or `None`
    /// if it has not been interned yet. This method does not modify the
    /// interner's state.
    ///
    /// # Arguments
    ///
    /// * `s` — The string slice to look up.
    ///
    /// # Returns
    ///
    /// `Some(InternId)` if `s` has been previously interned, `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut interner = bcc::common::intern::Interner::new();
    /// assert_eq!(interner.get("missing"), None);
    ///
    /// let id = interner.intern("present");
    /// assert_eq!(interner.get("present"), Some(id));
    /// ```
    #[inline]
    pub fn get(&self, s: &str) -> Option<InternId> {
        // HashMap::get with &str works because String implements Borrow<str>.
        self.map.get(s).copied()
    }

    /// Returns the number of unique strings currently interned.
    ///
    /// This count includes any pre-interned keywords if the interner was
    /// created via [`with_keywords()`](Interner::with_keywords).
    ///
    /// # Examples
    ///
    /// ```
    /// let mut interner = bcc::common::intern::Interner::new();
    /// assert_eq!(interner.len(), 0);
    /// interner.intern("a");
    /// interner.intern("b");
    /// interner.intern("a"); // duplicate, not counted twice
    /// assert_eq!(interner.len(), 2);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Returns `true` if the interner contains no interned strings.
    ///
    /// # Examples
    ///
    /// ```
    /// let interner = bcc::common::intern::Interner::new();
    /// assert!(interner.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }

    /// Checks whether the interner contains a given string.
    ///
    /// This is a convenience wrapper around [`get()`](Interner::get) that
    /// returns a boolean instead of an `Option<InternId>`.
    ///
    /// # Arguments
    ///
    /// * `s` — The string slice to check for.
    ///
    /// # Returns
    ///
    /// `true` if the string has been interned, `false` otherwise.
    #[inline]
    pub fn contains_key(&self, s: &str) -> bool {
        self.map.contains_key(s)
    }
}

/// Debug formatting for the `Interner` displays the count of interned strings
/// and the first few entries for diagnostic purposes.
impl fmt::Debug for Interner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Interner")
            .field("count", &self.strings.len())
            .field(
                "strings_preview",
                &if self.strings.len() <= 8 {
                    self.strings.as_slice()
                } else {
                    &self.strings[..8]
                },
            )
            .finish()
    }
}

/// Default trait implementation creates an empty interner.
impl Default for Interner {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- InternId basic properties ---

    #[test]
    fn intern_id_is_copy() {
        let id = InternId(42);
        let id_copy = id; // Copy
        assert_eq!(id, id_copy);
        // Both still usable (Copy, not moved)
        assert_eq!(id.as_u32(), 42);
        assert_eq!(id_copy.as_u32(), 42);
    }

    #[test]
    fn intern_id_equality() {
        let a = InternId(0);
        let b = InternId(0);
        let c = InternId(1);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn intern_id_ordering() {
        let a = InternId(0);
        let b = InternId(1);
        let c = InternId(2);
        assert!(a < b);
        assert!(b < c);
        assert!(a < c);
    }

    #[test]
    fn intern_id_hash() {
        use std::collections::HashMap;
        let mut map: HashMap<InternId, &str> = HashMap::new();
        let id = InternId(7);
        map.insert(id, "test");
        assert_eq!(map.get(&id), Some(&"test"));
    }

    #[test]
    fn intern_id_display() {
        let id = InternId(123);
        assert_eq!(format!("{}", id), "123");
    }

    #[test]
    fn intern_id_debug() {
        let id = InternId(5);
        assert_eq!(format!("{:?}", id), "InternId(5)");
    }

    #[test]
    fn intern_id_from_raw_roundtrip() {
        let id = InternId::from_raw(99);
        assert_eq!(id.as_u32(), 99);
    }

    // --- Interner::new() ---

    #[test]
    fn new_interner_is_empty() {
        let interner = Interner::new();
        assert!(interner.is_empty());
        assert_eq!(interner.len(), 0);
    }

    // --- Interner::intern() and resolve() ---

    #[test]
    fn intern_returns_valid_id() {
        let mut interner = Interner::new();
        let id = interner.intern("hello");
        assert_eq!(id.as_u32(), 0); // First interned string gets index 0
    }

    #[test]
    fn intern_same_string_returns_same_id() {
        let mut interner = Interner::new();
        let id1 = interner.intern("hello");
        let id2 = interner.intern("hello");
        assert_eq!(id1, id2);
        assert_eq!(interner.len(), 1); // Only one unique string
    }

    #[test]
    fn intern_different_strings_returns_different_ids() {
        let mut interner = Interner::new();
        let id1 = interner.intern("hello");
        let id2 = interner.intern("world");
        assert_ne!(id1, id2);
        assert_eq!(interner.len(), 2);
    }

    #[test]
    fn resolve_returns_correct_string() {
        let mut interner = Interner::new();
        let id_hello = interner.intern("hello");
        let id_world = interner.intern("world");
        assert_eq!(interner.resolve(id_hello), "hello");
        assert_eq!(interner.resolve(id_world), "world");
    }

    #[test]
    fn intern_resolve_roundtrip() {
        let mut interner = Interner::new();
        let strings = vec!["alpha", "beta", "gamma", "delta", "epsilon"];
        let ids: Vec<InternId> = strings.iter().map(|s| interner.intern(s)).collect();

        for (s, id) in strings.iter().zip(ids.iter()) {
            assert_eq!(interner.resolve(*id), *s);
        }
    }

    #[test]
    fn intern_empty_string() {
        let mut interner = Interner::new();
        let id = interner.intern("");
        assert_eq!(interner.resolve(id), "");
        assert_eq!(interner.len(), 1);
    }

    #[test]
    fn intern_unicode_strings() {
        let mut interner = Interner::new();
        let id1 = interner.intern("café");
        let id2 = interner.intern("naïve");
        let id3 = interner.intern("café"); // duplicate
        assert_eq!(id1, id3);
        assert_ne!(id1, id2);
        assert_eq!(interner.resolve(id1), "café");
        assert_eq!(interner.resolve(id2), "naïve");
    }

    #[test]
    fn intern_strings_with_special_chars() {
        let mut interner = Interner::new();
        let id1 = interner.intern("hello\nworld");
        let id2 = interner.intern("tab\there");
        let id3 = interner.intern("null\0byte");
        assert_eq!(interner.resolve(id1), "hello\nworld");
        assert_eq!(interner.resolve(id2), "tab\there");
        assert_eq!(interner.resolve(id3), "null\0byte");
        assert_eq!(interner.len(), 3);
    }

    // --- Interner::get() ---

    #[test]
    fn get_returns_none_for_non_interned() {
        let interner = Interner::new();
        assert_eq!(interner.get("missing"), None);
    }

    #[test]
    fn get_returns_some_for_interned() {
        let mut interner = Interner::new();
        let id = interner.intern("present");
        assert_eq!(interner.get("present"), Some(id));
    }

    #[test]
    fn get_does_not_modify_interner() {
        let mut interner = Interner::new();
        interner.intern("one");
        let len_before = interner.len();
        let _ = interner.get("two"); // Should not insert "two"
        assert_eq!(interner.len(), len_before);
        assert_eq!(interner.get("two"), None);
    }

    // --- Interner::len() and is_empty() ---

    #[test]
    fn len_tracks_unique_strings() {
        let mut interner = Interner::new();
        assert_eq!(interner.len(), 0);

        interner.intern("a");
        assert_eq!(interner.len(), 1);

        interner.intern("b");
        assert_eq!(interner.len(), 2);

        interner.intern("a"); // duplicate
        assert_eq!(interner.len(), 2);

        interner.intern("c");
        assert_eq!(interner.len(), 3);
    }

    #[test]
    fn is_empty_reflects_state() {
        let mut interner = Interner::new();
        assert!(interner.is_empty());

        interner.intern("x");
        assert!(!interner.is_empty());
    }

    // --- Interner::contains_key() ---

    #[test]
    fn contains_key_works() {
        let mut interner = Interner::new();
        assert!(!interner.contains_key("test"));
        interner.intern("test");
        assert!(interner.contains_key("test"));
        assert!(!interner.contains_key("other"));
    }

    // --- Interner::with_keywords() ---

    #[test]
    fn with_keywords_pre_interns_c11_keywords() {
        let interner = Interner::with_keywords();
        assert!(!interner.is_empty());

        // Check a sample of C11 keywords
        let c11_sample = [
            "auto",
            "break",
            "case",
            "char",
            "const",
            "continue",
            "default",
            "do",
            "double",
            "else",
            "enum",
            "extern",
            "float",
            "for",
            "goto",
            "if",
            "inline",
            "int",
            "long",
            "register",
            "restrict",
            "return",
            "short",
            "signed",
            "sizeof",
            "static",
            "struct",
            "switch",
            "typedef",
            "union",
            "unsigned",
            "void",
            "volatile",
            "while",
            "_Alignas",
            "_Alignof",
            "_Atomic",
            "_Bool",
            "_Complex",
            "_Generic",
            "_Imaginary",
            "_Noreturn",
            "_Static_assert",
            "_Thread_local",
        ];
        for kw in &c11_sample {
            assert!(
                interner.get(kw).is_some(),
                "C11 keyword '{}' should be pre-interned",
                kw
            );
        }
    }

    #[test]
    fn with_keywords_pre_interns_gcc_extensions() {
        let interner = Interner::with_keywords();

        let gcc_sample = [
            "__attribute__",
            "__asm__",
            "__typeof__",
            "typeof",
            "__extension__",
            "__inline__",
            "__volatile__",
            "asm",
            "__builtin_va_start",
            "__builtin_va_end",
            "__builtin_va_arg",
            "__builtin_va_copy",
            "__builtin_offsetof",
            "__int128",
        ];
        for kw in &gcc_sample {
            assert!(
                interner.get(kw).is_some(),
                "GCC keyword '{}' should be pre-interned",
                kw
            );
        }
    }

    #[test]
    fn with_keywords_idempotent_ids() {
        let mut interner = Interner::with_keywords();

        // Interning an already-interned keyword should return the same ID
        let id_int_1 = interner.get("int").unwrap();
        let id_int_2 = interner.intern("int");
        assert_eq!(id_int_1, id_int_2);

        // The count should not change
        let count_before = interner.len();
        interner.intern("void");
        assert_eq!(interner.len(), count_before); // "void" was already interned
    }

    #[test]
    fn with_keywords_allows_new_strings() {
        let mut interner = Interner::with_keywords();
        let count_before = interner.len();

        let id = interner.intern("my_custom_identifier");
        assert_eq!(interner.len(), count_before + 1);
        assert_eq!(interner.resolve(id), "my_custom_identifier");
    }

    // --- Sequential ID assignment ---

    #[test]
    fn ids_are_sequential() {
        let mut interner = Interner::new();
        let id0 = interner.intern("first");
        let id1 = interner.intern("second");
        let id2 = interner.intern("third");
        assert_eq!(id0.as_u32(), 0);
        assert_eq!(id1.as_u32(), 1);
        assert_eq!(id2.as_u32(), 2);
    }

    // --- Stress test: many strings ---

    #[test]
    fn intern_many_strings() {
        let mut interner = Interner::new();
        let count = 10_000;

        // Intern many unique strings
        let ids: Vec<InternId> = (0..count)
            .map(|i| interner.intern(&format!("string_{}", i)))
            .collect();

        assert_eq!(interner.len(), count);

        // Verify all resolve correctly
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(interner.resolve(*id), format!("string_{}", i));
        }

        // Verify re-interning returns the same IDs
        for (i, expected_id) in ids.iter().enumerate() {
            let actual_id = interner.intern(&format!("string_{}", i));
            assert_eq!(*expected_id, actual_id);
        }

        // Count should still be the same (no new strings added)
        assert_eq!(interner.len(), count);
    }

    #[test]
    fn intern_many_duplicates() {
        let mut interner = Interner::new();

        // Intern the same small set of strings many times
        let strings = ["a", "b", "c", "d", "e"];
        for _ in 0..1000 {
            for s in &strings {
                interner.intern(s);
            }
        }
        assert_eq!(interner.len(), 5);
    }

    // --- Default trait ---

    #[test]
    fn default_creates_empty_interner() {
        let interner: Interner = Default::default();
        assert!(interner.is_empty());
    }

    // --- Debug formatting ---

    #[test]
    fn debug_format() {
        let mut interner = Interner::new();
        interner.intern("hello");
        interner.intern("world");
        let debug_str = format!("{:?}", interner);
        assert!(debug_str.contains("Interner"));
        assert!(debug_str.contains("count"));
        assert!(debug_str.contains("2"));
    }

    // --- Edge cases ---

    #[test]
    fn intern_long_string() {
        let mut interner = Interner::new();
        let long_str: String = "a".repeat(100_000);
        let id = interner.intern(&long_str);
        assert_eq!(interner.resolve(id), long_str.as_str());
    }

    #[test]
    fn intern_similar_strings() {
        let mut interner = Interner::new();
        let id1 = interner.intern("abc");
        let id2 = interner.intern("abd");
        let id3 = interner.intern("ab");
        let id4 = interner.intern("abcd");
        assert_ne!(id1, id2);
        assert_ne!(id1, id3);
        assert_ne!(id1, id4);
        assert_eq!(interner.len(), 4);
    }

    #[test]
    fn intern_case_sensitive() {
        let mut interner = Interner::new();
        let id_lower = interner.intern("hello");
        let id_upper = interner.intern("Hello");
        let id_all_upper = interner.intern("HELLO");
        assert_ne!(id_lower, id_upper);
        assert_ne!(id_lower, id_all_upper);
        assert_ne!(id_upper, id_all_upper);
        assert_eq!(interner.len(), 3);
    }

    #[test]
    fn intern_c_identifiers() {
        // Test with realistic C identifiers that the lexer would encounter
        let mut interner = Interner::new();
        let identifiers = [
            "main",
            "printf",
            "argc",
            "argv",
            "NULL",
            "size_t",
            "uint32_t",
            "__attribute__",
            "_exit",
            "SQLITE_OK",
            "sqlite3_open",
            "REDIS_ERR",
        ];
        let ids: Vec<InternId> = identifiers.iter().map(|s| interner.intern(s)).collect();
        assert_eq!(interner.len(), identifiers.len());

        for (s, id) in identifiers.iter().zip(ids.iter()) {
            assert_eq!(interner.resolve(*id), *s);
        }
    }
}
