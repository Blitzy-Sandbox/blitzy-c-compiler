//! Arena Allocator for AST and IR Nodes
//!
//! This module provides a chunk-based arena allocator that supports O(1) bump
//! allocation and batch deallocation. It is the primary allocation strategy for
//! AST nodes during parsing and IR nodes during IR construction, enabling the
//! compiler to meet its performance constraints:
//!
//! - **O(1) allocation**: Each `alloc()` call simply bumps a pointer within
//!   the current chunk, with no per-object bookkeeping.
//! - **Cache-friendly**: Sequentially allocated objects reside in contiguous
//!   memory, maximizing cache line utilization during tree traversals.
//! - **Batch deallocation**: All arena memory is freed at once when the `Arena`
//!   is dropped — no per-object destructors, no free-list management.
//! - **Memory efficiency**: Critical for compiling large codebases such as the
//!   SQLite amalgamation (~230K LOC) within the <2 GB peak RSS constraint.
//!
//! # Design
//!
//! The arena manages a list of memory chunks. When the current chunk is
//! exhausted, a new chunk is allocated with a growing capacity strategy
//! (doubling up to a configurable maximum). This amortizes the cost of
//! system allocator calls over many small allocations.
//!
//! Interior mutability via `RefCell` allows allocation through shared `&self`
//! references, which is essential for the parser and IR builder patterns where
//! the arena is shared across multiple components during a compilation pass.
//!
//! # Safety
//!
//! The arena uses `unsafe` code internally for raw memory management. Every
//! `unsafe` block carries a three-part safety comment documenting the relied-upon
//! invariant, why a safe abstraction is insufficient, and the scope of the
//! unsafe region. The public API is entirely safe: references returned by
//! `alloc()` are tied to the arena's lifetime, preventing use-after-free.
//!
//! # Important Note on Drop
//!
//! Objects allocated in the arena do **not** have their `Drop` implementations
//! called when the arena is freed. This is by design — AST and IR nodes should
//! be plain data types (`Copy` or simple aggregates) that do not require
//! destruction. Allocating types with significant `Drop` behavior (e.g., types
//! owning heap allocations outside the arena) will result in resource leaks.

use std::alloc::{self, handle_alloc_error, Layout};
use std::cell::RefCell;
use std::cmp;
use std::marker::PhantomData;
use std::mem;
use std::ptr;

/// Default initial chunk capacity in bytes (8 KiB).
///
/// Chosen as a balance between minimizing wasted memory for small compilations
/// and reducing the number of chunk allocations for larger ones. Two pages
/// provides good amortization for typical function-sized AST subtrees.
const DEFAULT_CHUNK_SIZE: usize = 8192;

/// Maximum chunk capacity in bytes (1 MiB).
///
/// Caps the doubling growth strategy to prevent a single chunk from consuming
/// excessive memory. Once chunks reach this size, subsequent chunks are also
/// allocated at this capacity.
const MAX_CHUNK_SIZE: usize = 1024 * 1024;

/// Alignment used for all chunk backing allocations (16 bytes).
///
/// 16-byte alignment satisfies the alignment requirements of all primitive
/// types on all supported target architectures (x86-64, i686, AArch64, RV64)
/// and ensures SSE/NEON compatibility for any SIMD-aligned types that may
/// be stored in the arena.
const CHUNK_ALIGNMENT: usize = 16;

// ---------------------------------------------------------------------------
// Chunk — internal contiguous memory block
// ---------------------------------------------------------------------------

/// A contiguous block of memory used as backing storage for arena allocations.
///
/// Each chunk is allocated from the global allocator with a fixed capacity and
/// tracks a monotonically increasing offset that serves as the bump pointer.
/// When the offset reaches the capacity, the arena allocates a new chunk.
struct Chunk {
    /// Raw pointer to the start of the allocated memory region.
    data: *mut u8,
    /// Total usable capacity of this chunk in bytes.
    capacity: usize,
    /// Current allocation watermark within this chunk (bytes consumed so far).
    /// Always satisfies `offset <= capacity`.
    offset: usize,
}

impl Chunk {
    /// Allocates a new chunk with the given capacity.
    ///
    /// The backing memory is obtained from the global allocator with
    /// `CHUNK_ALIGNMENT`-byte alignment. Panics (via `handle_alloc_error`)
    /// if the system allocator fails.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero or if the layout cannot be constructed
    /// (e.g., capacity overflows when combined with alignment).
    fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Chunk capacity must be non-zero");
        let layout = Layout::from_size_align(capacity, CHUNK_ALIGNMENT)
            .expect("Chunk layout construction failed: capacity too large");
        // SAFETY: [Invariant] We allocate a fresh block of `capacity` bytes with
        // CHUNK_ALIGNMENT alignment. The layout is valid (non-zero size, power-of-two
        // alignment) as guaranteed by the Layout::from_size_align call above.
        // [Insufficiency] Safe Rust provides no API for allocating untyped, uninitialized
        // memory of a dynamically determined size; Vec<u8> would zero-initialize and
        // impose its own metadata overhead incompatible with bump allocation semantics.
        // [Scope] Single allocation of chunk backing storage; the returned pointer is
        // stored in `self.data` and freed in the Drop implementation.
        let data = unsafe { alloc::alloc(layout) };
        if data.is_null() {
            handle_alloc_error(layout);
        }
        Chunk {
            data,
            capacity,
            offset: 0,
        }
    }

    /// Returns the number of bytes remaining in this chunk.
    #[inline]
    fn remaining(&self) -> usize {
        self.capacity - self.offset
    }

    /// Attempts to bump-allocate `layout.size()` bytes with `layout.align()`
    /// alignment from this chunk.
    ///
    /// Returns `Some(ptr)` on success (advancing the offset), or `None` if the
    /// chunk does not have enough remaining capacity to satisfy the request
    /// after alignment padding.
    #[inline]
    fn try_alloc(&mut self, layout: Layout) -> Option<*mut u8> {
        let align = layout.align();
        let size = layout.size();

        // Round up the current offset to the required alignment.
        let aligned_offset = (self.offset + align - 1) & !(align - 1);

        if aligned_offset + size > self.capacity {
            return None;
        }

        // SAFETY: [Invariant] `self.data` is a valid pointer to `self.capacity` bytes
        // of allocated memory (established in Chunk::new). `aligned_offset` satisfies
        // `aligned_offset + size <= self.capacity`, so `self.data.add(aligned_offset)`
        // is within bounds of the allocation.
        // [Insufficiency] Pointer arithmetic on raw allocator memory cannot be expressed
        // through safe slice indexing because the memory is uninitialized and untyped.
        // [Scope] Single pointer offset computation within the chunk's valid range.
        let ptr = unsafe { self.data.add(aligned_offset) };
        self.offset = aligned_offset + size;
        Some(ptr)
    }
}

impl Drop for Chunk {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.capacity, CHUNK_ALIGNMENT)
            .expect("Chunk layout reconstruction failed during drop");
        // SAFETY: [Invariant] `self.data` was allocated via `alloc::alloc` with this
        // exact layout in `Chunk::new()`. The pointer has not been deallocated elsewhere
        // because `Chunk` is the sole owner and `Drop` is called exactly once.
        // [Insufficiency] The global allocator's dealloc function requires a raw pointer
        // and the original Layout; there is no safe wrapper for freeing untyped memory.
        // [Scope] Single deallocation of this chunk's backing storage. After this call,
        // `self.data` is dangling, but the Chunk is being dropped so no further access
        // occurs.
        unsafe { alloc::dealloc(self.data, layout) }
    }
}

// ---------------------------------------------------------------------------
// Arena — the public arena allocator
// ---------------------------------------------------------------------------

/// A chunk-based arena allocator providing O(1) bump allocation and batch
/// deallocation for AST nodes, IR nodes, and interned strings.
///
/// # Usage
///
/// ```ignore
/// let arena = Arena::new();
/// let x: &i32 = arena.alloc(42);
/// let s: &str = arena.alloc_str("hello");
/// assert_eq!(*x, 42);
/// assert_eq!(s, "hello");
/// // All memory is freed when `arena` is dropped.
/// ```
///
/// # Interior Mutability
///
/// The arena uses `RefCell` internally so that `alloc` methods take `&self`
/// (shared reference) rather than `&mut self`. This allows the arena to be
/// shared across multiple components (e.g., parser and AST builder) without
/// requiring exclusive mutable access at each allocation site.
pub struct Arena {
    /// Backing memory chunks, wrapped in `RefCell` for interior mutability.
    chunks: RefCell<Vec<Chunk>>,
}

impl Arena {
    /// Creates a new, empty arena.
    ///
    /// The first chunk is allocated lazily on the first `alloc` call, so
    /// constructing an arena is virtually free.
    pub fn new() -> Self {
        Arena {
            chunks: RefCell::new(Vec::new()),
        }
    }

    /// Creates a new arena with an initial chunk of the specified capacity.
    ///
    /// Use this when the approximate total allocation size is known in advance
    /// to avoid early chunk exhaustion and reallocation. For example, when
    /// parsing a large file, pre-sizing the arena reduces chunk transitions.
    ///
    /// # Panics
    ///
    /// Panics if `bytes` is zero.
    pub fn with_capacity(bytes: usize) -> Self {
        assert!(bytes > 0, "Arena initial capacity must be non-zero");
        let capacity = cmp::max(bytes, CHUNK_ALIGNMENT);
        let arena = Arena {
            chunks: RefCell::new(Vec::with_capacity(4)),
        };
        arena.chunks.borrow_mut().push(Chunk::new(capacity));
        arena
    }

    /// Allocates space for a value of type `T` in the arena and writes the
    /// value into it, returning a reference with the arena's lifetime.
    ///
    /// This is the primary allocation method for AST and IR nodes. The returned
    /// reference is valid for the entire lifetime of the `Arena`.
    ///
    /// # Zero-Sized Types
    ///
    /// For zero-sized types, no actual memory is consumed; a properly aligned
    /// dangling pointer is returned (consistent with Rust's ZST semantics).
    pub fn alloc<T>(&self, value: T) -> &T {
        let layout = Layout::new::<T>();

        // Handle zero-sized types without consuming arena memory.
        if layout.size() == 0 {
            // SAFETY: [Invariant] For zero-sized types, `std::ptr::dangling_mut::<T>()`
            // produces a well-aligned, non-null pointer that is valid for ZST references.
            // `ptr::write` on a ZST is a no-op (writes zero bytes). The resulting reference
            // is valid because ZST references do not dereference any actual memory.
            // [Insufficiency] There is no safe API to create a reference to a ZST without
            // allocating actual storage; the dangling pointer pattern is the standard
            // approach used by Rust's standard library for ZST handling.
            // [Scope] Single typed write (no-op for ZST) and reference creation.
            let ptr = unsafe {
                let raw = ptr::dangling_mut::<T>();
                ptr::write(raw, value);
                &*raw
            };
            return ptr;
        }

        let raw = self.alloc_raw(layout);
        // SAFETY: [Invariant] `raw` is a freshly allocated pointer from the arena with
        // correct size (`mem::size_of::<T>()` bytes) and alignment (`mem::align_of::<T>()`)
        // for type `T`. The memory is uninitialized, so we use `ptr::write` to place the
        // value without reading the previous (undefined) contents. The resulting reference
        // is valid for the lifetime of the Arena because chunk memory is not freed until
        // the Arena is dropped.
        // [Insufficiency] Safe Rust cannot write a value into uninitialized memory obtained
        // from a raw allocator; `ptr::write` is required to avoid a read of uninitialized
        // data that `*ptr = value` would perform (which is UB for non-Copy types).
        // [Scope] Single typed write of `value` into arena-allocated memory, followed by
        // creation of a shared reference to the written value.
        unsafe {
            (raw as *mut T).write(value);
            &*(raw as *const T)
        }
    }

    /// Allocates a contiguous slice of `T` values in the arena by copying
    /// from the source slice.
    ///
    /// The returned slice reference has the arena's lifetime. This is useful
    /// for storing arrays of AST children, instruction operands, or other
    /// variable-length sequences.
    ///
    /// # Constraints
    ///
    /// `T` must implement `Copy` to ensure that simple bitwise copying is
    /// sufficient and that no `Drop` glue is required for the copied elements.
    pub fn alloc_slice<T: Copy>(&self, slice: &[T]) -> &[T] {
        if slice.is_empty() {
            return &[];
        }

        let size = mem::size_of::<T>();
        let align = mem::align_of::<T>();

        // For zero-sized element types, return a slice backed by a dangling pointer.
        if size == 0 {
            // SAFETY: [Invariant] For ZSTs, `std::slice::from_raw_parts` with a
            // properly aligned dangling pointer and the given length produces a valid
            // slice because no memory is actually accessed through ZST elements.
            // [Insufficiency] There is no safe way to create a non-empty slice of ZSTs
            // without a backing allocation; the dangling-pointer pattern is standard.
            // [Scope] Single construction of a ZST slice reference.
            return unsafe { std::slice::from_raw_parts(ptr::dangling::<T>(), slice.len()) };
        }

        let total_size = size
            .checked_mul(slice.len())
            .expect("Arena::alloc_slice: total size overflow");
        let layout =
            Layout::from_size_align(total_size, align).expect("Arena::alloc_slice: invalid layout");

        let raw = self.alloc_raw(layout);

        // SAFETY: [Invariant] `raw` points to `total_size` bytes of arena memory with
        // alignment >= `align`. `slice` is a valid slice of `T: Copy` elements. We copy
        // `total_size` bytes from the source slice into the arena. The destination does
        // not overlap the source (arena memory is freshly allocated). After the copy, the
        // memory contains valid `T` values.
        // [Insufficiency] `ptr::copy_nonoverlapping` is needed to bitwise-copy from the
        // source slice into uninitialized arena memory; there is no safe bulk-copy API
        // for raw allocator memory. `from_raw_parts` is needed to construct a slice from
        // a raw pointer.
        // [Scope] Bulk copy of `slice.len()` elements into arena memory, followed by
        // slice reference construction.
        unsafe {
            ptr::copy_nonoverlapping(slice.as_ptr() as *const u8, raw, total_size);
            std::slice::from_raw_parts(raw as *const T, slice.len())
        }
    }

    /// Allocates a string in the arena and returns a `&str` reference with
    /// the arena's lifetime.
    ///
    /// This method copies the string's UTF-8 bytes into the arena. It is used
    /// by the string interner for backing storage, enabling interned strings
    /// to live as long as the arena without separate heap allocations.
    pub fn alloc_str(&self, s: &str) -> &str {
        if s.is_empty() {
            return "";
        }
        let bytes = self.alloc_slice(s.as_bytes());
        // SAFETY: [Invariant] `bytes` were copied from `s.as_bytes()`, which is
        // guaranteed to be valid UTF-8 because `s` is a `&str`. The arena does not
        // modify the bytes after copying, so they remain valid UTF-8.
        // [Insufficiency] `std::str::from_utf8_unchecked` avoids re-validating UTF-8
        // on bytes that are already known-valid. The safe alternative
        // `std::str::from_utf8` would add an O(n) validation pass for no benefit.
        // [Scope] Single conversion from known-valid UTF-8 byte slice to `&str`.
        unsafe { std::str::from_utf8_unchecked(bytes) }
    }

    /// Low-level allocation of `layout.size()` bytes with `layout.align()`
    /// alignment from the arena.
    ///
    /// Returns a raw pointer to the allocated region. The memory is
    /// **uninitialized**; the caller is responsible for writing valid data
    /// before reading.
    ///
    /// This method is public to allow advanced use cases (e.g., custom DST
    /// construction), but most callers should prefer `alloc()`, `alloc_slice()`,
    /// or `alloc_str()`.
    ///
    /// # Panics
    ///
    /// Panics if `layout.size()` is zero (use ZST-aware methods instead) or
    /// if the system allocator fails to provide a new chunk.
    pub fn alloc_raw(&self, layout: Layout) -> *mut u8 {
        debug_assert!(layout.size() > 0, "alloc_raw called with zero-size layout");

        let mut chunks = self.chunks.borrow_mut();

        // Fast path: try to allocate from the current (last) chunk.
        if let Some(chunk) = chunks.last_mut() {
            if let Some(ptr) = chunk.try_alloc(layout) {
                return ptr;
            }
        }

        // Slow path: need a new chunk.
        // Compute the new chunk capacity using a doubling growth strategy,
        // ensuring it is at least large enough to satisfy this allocation.
        let prev_capacity = chunks.last().map_or(DEFAULT_CHUNK_SIZE, |c| c.capacity);
        let grown = cmp::min(prev_capacity.saturating_mul(2), MAX_CHUNK_SIZE);
        let min_required = layout.size() + layout.align(); // worst-case alignment waste
        let new_capacity = cmp::max(grown, cmp::max(min_required, DEFAULT_CHUNK_SIZE));

        let mut new_chunk = Chunk::new(new_capacity);

        // The new chunk is empty, so this allocation must succeed (we ensured
        // the capacity is sufficient above).
        let ptr = new_chunk
            .try_alloc(layout)
            .expect("Fresh chunk too small for allocation — this is a bug");

        chunks.push(new_chunk);
        ptr
    }

    /// Returns the total number of bytes allocated across all chunks.
    ///
    /// This counts the total *capacity* of all chunks, not the bytes actually
    /// consumed by user allocations. It is useful for monitoring memory usage
    /// against the <2 GB RSS constraint during large compilations.
    pub fn bytes_allocated(&self) -> usize {
        self.chunks
            .borrow()
            .iter()
            .map(|chunk| chunk.capacity)
            .sum()
    }

    /// Resets the arena for reuse without deallocating the first chunk.
    ///
    /// All allocation offsets are set to zero, and all chunks except the first
    /// are deallocated. This allows the arena to be reused for a new compilation
    /// pass without paying the cost of re-allocating the initial chunk.
    ///
    /// # Safety Invariant
    ///
    /// After calling `reset()`, any previously returned references from
    /// `alloc()`, `alloc_slice()`, or `alloc_str()` are **dangling** and must
    /// not be used. The caller must ensure no such references survive across
    /// a `reset()` call. This is enforced by requiring `&mut self`.
    pub fn reset(&mut self) {
        let chunks = self.chunks.get_mut();
        if chunks.is_empty() {
            return;
        }
        // Keep only the first chunk and reset its offset.
        chunks.truncate(1);
        chunks[0].offset = 0;
    }
}

impl Default for Arena {
    fn default() -> Self {
        Arena::new()
    }
}

// ---------------------------------------------------------------------------
// TypedArena<T> — type-safe arena wrapper
// ---------------------------------------------------------------------------

/// A type-safe wrapper around [`Arena`] that only allocates values of type `T`.
///
/// `TypedArena<T>` provides the same O(1) bump allocation and batch
/// deallocation as `Arena`, but restricts allocations to a single type. This
/// is useful when a module exclusively allocates one kind of node (e.g., all
/// AST expression nodes or all IR instructions).
///
/// # Example
///
/// ```ignore
/// let arena: TypedArena<MyNode> = TypedArena::new();
/// let node: &MyNode = arena.alloc(MyNode { kind: NodeKind::Expr, .. });
/// ```
pub struct TypedArena<T> {
    /// The underlying untyped arena.
    inner: Arena,
    /// Marker to make `TypedArena` generic over `T` without owning a `T`.
    _marker: PhantomData<T>,
}

impl<T> TypedArena<T> {
    /// Creates a new, empty typed arena.
    pub fn new() -> Self {
        TypedArena {
            inner: Arena::new(),
            _marker: PhantomData,
        }
    }

    /// Allocates a value of type `T` in the arena and returns a reference
    /// with the arena's lifetime.
    pub fn alloc(&self, value: T) -> &T {
        self.inner.alloc(value)
    }
}

impl<T> Default for TypedArena<T> {
    fn default() -> Self {
        TypedArena::new()
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Basic allocation tests ---------------------------------------------

    #[test]
    fn test_alloc_single_i32() {
        let arena = Arena::new();
        let val = arena.alloc(42i32);
        assert_eq!(*val, 42);
    }

    #[test]
    fn test_alloc_single_u64() {
        let arena = Arena::new();
        let val = arena.alloc(0xDEAD_BEEF_CAFE_BABEu64);
        assert_eq!(*val, 0xDEAD_BEEF_CAFE_BABE);
    }

    #[test]
    fn test_alloc_multiple_same_type() {
        let arena = Arena::new();
        let a = arena.alloc(1i32);
        let b = arena.alloc(2i32);
        let c = arena.alloc(3i32);
        assert_eq!(*a, 1);
        assert_eq!(*b, 2);
        assert_eq!(*c, 3);
    }

    #[test]
    fn test_alloc_different_types() {
        let arena = Arena::new();
        let x = arena.alloc(10u8);
        let y = arena.alloc(20u64);
        let z = arena.alloc(-5i32);
        assert_eq!(*x, 10u8);
        assert_eq!(*y, 20u64);
        assert_eq!(*z, -5i32);
    }

    // -- Struct allocation tests --------------------------------------------

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct TestNode {
        kind: u32,
        value: i64,
        flags: u8,
    }

    #[test]
    fn test_alloc_struct() {
        let arena = Arena::new();
        let node = arena.alloc(TestNode {
            kind: 7,
            value: -100,
            flags: 0xFF,
        });
        assert_eq!(node.kind, 7);
        assert_eq!(node.value, -100);
        assert_eq!(node.flags, 0xFF);
    }

    #[test]
    fn test_alloc_many_structs() {
        let arena = Arena::new();
        let mut refs = Vec::new();
        for i in 0..1000u32 {
            let node = arena.alloc(TestNode {
                kind: i,
                value: i as i64 * 2,
                flags: (i % 256) as u8,
            });
            refs.push(node);
        }
        // Verify all references are still valid after many allocations.
        for (i, node) in refs.iter().enumerate() {
            let i = i as u32;
            assert_eq!(node.kind, i);
            assert_eq!(node.value, i as i64 * 2);
            assert_eq!(node.flags, (i % 256) as u8);
        }
    }

    // -- Alignment tests ----------------------------------------------------

    #[test]
    fn test_alignment_u8_then_u64() {
        let arena = Arena::new();
        let _byte = arena.alloc(1u8);
        let qword = arena.alloc(0x1234_5678_9ABC_DEF0u64);
        // Verify the u64 is properly aligned.
        let ptr = qword as *const u64;
        assert_eq!(
            ptr as usize % mem::align_of::<u64>(),
            0,
            "u64 pointer is not aligned to {}",
            mem::align_of::<u64>()
        );
        assert_eq!(*qword, 0x1234_5678_9ABC_DEF0);
    }

    #[test]
    fn test_alignment_mixed_types() {
        let arena = Arena::new();
        for _ in 0..100 {
            let _a = arena.alloc(1u8);
            let b = arena.alloc(2u32);
            let c = arena.alloc(3u64);
            let d = arena.alloc(4u16);
            assert_eq!(b as *const u32 as usize % mem::align_of::<u32>(), 0);
            assert_eq!(c as *const u64 as usize % mem::align_of::<u64>(), 0);
            assert_eq!(d as *const u16 as usize % mem::align_of::<u16>(), 0);
        }
    }

    // -- Slice allocation tests ---------------------------------------------

    #[test]
    fn test_alloc_slice_basic() {
        let arena = Arena::new();
        let src = [1u32, 2, 3, 4, 5];
        let slice = arena.alloc_slice(&src);
        assert_eq!(slice, &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_alloc_slice_empty() {
        let arena = Arena::new();
        let empty: &[u32] = &[];
        let slice = arena.alloc_slice(empty);
        assert!(slice.is_empty());
    }

    #[test]
    fn test_alloc_slice_large() {
        let arena = Arena::new();
        let data: Vec<u64> = (0..10_000).collect();
        let slice = arena.alloc_slice(&data);
        assert_eq!(slice.len(), 10_000);
        for (i, &val) in slice.iter().enumerate() {
            assert_eq!(val, i as u64);
        }
    }

    #[test]
    fn test_alloc_slice_u8() {
        let arena = Arena::new();
        let data = [0xFFu8, 0x00, 0xAB, 0xCD];
        let slice = arena.alloc_slice(&data);
        assert_eq!(slice, &[0xFF, 0x00, 0xAB, 0xCD]);
    }

    // -- String allocation tests --------------------------------------------

    #[test]
    fn test_alloc_str_basic() {
        let arena = Arena::new();
        let s = arena.alloc_str("hello, world");
        assert_eq!(s, "hello, world");
    }

    #[test]
    fn test_alloc_str_empty() {
        let arena = Arena::new();
        let s = arena.alloc_str("");
        assert_eq!(s, "");
        assert!(s.is_empty());
    }

    #[test]
    fn test_alloc_str_unicode() {
        let arena = Arena::new();
        let s = arena.alloc_str("日本語テスト 🎉");
        assert_eq!(s, "日本語テスト 🎉");
    }

    #[test]
    fn test_alloc_str_multiple() {
        let arena = Arena::new();
        let a = arena.alloc_str("int");
        let b = arena.alloc_str("main");
        let c = arena.alloc_str("return");
        assert_eq!(a, "int");
        assert_eq!(b, "main");
        assert_eq!(c, "return");
    }

    // -- Constructor and utility tests --------------------------------------

    #[test]
    fn test_new_arena_is_empty() {
        let arena = Arena::new();
        assert_eq!(arena.bytes_allocated(), 0);
    }

    #[test]
    fn test_with_capacity() {
        let arena = Arena::with_capacity(4096);
        // Should have at least 4096 bytes allocated (the initial chunk).
        assert!(arena.bytes_allocated() >= 4096);
    }

    #[test]
    fn test_bytes_allocated_grows() {
        let arena = Arena::new();
        let before = arena.bytes_allocated();
        // Allocate enough to force at least one chunk.
        let _v = arena.alloc(42u64);
        let after = arena.bytes_allocated();
        assert!(
            after > before,
            "bytes_allocated should grow after allocation"
        );
    }

    #[test]
    fn test_bytes_allocated_multiple_chunks() {
        let arena = Arena::with_capacity(64); // Small initial chunk.
        let initial = arena.bytes_allocated();
        // Allocate more than fits in 64 bytes to force a second chunk.
        for i in 0..100u64 {
            let _ = arena.alloc(i);
        }
        let total = arena.bytes_allocated();
        assert!(
            total > initial,
            "bytes_allocated should grow across multiple chunks"
        );
    }

    #[test]
    fn test_reset() {
        let mut arena = Arena::with_capacity(1024);
        // Allocate some values.
        for i in 0..50u32 {
            let _ = arena.alloc(i);
        }
        let alloc_before = arena.bytes_allocated();
        assert!(alloc_before > 0);

        arena.reset();

        // After reset, we should still have one chunk (the first).
        assert!(arena.bytes_allocated() > 0);
        // The arena should now accept new allocations reusing the first chunk.
        let val = arena.alloc(999u32);
        assert_eq!(*val, 999);
    }

    #[test]
    fn test_reset_deallocates_extra_chunks() {
        let mut arena = Arena::with_capacity(64);
        // Force many chunks.
        for i in 0..10_000u64 {
            let _ = arena.alloc(i);
        }
        let before_reset = arena.bytes_allocated();

        arena.reset();

        let after_reset = arena.bytes_allocated();
        // After reset, only the first chunk should remain.
        assert!(
            after_reset < before_reset,
            "reset should deallocate extra chunks: before={}, after={}",
            before_reset,
            after_reset
        );
    }

    // -- Chunk growth / multi-chunk tests -----------------------------------

    #[test]
    fn test_multi_chunk_stress() {
        let arena = Arena::new();
        let count = 100_000;
        let mut refs = Vec::with_capacity(count);
        for i in 0..count {
            refs.push(arena.alloc(i as u64));
        }
        // Verify all references are still valid.
        for (i, val) in refs.iter().enumerate() {
            assert_eq!(**val, i as u64, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_large_allocation_exceeding_default_chunk() {
        let arena = Arena::new();
        // Allocate a slice larger than DEFAULT_CHUNK_SIZE.
        let big: Vec<u8> = vec![0xAB; DEFAULT_CHUNK_SIZE * 2];
        let slice = arena.alloc_slice(&big);
        assert_eq!(slice.len(), DEFAULT_CHUNK_SIZE * 2);
        assert!(slice.iter().all(|&b| b == 0xAB));
    }

    // -- Zero-sized type tests ----------------------------------------------

    #[test]
    fn test_alloc_zst() {
        let arena = Arena::new();
        let _unit = arena.alloc(());
        // ZST allocation should not consume arena memory.
        // (The arena might still have 0 bytes allocated if no chunk was created.)
    }

    #[test]
    fn test_alloc_slice_zst() {
        let arena = Arena::new();
        let src = [(), (), ()];
        let slice = arena.alloc_slice(&src);
        assert_eq!(slice.len(), 3);
    }

    // -- TypedArena tests ---------------------------------------------------

    #[test]
    fn test_typed_arena_basic() {
        let arena: TypedArena<i32> = TypedArena::new();
        let a = arena.alloc(10);
        let b = arena.alloc(20);
        assert_eq!(*a, 10);
        assert_eq!(*b, 20);
    }

    #[test]
    fn test_typed_arena_struct() {
        let arena: TypedArena<TestNode> = TypedArena::new();
        let node = arena.alloc(TestNode {
            kind: 1,
            value: 42,
            flags: 0,
        });
        assert_eq!(node.kind, 1);
        assert_eq!(node.value, 42);
    }

    #[test]
    fn test_typed_arena_many() {
        let arena: TypedArena<u64> = TypedArena::new();
        let mut refs = Vec::new();
        for i in 0..5000u64 {
            refs.push(arena.alloc(i));
        }
        for (i, val) in refs.iter().enumerate() {
            assert_eq!(**val, i as u64);
        }
    }

    // -- Default trait tests ------------------------------------------------

    #[test]
    fn test_arena_default() {
        let arena = Arena::default();
        let val = arena.alloc(7u32);
        assert_eq!(*val, 7);
    }

    #[test]
    fn test_typed_arena_default() {
        let arena = TypedArena::<i32>::default();
        let val = arena.alloc(7);
        assert_eq!(*val, 7);
    }

    // -- alloc_raw direct test ----------------------------------------------

    #[test]
    fn test_alloc_raw_directly() {
        let arena = Arena::new();
        let layout = Layout::from_size_align(128, 8).unwrap();
        let ptr = arena.alloc_raw(layout);
        assert!(!ptr.is_null());
        assert_eq!(ptr as usize % 8, 0, "alloc_raw pointer not aligned");
    }

    // -- Edge case: with_capacity then heavy allocation ---------------------

    #[test]
    fn test_with_capacity_then_exceed() {
        let arena = Arena::with_capacity(32);
        // Allocate well beyond the initial 32 bytes.
        for i in 0..1000u64 {
            let val = arena.alloc(i);
            assert_eq!(*val, i);
        }
    }

    // -- Simulate AST-like allocation pattern --------------------------------

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum ExprKind {
        IntLiteral(i64),
        BinaryAdd,
        BinarySub,
        Identifier,
    }

    #[derive(Debug, Clone, Copy)]
    struct AstNode {
        kind: ExprKind,
        left: Option<usize>, // Index into some external storage
        right: Option<usize>,
        line: u32,
        column: u32,
    }

    #[test]
    fn test_ast_simulation() {
        let arena = Arena::new();
        let mut nodes: Vec<&AstNode> = Vec::new();

        // Simulate building a small AST: 1 + (2 - 3)
        let lit1 = arena.alloc(AstNode {
            kind: ExprKind::IntLiteral(1),
            left: None,
            right: None,
            line: 1,
            column: 1,
        });
        nodes.push(lit1);

        let lit2 = arena.alloc(AstNode {
            kind: ExprKind::IntLiteral(2),
            left: None,
            right: None,
            line: 1,
            column: 5,
        });
        nodes.push(lit2);

        let lit3 = arena.alloc(AstNode {
            kind: ExprKind::IntLiteral(3),
            left: None,
            right: None,
            line: 1,
            column: 9,
        });
        nodes.push(lit3);

        let sub = arena.alloc(AstNode {
            kind: ExprKind::BinarySub,
            left: Some(1),
            right: Some(2),
            line: 1,
            column: 7,
        });
        nodes.push(sub);

        let add = arena.alloc(AstNode {
            kind: ExprKind::BinaryAdd,
            left: Some(0),
            right: Some(3),
            line: 1,
            column: 3,
        });
        nodes.push(add);

        // Verify the AST structure.
        assert_eq!(nodes[0].kind, ExprKind::IntLiteral(1));
        assert_eq!(nodes[4].kind, ExprKind::BinaryAdd);
        assert_eq!(nodes[4].left, Some(0));
        assert_eq!(nodes[4].right, Some(3));
        assert_eq!(nodes[3].kind, ExprKind::BinarySub);
    }

    // -- Memory tracking accuracy -------------------------------------------

    #[test]
    fn test_bytes_allocated_reasonable() {
        let arena = Arena::new();
        // Allocate 100 u64 values = 800 bytes of payload.
        for i in 0..100u64 {
            let _ = arena.alloc(i);
        }
        let total = arena.bytes_allocated();
        // The arena should have allocated at least 800 bytes (probably more
        // due to alignment and chunk overhead).
        assert!(
            total >= 800,
            "Expected at least 800 bytes allocated, got {}",
            total
        );
        // But not absurdly more (sanity check).
        assert!(
            total < 1_000_000,
            "Unexpected amount of memory: {} bytes",
            total
        );
    }
}
