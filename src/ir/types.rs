//! IR-level type system for the `bcc` compiler's intermediate representation.
//!
//! This module defines [`IrType`], the central type representation used throughout
//! the IR layer. It maps C types to simplified IR-level representations, stripping
//! away qualifiers, typedefs, and C-specific semantics to produce types closer to
//! machine-level representations.
//!
//! # Target-Parametric Sizing
//!
//! Type sizes and alignments are parameterized by [`TargetConfig`] to support the
//! four target architectures:
//!
//! | Target         | Pointer Size | `long` Size | Notes            |
//! |----------------|-------------|-------------|------------------|
//! | x86-64         | 8 bytes     | 8 bytes     | LP64 model       |
//! | i686           | 4 bytes     | 4 bytes     | ILP32 model      |
//! | AArch64        | 8 bytes     | 8 bytes     | LP64 model       |
//! | RISC-V 64      | 8 bytes     | 8 bytes     | LP64 model       |
//!
//! # Display Format
//!
//! Types are formatted in an LLVM-IR-inspired textual form:
//! - Scalars: `void`, `i1`, `i8`, `i16`, `i32`, `i64`, `f32`, `f64`
//! - Pointers: `i32*`
//! - Arrays: `[10 x i32]`
//! - Structs: `{ i32, f64 }` or `<{ i32, f64 }>` (packed)
//! - Functions: `i32 (i32, i8*)`
//! - Labels: `label`

use std::fmt;

use crate::driver::target::TargetConfig;

// ---------------------------------------------------------------------------
// IrType enum — central IR type representation
// ---------------------------------------------------------------------------

/// The central type representation for the `bcc` compiler's intermediate
/// representation.
///
/// `IrType` is simpler than the C-level type system (`CType`): it strips away
/// qualifiers (`const`, `volatile`, `restrict`), typedefs, and decay semantics.
/// The resulting types are close to machine types, with target-parametric sizing
/// for pointers and certain integer widths.
///
/// # Variants
///
/// - **Scalar types**: `Void`, `I1`, `I8`, `I16`, `I32`, `I64`, `F32`, `F64`
/// - **Composite types**: `Pointer`, `Array`, `Struct`, `Function`
/// - **Control flow**: `Label` (basic block references)
///
/// # Derive Traits
///
/// - `Clone` — IR types are freely cloneable for instruction construction.
/// - `PartialEq`, `Eq`, `Hash` — Enable use as `HashMap` keys for type-based
///   lookups and deduplication.
/// - `Debug` — Developer-facing debug output.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IrType {
    /// Void type — used for void function returns and void pointers.
    ///
    /// Has zero size. Should not appear as the type of a stored value.
    Void,

    /// 1-bit boolean type — the result of comparison operations (`ICmp`, `FCmp`).
    ///
    /// Stored as a single byte in memory but logically represents a single bit.
    I1,

    /// 8-bit integer type.
    ///
    /// Maps to C `char`, `signed char`, `unsigned char`, and `_Bool` storage.
    I8,

    /// 16-bit integer type.
    ///
    /// Maps to C `short` and `unsigned short`.
    I16,

    /// 32-bit integer type.
    ///
    /// Maps to C `int`, `unsigned int`, and `long`/`unsigned long` on i686 (ILP32).
    I32,

    /// 64-bit integer type.
    ///
    /// Maps to C `long long`, `unsigned long long`, and `long`/`unsigned long`
    /// on LP64 targets (x86-64, AArch64, RISC-V 64).
    I64,

    /// 32-bit IEEE 754 single-precision floating-point type.
    ///
    /// Maps to C `float`.
    F32,

    /// 64-bit IEEE 754 double-precision floating-point type.
    ///
    /// Maps to C `double`.
    F64,

    /// Pointer type with a pointee type.
    ///
    /// Size is target-dependent: 4 bytes on i686, 8 bytes on x86-64/AArch64/RISC-V 64.
    Pointer(Box<IrType>),

    /// Fixed-size array type with an element type and count.
    ///
    /// Size is `element.size(target) * count`. Alignment matches the element alignment.
    Array {
        /// The element type of each array entry.
        element: Box<IrType>,
        /// The number of elements in the array.
        count: usize,
    },

    /// Struct (aggregate) type — an ordered sequence of field types.
    ///
    /// When `packed` is `false`, fields are laid out with natural alignment padding
    /// between them per the target ABI. When `packed` is `true` (corresponding to
    /// `__attribute__((packed))`), no padding is inserted between fields.
    Struct {
        /// The types of each field, in declaration order.
        fields: Vec<IrType>,
        /// If `true`, the struct uses packed layout with no inter-field padding
        /// and alignment of 1.
        packed: bool,
    },

    /// Function type — describes the signature of a function for call type checking
    /// and function pointer representation.
    ///
    /// Has zero size (function values are not directly storable; use `Pointer(Function {...})`)
    /// for function pointers.
    Function {
        /// The return type of the function.
        return_type: Box<IrType>,
        /// The types of the function's fixed parameters.
        param_types: Vec<IrType>,
        /// Whether the function accepts variadic arguments (`...`).
        is_variadic: bool,
    },

    /// Label type — represents basic block references in branch instructions.
    ///
    /// Not a data type; has zero size and is not stored in memory.
    Label,
}

// ---------------------------------------------------------------------------
// StructLayout — result of struct layout computation
// ---------------------------------------------------------------------------

/// The computed memory layout of a struct type.
///
/// Produced by [`compute_struct_layout`] and used by `GetElementPtr` instruction
/// lowering and code generation for struct member access.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructLayout {
    /// Byte offset of each field within the struct.
    ///
    /// `offsets[i]` is the byte offset of the `i`-th field from the start of
    /// the struct. Length matches the number of fields.
    pub offsets: Vec<usize>,

    /// Total size of the struct in bytes, including trailing padding to satisfy
    /// the struct's alignment requirement.
    pub size: usize,

    /// Alignment requirement of the struct in bytes.
    ///
    /// For unpacked structs, this is the maximum alignment among all field types.
    /// For packed structs, this is always 1.
    pub alignment: usize,
}

// ---------------------------------------------------------------------------
// Helper: alignment rounding
// ---------------------------------------------------------------------------

/// Rounds `offset` up to the next multiple of `alignment`.
///
/// Returns `offset` unchanged if it is already aligned. `alignment` must be
/// a non-zero power of two.
#[inline]
fn align_to(offset: usize, alignment: usize) -> usize {
    debug_assert!(alignment > 0, "alignment must be non-zero");
    debug_assert!(
        alignment.is_power_of_two(),
        "alignment must be a power of two, got {}",
        alignment
    );
    (offset + alignment - 1) & !(alignment - 1)
}

// ---------------------------------------------------------------------------
// IrType — size and alignment computation
// ---------------------------------------------------------------------------

impl IrType {
    /// Returns the size of this IR type in bytes for the given target.
    ///
    /// # Target Dependence
    ///
    /// - `Pointer` size depends on `target.pointer_size` (4 on i686, 8 on 64-bit targets).
    /// - `Array` and `Struct` sizes are recursively computed from element/field types.
    ///
    /// # Special Cases
    ///
    /// - `Void`, `Function`, and `Label` have zero size (they are not data types).
    /// - `I1` occupies 1 byte in memory despite being logically 1 bit.
    pub fn size(&self, target: &TargetConfig) -> usize {
        match self {
            IrType::Void => 0,
            IrType::I1 => 1,
            IrType::I8 => 1,
            IrType::I16 => 2,
            IrType::I32 => 4,
            IrType::I64 => 8,
            IrType::F32 => 4,
            IrType::F64 => 8,
            IrType::Pointer(_) => target.pointer_size as usize,
            IrType::Array { element, count } => element.size(target) * count,
            IrType::Struct { fields, packed } => {
                if fields.is_empty() {
                    return 0;
                }
                let layout = compute_struct_layout(fields, *packed, target);
                layout.size
            }
            IrType::Function { .. } => 0,
            IrType::Label => 0,
        }
    }

    /// Returns the alignment requirement of this IR type in bytes for the given target.
    ///
    /// # Target Dependence
    ///
    /// - `Pointer` alignment matches `target.pointer_size`.
    /// - `Array` alignment matches its element alignment.
    /// - Unpacked `Struct` alignment is the maximum of all field alignments.
    /// - Packed `Struct` alignment is always 1.
    ///
    /// # Special Cases
    ///
    /// - `Void`, `Function`, and `Label` default to alignment 1.
    pub fn alignment(&self, target: &TargetConfig) -> usize {
        match self {
            IrType::Void => 1,
            IrType::I1 | IrType::I8 => 1,
            IrType::I16 => 2,
            IrType::I32 | IrType::F32 => 4,
            IrType::I64 | IrType::F64 => 8,
            IrType::Pointer(_) => target.pointer_size as usize,
            IrType::Array { element, .. } => element.alignment(target),
            IrType::Struct { fields, packed } => {
                if *packed {
                    return 1;
                }
                fields
                    .iter()
                    .map(|f| f.alignment(target))
                    .max()
                    .unwrap_or(1)
            }
            IrType::Function { .. } => 1,
            IrType::Label => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// IrType — type classification methods
// ---------------------------------------------------------------------------

impl IrType {
    /// Returns `true` if this is an integer type (`I1`, `I8`, `I16`, `I32`, `I64`).
    #[inline]
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            IrType::I1 | IrType::I8 | IrType::I16 | IrType::I32 | IrType::I64
        )
    }

    /// Returns `true` if this is a floating-point type (`F32`, `F64`).
    #[inline]
    pub fn is_float(&self) -> bool {
        matches!(self, IrType::F32 | IrType::F64)
    }

    /// Returns `true` if this is a numeric type (integer or floating-point).
    #[inline]
    pub fn is_numeric(&self) -> bool {
        self.is_integer() || self.is_float()
    }

    /// Returns `true` if this is a pointer type.
    #[inline]
    pub fn is_pointer(&self) -> bool {
        matches!(self, IrType::Pointer(_))
    }

    /// Returns `true` if this is the `Void` type.
    #[inline]
    pub fn is_void(&self) -> bool {
        matches!(self, IrType::Void)
    }

    /// Returns `true` if this is an array type.
    #[inline]
    pub fn is_array(&self) -> bool {
        matches!(self, IrType::Array { .. })
    }

    /// Returns `true` if this is a struct type.
    #[inline]
    pub fn is_struct(&self) -> bool {
        matches!(self, IrType::Struct { .. })
    }

    /// Returns `true` if this is a function type.
    #[inline]
    pub fn is_function(&self) -> bool {
        matches!(self, IrType::Function { .. })
    }

    /// Returns `true` if this is an aggregate type (struct or array).
    ///
    /// Aggregate types have composite memory layout and cannot be stored
    /// in a single register.
    #[inline]
    pub fn is_aggregate(&self) -> bool {
        self.is_struct() || self.is_array()
    }

    /// Returns `true` if this is a scalar type (integer, floating-point, or pointer).
    ///
    /// Scalar types can be stored in a single register and participate directly
    /// in arithmetic and comparison operations.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.is_integer() || self.is_float() || self.is_pointer()
    }
}

// ---------------------------------------------------------------------------
// IrType — utility / accessor methods
// ---------------------------------------------------------------------------

impl IrType {
    /// Returns the pointee type if this is a `Pointer`, otherwise `None`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let ptr = IrType::Pointer(Box::new(IrType::I32));
    /// assert_eq!(ptr.pointee(), Some(&IrType::I32));
    /// assert_eq!(IrType::I32.pointee(), None);
    /// ```
    pub fn pointee(&self) -> Option<&IrType> {
        match self {
            IrType::Pointer(ty) => Some(ty),
            _ => None,
        }
    }

    /// Returns the element type if this is an `Array`, otherwise `None`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let arr = IrType::Array { element: Box::new(IrType::I32), count: 10 };
    /// assert_eq!(arr.element_type(), Some(&IrType::I32));
    /// ```
    pub fn element_type(&self) -> Option<&IrType> {
        match self {
            IrType::Array { element, .. } => Some(element),
            _ => None,
        }
    }

    /// Returns the return type if this is a `Function`, otherwise `None`.
    pub fn return_type(&self) -> Option<&IrType> {
        match self {
            IrType::Function { return_type, .. } => Some(return_type),
            _ => None,
        }
    }

    /// Returns the bit width of this type if it is an integer, otherwise `None`.
    ///
    /// # Mapping
    ///
    /// | Type | Bit Width |
    /// |------|-----------|
    /// | `I1` | 1         |
    /// | `I8` | 8         |
    /// | `I16`| 16        |
    /// | `I32`| 32        |
    /// | `I64`| 64        |
    pub fn integer_bit_width(&self) -> Option<u32> {
        match self {
            IrType::I1 => Some(1),
            IrType::I8 => Some(8),
            IrType::I16 => Some(16),
            IrType::I32 => Some(32),
            IrType::I64 => Some(64),
            _ => None,
        }
    }

    /// Wraps this type in a `Pointer`, consuming `self`.
    ///
    /// This is a convenience method equivalent to `IrType::Pointer(Box::new(self))`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let ptr = IrType::I32.pointer_to();
    /// assert_eq!(ptr, IrType::Pointer(Box::new(IrType::I32)));
    /// ```
    pub fn pointer_to(self) -> IrType {
        IrType::Pointer(Box::new(self))
    }
}

// ---------------------------------------------------------------------------
// IrType — target-parametric integer selection
// ---------------------------------------------------------------------------

impl IrType {
    /// Returns the integer IR type whose byte size matches `bytes`.
    ///
    /// This is used by the IR builder when mapping C types whose width depends
    /// on the target (e.g., `long` is 4 bytes on i686 but 8 bytes on x86-64).
    ///
    /// # Panics
    ///
    /// Panics if `bytes` is not one of 1, 2, 4, or 8.
    pub fn int_type_for_size(bytes: usize) -> IrType {
        match bytes {
            1 => IrType::I8,
            2 => IrType::I16,
            4 => IrType::I32,
            8 => IrType::I64,
            _ => panic!(
                "no integer IR type with size {} bytes; expected 1, 2, 4, or 8",
                bytes
            ),
        }
    }

    /// Returns the integer IR type whose size matches the target's pointer width.
    ///
    /// This is used for lowering `size_t`, `ptrdiff_t`, and `uintptr_t` to IR types.
    ///
    /// # Returns
    ///
    /// - `I32` on i686 (pointer_size = 4)
    /// - `I64` on x86-64, AArch64, RISC-V 64 (pointer_size = 8)
    pub fn pointer_sized_int(target: &TargetConfig) -> IrType {
        IrType::int_type_for_size(target.pointer_size as usize)
    }
}

// ---------------------------------------------------------------------------
// compute_struct_layout — free function
// ---------------------------------------------------------------------------

/// Computes the memory layout of a struct with the given field types.
///
/// This function calculates the byte offset of each field, the total size
/// (including trailing padding), and the overall alignment of the struct.
///
/// # Packed vs. Unpacked Layout
///
/// - **Unpacked** (`packed = false`): Each field is aligned to its natural
///   alignment. Padding bytes are inserted between fields as needed. The struct's
///   total size is rounded up to a multiple of the struct's alignment (which is
///   the maximum of all field alignments).
///
/// - **Packed** (`packed = true`): Fields are placed at consecutive byte offsets
///   with no padding. The struct's alignment is 1.
///
/// # Parameters
///
/// - `fields` — Slice of field types in declaration order.
/// - `packed` — Whether the struct uses packed layout.
/// - `target` — Target configuration providing pointer size and type alignments.
///
/// # Returns
///
/// A [`StructLayout`] containing field offsets, total size, and alignment.
///
/// # Examples
///
/// ```ignore
/// // struct { int a; char b; int c; } on x86-64
/// let target = TargetConfig::x86_64();
/// let layout = compute_struct_layout(
///     &[IrType::I32, IrType::I8, IrType::I32],
///     false,
///     &target,
/// );
/// assert_eq!(layout.offsets, vec![0, 4, 8]);
/// assert_eq!(layout.size, 12);
/// assert_eq!(layout.alignment, 4);
/// ```
pub fn compute_struct_layout(
    fields: &[IrType],
    packed: bool,
    target: &TargetConfig,
) -> StructLayout {
    let mut offsets = Vec::with_capacity(fields.len());
    let mut current_offset: usize = 0;
    let mut max_alignment: usize = 1;

    for field in fields {
        let field_alignment = if packed { 1 } else { field.alignment(target) };
        let field_size = field.size(target);

        // Align the current offset to the field's required alignment.
        current_offset = align_to(current_offset, field_alignment);
        offsets.push(current_offset);

        // Advance past this field.
        current_offset += field_size;

        // Track the maximum field alignment for the overall struct alignment.
        if field_alignment > max_alignment {
            max_alignment = field_alignment;
        }
    }

    // The struct's overall alignment is 1 if packed, else the max field alignment.
    let struct_alignment = if packed { 1 } else { max_alignment };

    // Pad the total size to a multiple of the struct alignment.
    let total_size = align_to(current_offset, struct_alignment);

    StructLayout {
        offsets,
        size: total_size,
        alignment: struct_alignment,
    }
}

// ---------------------------------------------------------------------------
// Display implementation for IrType
// ---------------------------------------------------------------------------

impl fmt::Display for IrType {
    /// Formats the IR type in a human-readable textual representation inspired
    /// by LLVM IR syntax.
    ///
    /// # Format
    ///
    /// | Type               | Output             |
    /// |--------------------|--------------------|
    /// | `Void`             | `void`             |
    /// | `I1`               | `i1`               |
    /// | `I8`               | `i8`               |
    /// | `I16`              | `i16`              |
    /// | `I32`              | `i32`              |
    /// | `I64`              | `i64`              |
    /// | `F32`              | `f32`              |
    /// | `F64`              | `f64`              |
    /// | `Pointer(I32)`     | `i32*`             |
    /// | `Array(I32, 10)`   | `[10 x i32]`       |
    /// | `Struct{I32, F64}` | `{ i32, f64 }`     |
    /// | `Struct{..packed}` | `<{ i32, f64 }>`   |
    /// | `Function`         | `i32 (i32, i8*)`   |
    /// | `Label`            | `label`            |
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrType::Void => write!(f, "void"),
            IrType::I1 => write!(f, "i1"),
            IrType::I8 => write!(f, "i8"),
            IrType::I16 => write!(f, "i16"),
            IrType::I32 => write!(f, "i32"),
            IrType::I64 => write!(f, "i64"),
            IrType::F32 => write!(f, "f32"),
            IrType::F64 => write!(f, "f64"),
            IrType::Pointer(pointee) => write!(f, "{}*", pointee),
            IrType::Array { element, count } => write!(f, "[{} x {}]", count, element),
            IrType::Struct { fields, packed } => {
                if *packed {
                    write!(f, "<")?;
                }
                write!(f, "{{ ")?;
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", field)?;
                }
                write!(f, " }}")?;
                if *packed {
                    write!(f, ">")?;
                }
                Ok(())
            }
            IrType::Function {
                return_type,
                param_types,
                is_variadic,
            } => {
                write!(f, "{} (", return_type)?;
                for (i, param) in param_types.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", param)?;
                }
                if *is_variadic {
                    if !param_types.is_empty() {
                        write!(f, ", ")?;
                    }
                    write!(f, "...")?;
                }
                write!(f, ")")
            }
            IrType::Label => write!(f, "label"),
        }
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Test helpers — create target configurations for testing
    // -----------------------------------------------------------------------

    /// Returns a mock TargetConfig for x86-64 (pointer_size = 8).
    fn target_x86_64() -> TargetConfig {
        TargetConfig::x86_64()
    }

    /// Returns a mock TargetConfig for i686 (pointer_size = 4).
    fn target_i686() -> TargetConfig {
        TargetConfig::i686()
    }

    // -----------------------------------------------------------------------
    // Size tests — x86-64 (pointer_size = 8)
    // -----------------------------------------------------------------------

    #[test]
    fn test_size_scalars_x86_64() {
        let t = target_x86_64();
        assert_eq!(IrType::I1.size(&t), 1);
        assert_eq!(IrType::I8.size(&t), 1);
        assert_eq!(IrType::I16.size(&t), 2);
        assert_eq!(IrType::I32.size(&t), 4);
        assert_eq!(IrType::I64.size(&t), 8);
        assert_eq!(IrType::F32.size(&t), 4);
        assert_eq!(IrType::F64.size(&t), 8);
    }

    #[test]
    fn test_size_void_function_label() {
        let t = target_x86_64();
        assert_eq!(IrType::Void.size(&t), 0);
        assert_eq!(IrType::Label.size(&t), 0);
        let func_ty = IrType::Function {
            return_type: Box::new(IrType::I32),
            param_types: vec![IrType::I32],
            is_variadic: false,
        };
        assert_eq!(func_ty.size(&t), 0);
    }

    #[test]
    fn test_size_pointer_x86_64() {
        let t = target_x86_64();
        let ptr = IrType::Pointer(Box::new(IrType::I32));
        assert_eq!(ptr.size(&t), 8);
    }

    #[test]
    fn test_size_array_x86_64() {
        let t = target_x86_64();
        let arr = IrType::Array {
            element: Box::new(IrType::I32),
            count: 10,
        };
        assert_eq!(arr.size(&t), 40);
    }

    #[test]
    fn test_size_empty_array() {
        let t = target_x86_64();
        let arr = IrType::Array {
            element: Box::new(IrType::I32),
            count: 0,
        };
        assert_eq!(arr.size(&t), 0);
    }

    #[test]
    fn test_size_struct_unpacked_x86_64() {
        let t = target_x86_64();
        // struct { int a; char b; int c; }
        // offsets: [0, 4, 8], size: 12
        let s = IrType::Struct {
            fields: vec![IrType::I32, IrType::I8, IrType::I32],
            packed: false,
        };
        assert_eq!(s.size(&t), 12);
    }

    #[test]
    fn test_size_struct_packed_x86_64() {
        let t = target_x86_64();
        // packed struct { int a; char b; int c; }
        // offsets: [0, 4, 5], size: 9
        let s = IrType::Struct {
            fields: vec![IrType::I32, IrType::I8, IrType::I32],
            packed: true,
        };
        assert_eq!(s.size(&t), 9);
    }

    #[test]
    fn test_size_struct_with_i64_alignment() {
        let t = target_x86_64();
        // struct { char a; long long b; }
        // offsets: [0, 8], size: 16
        let s = IrType::Struct {
            fields: vec![IrType::I8, IrType::I64],
            packed: false,
        };
        assert_eq!(s.size(&t), 16);
    }

    #[test]
    fn test_size_empty_struct() {
        let t = target_x86_64();
        let s = IrType::Struct {
            fields: vec![],
            packed: false,
        };
        assert_eq!(s.size(&t), 0);
    }

    // -----------------------------------------------------------------------
    // Size tests — i686 (pointer_size = 4)
    // -----------------------------------------------------------------------

    #[test]
    fn test_size_pointer_i686() {
        let t = target_i686();
        let ptr = IrType::Pointer(Box::new(IrType::I32));
        assert_eq!(ptr.size(&t), 4);
    }

    #[test]
    fn test_size_scalars_same_on_i686() {
        let t = target_i686();
        // Scalar integer/float sizes don't change with target.
        assert_eq!(IrType::I32.size(&t), 4);
        assert_eq!(IrType::I64.size(&t), 8);
        assert_eq!(IrType::F64.size(&t), 8);
    }

    // -----------------------------------------------------------------------
    // Alignment tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_alignment_scalars() {
        let t = target_x86_64();
        assert_eq!(IrType::I1.alignment(&t), 1);
        assert_eq!(IrType::I8.alignment(&t), 1);
        assert_eq!(IrType::I16.alignment(&t), 2);
        assert_eq!(IrType::I32.alignment(&t), 4);
        assert_eq!(IrType::I64.alignment(&t), 8);
        assert_eq!(IrType::F32.alignment(&t), 4);
        assert_eq!(IrType::F64.alignment(&t), 8);
    }

    #[test]
    fn test_alignment_pointer_x86_64() {
        let t = target_x86_64();
        let ptr = IrType::Pointer(Box::new(IrType::I8));
        assert_eq!(ptr.alignment(&t), 8);
    }

    #[test]
    fn test_alignment_pointer_i686() {
        let t = target_i686();
        let ptr = IrType::Pointer(Box::new(IrType::I8));
        assert_eq!(ptr.alignment(&t), 4);
    }

    #[test]
    fn test_alignment_array() {
        let t = target_x86_64();
        let arr = IrType::Array {
            element: Box::new(IrType::I32),
            count: 5,
        };
        assert_eq!(arr.alignment(&t), 4);
    }

    #[test]
    fn test_alignment_struct_unpacked() {
        let t = target_x86_64();
        // struct { int, long long } → alignment = max(4, 8) = 8
        let s = IrType::Struct {
            fields: vec![IrType::I32, IrType::I64],
            packed: false,
        };
        assert_eq!(s.alignment(&t), 8);
    }

    #[test]
    fn test_alignment_struct_packed() {
        let t = target_x86_64();
        let s = IrType::Struct {
            fields: vec![IrType::I32, IrType::I64],
            packed: true,
        };
        assert_eq!(s.alignment(&t), 1);
    }

    #[test]
    fn test_alignment_void_function_label() {
        let t = target_x86_64();
        assert_eq!(IrType::Void.alignment(&t), 1);
        assert_eq!(IrType::Label.alignment(&t), 1);
        let func_ty = IrType::Function {
            return_type: Box::new(IrType::Void),
            param_types: vec![],
            is_variadic: false,
        };
        assert_eq!(func_ty.alignment(&t), 1);
    }

    // -----------------------------------------------------------------------
    // Classification tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_integer() {
        assert!(IrType::I1.is_integer());
        assert!(IrType::I8.is_integer());
        assert!(IrType::I16.is_integer());
        assert!(IrType::I32.is_integer());
        assert!(IrType::I64.is_integer());
        assert!(!IrType::F32.is_integer());
        assert!(!IrType::F64.is_integer());
        assert!(!IrType::Void.is_integer());
        assert!(!IrType::Pointer(Box::new(IrType::I32)).is_integer());
    }

    #[test]
    fn test_is_float() {
        assert!(IrType::F32.is_float());
        assert!(IrType::F64.is_float());
        assert!(!IrType::I32.is_float());
        assert!(!IrType::Void.is_float());
    }

    #[test]
    fn test_is_numeric() {
        assert!(IrType::I32.is_numeric());
        assert!(IrType::F64.is_numeric());
        assert!(!IrType::Void.is_numeric());
        assert!(!IrType::Pointer(Box::new(IrType::I8)).is_numeric());
    }

    #[test]
    fn test_is_pointer() {
        assert!(IrType::Pointer(Box::new(IrType::I32)).is_pointer());
        assert!(!IrType::I32.is_pointer());
    }

    #[test]
    fn test_is_void() {
        assert!(IrType::Void.is_void());
        assert!(!IrType::I32.is_void());
    }

    #[test]
    fn test_is_array() {
        let arr = IrType::Array {
            element: Box::new(IrType::I32),
            count: 5,
        };
        assert!(arr.is_array());
        assert!(!IrType::I32.is_array());
    }

    #[test]
    fn test_is_struct() {
        let s = IrType::Struct {
            fields: vec![IrType::I32],
            packed: false,
        };
        assert!(s.is_struct());
        assert!(!IrType::I32.is_struct());
    }

    #[test]
    fn test_is_function() {
        let f = IrType::Function {
            return_type: Box::new(IrType::I32),
            param_types: vec![],
            is_variadic: false,
        };
        assert!(f.is_function());
        assert!(!IrType::I32.is_function());
    }

    #[test]
    fn test_is_aggregate() {
        let arr = IrType::Array {
            element: Box::new(IrType::I32),
            count: 5,
        };
        let s = IrType::Struct {
            fields: vec![IrType::I32],
            packed: false,
        };
        assert!(arr.is_aggregate());
        assert!(s.is_aggregate());
        assert!(!IrType::I32.is_aggregate());
        assert!(!IrType::Pointer(Box::new(IrType::I32)).is_aggregate());
    }

    #[test]
    fn test_is_scalar() {
        assert!(IrType::I32.is_scalar());
        assert!(IrType::F64.is_scalar());
        assert!(IrType::Pointer(Box::new(IrType::I8)).is_scalar());
        assert!(!IrType::Void.is_scalar());
        let s = IrType::Struct {
            fields: vec![IrType::I32],
            packed: false,
        };
        assert!(!s.is_scalar());
        let arr = IrType::Array {
            element: Box::new(IrType::I32),
            count: 1,
        };
        assert!(!arr.is_scalar());
    }

    // -----------------------------------------------------------------------
    // Utility / accessor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pointee() {
        let ptr = IrType::Pointer(Box::new(IrType::I32));
        assert_eq!(ptr.pointee(), Some(&IrType::I32));
        assert_eq!(IrType::I32.pointee(), None);
    }

    #[test]
    fn test_element_type() {
        let arr = IrType::Array {
            element: Box::new(IrType::I32),
            count: 5,
        };
        assert_eq!(arr.element_type(), Some(&IrType::I32));
        assert_eq!(IrType::I32.element_type(), None);
    }

    #[test]
    fn test_return_type() {
        let f = IrType::Function {
            return_type: Box::new(IrType::I32),
            param_types: vec![IrType::I8],
            is_variadic: false,
        };
        assert_eq!(f.return_type(), Some(&IrType::I32));
        assert_eq!(IrType::I32.return_type(), None);
    }

    #[test]
    fn test_integer_bit_width() {
        assert_eq!(IrType::I1.integer_bit_width(), Some(1));
        assert_eq!(IrType::I8.integer_bit_width(), Some(8));
        assert_eq!(IrType::I16.integer_bit_width(), Some(16));
        assert_eq!(IrType::I32.integer_bit_width(), Some(32));
        assert_eq!(IrType::I64.integer_bit_width(), Some(64));
        assert_eq!(IrType::F32.integer_bit_width(), None);
        assert_eq!(IrType::Void.integer_bit_width(), None);
    }

    #[test]
    fn test_pointer_to() {
        let ptr = IrType::I32.pointer_to();
        assert_eq!(ptr, IrType::Pointer(Box::new(IrType::I32)));
    }

    #[test]
    fn test_pointer_to_chained() {
        // i32** (pointer to pointer to i32)
        let ptr_ptr = IrType::I32.pointer_to().pointer_to();
        assert_eq!(
            ptr_ptr,
            IrType::Pointer(Box::new(IrType::Pointer(Box::new(IrType::I32))))
        );
    }

    // -----------------------------------------------------------------------
    // Target-parametric integer selection tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_int_type_for_size() {
        assert_eq!(IrType::int_type_for_size(1), IrType::I8);
        assert_eq!(IrType::int_type_for_size(2), IrType::I16);
        assert_eq!(IrType::int_type_for_size(4), IrType::I32);
        assert_eq!(IrType::int_type_for_size(8), IrType::I64);
    }

    #[test]
    #[should_panic(expected = "no integer IR type with size 3 bytes")]
    fn test_int_type_for_size_invalid() {
        IrType::int_type_for_size(3);
    }

    #[test]
    fn test_pointer_sized_int_x86_64() {
        let t = target_x86_64();
        assert_eq!(IrType::pointer_sized_int(&t), IrType::I64);
    }

    #[test]
    fn test_pointer_sized_int_i686() {
        let t = target_i686();
        assert_eq!(IrType::pointer_sized_int(&t), IrType::I32);
    }

    // -----------------------------------------------------------------------
    // Struct layout tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_struct_layout_i32_i8_i32() {
        let t = target_x86_64();
        let layout = compute_struct_layout(
            &[IrType::I32, IrType::I8, IrType::I32],
            false,
            &t,
        );
        assert_eq!(layout.offsets, vec![0, 4, 8]);
        assert_eq!(layout.size, 12);
        assert_eq!(layout.alignment, 4);
    }

    #[test]
    fn test_struct_layout_i8_i64() {
        let t = target_x86_64();
        let layout = compute_struct_layout(&[IrType::I8, IrType::I64], false, &t);
        assert_eq!(layout.offsets, vec![0, 8]);
        assert_eq!(layout.size, 16);
        assert_eq!(layout.alignment, 8);
    }

    #[test]
    fn test_struct_layout_packed() {
        let t = target_x86_64();
        let layout = compute_struct_layout(&[IrType::I32, IrType::I8], true, &t);
        assert_eq!(layout.offsets, vec![0, 4]);
        assert_eq!(layout.size, 5);
        assert_eq!(layout.alignment, 1);
    }

    #[test]
    fn test_struct_layout_empty() {
        let t = target_x86_64();
        let layout = compute_struct_layout(&[], false, &t);
        assert_eq!(layout.offsets, Vec::<usize>::new());
        assert_eq!(layout.size, 0);
        assert_eq!(layout.alignment, 1);
    }

    #[test]
    fn test_struct_layout_single_field() {
        let t = target_x86_64();
        let layout = compute_struct_layout(&[IrType::I64], false, &t);
        assert_eq!(layout.offsets, vec![0]);
        assert_eq!(layout.size, 8);
        assert_eq!(layout.alignment, 8);
    }

    #[test]
    fn test_struct_layout_trailing_padding() {
        let t = target_x86_64();
        // struct { long long a; char b; } → alignment 8, so trailing padding to 16
        let layout = compute_struct_layout(&[IrType::I64, IrType::I8], false, &t);
        assert_eq!(layout.offsets, vec![0, 8]);
        assert_eq!(layout.size, 16);
        assert_eq!(layout.alignment, 8);
    }

    #[test]
    fn test_struct_layout_nested_struct() {
        let t = target_x86_64();
        // inner struct { i32, i8 } → size 8 (with trailing padding to align 4), alignment 4
        let inner = IrType::Struct {
            fields: vec![IrType::I32, IrType::I8],
            packed: false,
        };
        // outer struct { i8, inner } → offsets [0, 4], size 12 (inner size is 8), alignment 4
        let layout = compute_struct_layout(&[IrType::I8, inner], false, &t);
        assert_eq!(layout.offsets, vec![0, 4]);
        assert_eq!(layout.size, 12);
        assert_eq!(layout.alignment, 4);
    }

    #[test]
    fn test_struct_layout_with_pointer_i686() {
        let t = target_i686();
        // struct { i32, ptr, i8 } on i686 → pointer is 4 bytes, alignment 4
        let ptr_ty = IrType::Pointer(Box::new(IrType::I8));
        let layout = compute_struct_layout(
            &[IrType::I32, ptr_ty, IrType::I8],
            false,
            &t,
        );
        assert_eq!(layout.offsets, vec![0, 4, 8]);
        assert_eq!(layout.size, 12);
        assert_eq!(layout.alignment, 4);
    }

    #[test]
    fn test_struct_layout_packed_complex() {
        let t = target_x86_64();
        // packed struct { i8, i64, i8 } → no padding: offsets [0, 1, 9], size 10
        let layout = compute_struct_layout(
            &[IrType::I8, IrType::I64, IrType::I8],
            true,
            &t,
        );
        assert_eq!(layout.offsets, vec![0, 1, 9]);
        assert_eq!(layout.size, 10);
        assert_eq!(layout.alignment, 1);
    }

    // -----------------------------------------------------------------------
    // Display formatting tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_display_scalars() {
        assert_eq!(format!("{}", IrType::Void), "void");
        assert_eq!(format!("{}", IrType::I1), "i1");
        assert_eq!(format!("{}", IrType::I8), "i8");
        assert_eq!(format!("{}", IrType::I16), "i16");
        assert_eq!(format!("{}", IrType::I32), "i32");
        assert_eq!(format!("{}", IrType::I64), "i64");
        assert_eq!(format!("{}", IrType::F32), "f32");
        assert_eq!(format!("{}", IrType::F64), "f64");
        assert_eq!(format!("{}", IrType::Label), "label");
    }

    #[test]
    fn test_display_pointer() {
        let ptr = IrType::Pointer(Box::new(IrType::I8));
        assert_eq!(format!("{}", ptr), "i8*");
    }

    #[test]
    fn test_display_pointer_to_pointer() {
        let ptr = IrType::Pointer(Box::new(IrType::Pointer(Box::new(IrType::I32))));
        assert_eq!(format!("{}", ptr), "i32**");
    }

    #[test]
    fn test_display_array() {
        let arr = IrType::Array {
            element: Box::new(IrType::I32),
            count: 10,
        };
        assert_eq!(format!("{}", arr), "[10 x i32]");
    }

    #[test]
    fn test_display_struct() {
        let s = IrType::Struct {
            fields: vec![IrType::I32, IrType::F64],
            packed: false,
        };
        assert_eq!(format!("{}", s), "{ i32, f64 }");
    }

    #[test]
    fn test_display_struct_packed() {
        let s = IrType::Struct {
            fields: vec![IrType::I32, IrType::I8],
            packed: true,
        };
        assert_eq!(format!("{}", s), "<{ i32, i8 }>");
    }

    #[test]
    fn test_display_struct_single_field() {
        let s = IrType::Struct {
            fields: vec![IrType::I32],
            packed: false,
        };
        assert_eq!(format!("{}", s), "{ i32 }");
    }

    #[test]
    fn test_display_function() {
        let f = IrType::Function {
            return_type: Box::new(IrType::I32),
            param_types: vec![
                IrType::I32,
                IrType::Pointer(Box::new(IrType::I8)),
            ],
            is_variadic: false,
        };
        assert_eq!(format!("{}", f), "i32 (i32, i8*)");
    }

    #[test]
    fn test_display_function_variadic() {
        let f = IrType::Function {
            return_type: Box::new(IrType::I32),
            param_types: vec![IrType::Pointer(Box::new(IrType::I8))],
            is_variadic: true,
        };
        assert_eq!(format!("{}", f), "i32 (i8*, ...)");
    }

    #[test]
    fn test_display_function_void_no_params() {
        let f = IrType::Function {
            return_type: Box::new(IrType::Void),
            param_types: vec![],
            is_variadic: false,
        };
        assert_eq!(format!("{}", f), "void ()");
    }

    #[test]
    fn test_display_function_variadic_no_fixed() {
        let f = IrType::Function {
            return_type: Box::new(IrType::Void),
            param_types: vec![],
            is_variadic: true,
        };
        assert_eq!(format!("{}", f), "void (...)");
    }

    #[test]
    fn test_display_complex_nested() {
        // [5 x { i32, i8* }]
        let s = IrType::Struct {
            fields: vec![IrType::I32, IrType::Pointer(Box::new(IrType::I8))],
            packed: false,
        };
        let arr = IrType::Array {
            element: Box::new(s),
            count: 5,
        };
        assert_eq!(format!("{}", arr), "[5 x { i32, i8* }]");
    }

    // -----------------------------------------------------------------------
    // Equality and Hash tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_equality_same_types() {
        assert_eq!(IrType::I32, IrType::I32);
        assert_eq!(IrType::Void, IrType::Void);
        assert_eq!(
            IrType::Pointer(Box::new(IrType::I32)),
            IrType::Pointer(Box::new(IrType::I32))
        );
    }

    #[test]
    fn test_inequality_different_types() {
        assert_ne!(IrType::I32, IrType::I64);
        assert_ne!(IrType::F32, IrType::F64);
        assert_ne!(IrType::I32, IrType::F32);
        assert_ne!(
            IrType::Pointer(Box::new(IrType::I32)),
            IrType::Pointer(Box::new(IrType::I64))
        );
    }

    #[test]
    fn test_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn compute_hash(ty: &IrType) -> u64 {
            let mut hasher = DefaultHasher::new();
            ty.hash(&mut hasher);
            hasher.finish()
        }

        // Equal types must produce the same hash.
        assert_eq!(compute_hash(&IrType::I32), compute_hash(&IrType::I32));
        assert_eq!(
            compute_hash(&IrType::Pointer(Box::new(IrType::I8))),
            compute_hash(&IrType::Pointer(Box::new(IrType::I8)))
        );
    }

    #[test]
    fn test_clone_preserves_equality() {
        let original = IrType::Struct {
            fields: vec![IrType::I32, IrType::I64],
            packed: false,
        };
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    // -----------------------------------------------------------------------
    // IrType can be used as HashMap key
    // -----------------------------------------------------------------------

    #[test]
    fn test_hashmap_key() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(IrType::I32, "int");
        map.insert(IrType::F64, "double");
        map.insert(IrType::Pointer(Box::new(IrType::I8)), "char*");

        assert_eq!(map.get(&IrType::I32), Some(&"int"));
        assert_eq!(map.get(&IrType::F64), Some(&"double"));
        assert_eq!(
            map.get(&IrType::Pointer(Box::new(IrType::I8))),
            Some(&"char*")
        );
        assert_eq!(map.get(&IrType::I64), None);
    }

    // -----------------------------------------------------------------------
    // align_to helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_align_to_already_aligned() {
        assert_eq!(align_to(0, 4), 0);
        assert_eq!(align_to(4, 4), 4);
        assert_eq!(align_to(8, 4), 8);
        assert_eq!(align_to(16, 8), 16);
    }

    #[test]
    fn test_align_to_needs_padding() {
        assert_eq!(align_to(1, 4), 4);
        assert_eq!(align_to(5, 4), 8);
        assert_eq!(align_to(1, 8), 8);
        assert_eq!(align_to(9, 8), 16);
        assert_eq!(align_to(3, 2), 4);
    }

    #[test]
    fn test_align_to_alignment_one() {
        // Alignment of 1 never adds padding.
        assert_eq!(align_to(0, 1), 0);
        assert_eq!(align_to(1, 1), 1);
        assert_eq!(align_to(7, 1), 7);
    }

    // -----------------------------------------------------------------------
    // Edge case: nested pointer types
    // -----------------------------------------------------------------------

    #[test]
    fn test_nested_pointer_pointee() {
        let inner = IrType::Pointer(Box::new(IrType::I32));
        let outer = IrType::Pointer(Box::new(inner.clone()));
        assert_eq!(outer.pointee(), Some(&inner));
        assert_eq!(outer.pointee().unwrap().pointee(), Some(&IrType::I32));
    }

    // -----------------------------------------------------------------------
    // Edge case: array of arrays
    // -----------------------------------------------------------------------

    #[test]
    fn test_array_of_arrays_size() {
        let t = target_x86_64();
        // [3 x [4 x i32]] → 3 * (4 * 4) = 48
        let inner = IrType::Array {
            element: Box::new(IrType::I32),
            count: 4,
        };
        let outer = IrType::Array {
            element: Box::new(inner),
            count: 3,
        };
        assert_eq!(outer.size(&t), 48);
    }

    // -----------------------------------------------------------------------
    // StructLayout derives and field access
    // -----------------------------------------------------------------------

    #[test]
    fn test_struct_layout_derives() {
        let t = target_x86_64();
        let layout1 = compute_struct_layout(&[IrType::I32], false, &t);
        let layout2 = compute_struct_layout(&[IrType::I32], false, &t);
        // PartialEq, Eq
        assert_eq!(layout1, layout2);
        // Clone
        let layout3 = layout1.clone();
        assert_eq!(layout1, layout3);
    }
}
