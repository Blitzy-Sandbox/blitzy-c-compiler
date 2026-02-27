//! C type system representation with target-parametric sizes for the `bcc` compiler.
//!
//! This is the most foundational file in the semantic analysis module — every other
//! sema file depends on it. It defines:
//!
//! - [`CType`] — The complete C type enum covering all C11 types plus GCC extensions
//! - [`TypeQualifiers`] — `const`, `volatile`, `restrict`, `_Atomic` qualifier tracking
//! - [`IntegerKind`] — All C integer type variants with target-parametric sizing
//! - [`FloatKind`] — All C floating-point type variants with target-parametric sizing
//! - [`StructType`] / [`StructField`] — Struct and union layout with `__attribute__` support
//! - [`EnumType`] — Enum type with variant values
//! - [`FunctionType`] / [`FunctionParam`] — Function types with variadic and K&R support
//! - [`ArraySize`] — Fixed, variable-length, and incomplete array dimensions
//!
//! # Four-Architecture Support
//!
//! All type sizes are parameterized by [`TargetConfig`] from `crate::driver::target`:
//!
//! | Target          | `sizeof(void*)` | `sizeof(long)` | `sizeof(long double)` |
//! |-----------------|-----------------|----------------|-----------------------|
//! | x86_64-linux    | 8               | 8              | 16                    |
//! | i686-linux      | 4               | 4              | 12                    |
//! | aarch64-linux   | 8               | 8              | 16                    |
//! | riscv64-linux   | 8               | 8              | 16                    |

use std::fmt;

use crate::driver::target::TargetConfig;

// ===========================================================================
// TypeQualifiers
// ===========================================================================

/// C type qualifiers: `const`, `volatile`, `restrict`, `_Atomic`.
///
/// Implemented as a simple struct with boolean fields rather than bitflags,
/// following the zero-external-dependency constraint. Used by `type_check.rs`
/// for assignment qualification checking and by `type_conversion.rs` for
/// qualifier preservation during implicit conversions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct TypeQualifiers {
    /// C `const` qualifier — object cannot be modified after initialization.
    pub is_const: bool,
    /// C `volatile` qualifier — accesses cannot be optimized away.
    pub is_volatile: bool,
    /// C `restrict` qualifier — pointer is the sole means of accessing the object.
    /// Only semantically valid on pointer types per C11 §6.7.3.
    pub is_restrict: bool,
    /// C11 `_Atomic` qualifier — accesses are atomic per C11 §6.7.3.
    pub is_atomic: bool,
}

impl TypeQualifiers {
    /// Returns `true` if no qualifiers are set (all fields are `false`).
    #[inline]
    pub fn is_empty(&self) -> bool {
        !self.is_const && !self.is_volatile && !self.is_restrict && !self.is_atomic
    }

    /// Returns the union of two qualifier sets. If either set has a qualifier,
    /// the result has that qualifier. Used when combining qualifiers from
    /// multiple declaration specifiers (e.g., `const volatile int x`).
    #[inline]
    pub fn merge(&self, other: &TypeQualifiers) -> TypeQualifiers {
        TypeQualifiers {
            is_const: self.is_const || other.is_const,
            is_volatile: self.is_volatile || other.is_volatile,
            is_restrict: self.is_restrict || other.is_restrict,
            is_atomic: self.is_atomic || other.is_atomic,
        }
    }

    /// Returns `true` if `self` has at least all qualifiers that `other` has.
    /// Used by assignment type checking: the target qualifier set must contain
    /// the source qualifier set per C11 §6.5.16.1.
    #[inline]
    pub fn contains(&self, other: &TypeQualifiers) -> bool {
        (!other.is_const || self.is_const)
            && (!other.is_volatile || self.is_volatile)
            && (!other.is_restrict || self.is_restrict)
            && (!other.is_atomic || self.is_atomic)
    }
}

impl fmt::Display for TypeQualifiers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if self.is_const {
            parts.push("const");
        }
        if self.is_volatile {
            parts.push("volatile");
        }
        if self.is_restrict {
            parts.push("restrict");
        }
        if self.is_atomic {
            parts.push("_Atomic");
        }
        write!(f, "{}", parts.join(" "))
    }
}

// ===========================================================================
// IntegerKind
// ===========================================================================

/// Enumeration of all C integer type variants.
///
/// Each variant corresponds to a distinct C integer type. Sizes are
/// target-dependent for `Long`/`UnsignedLong` (4 bytes on ILP32 i686,
/// 8 bytes on LP64 x86-64/AArch64/RISC-V 64). All other sizes are
/// fixed across all four supported targets.
///
/// `Char` is treated as signed for GCC compatibility (`-fsigned-char` default).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntegerKind {
    /// `_Bool` — always 1 byte, unsigned semantics (0 or 1).
    Bool,
    /// `char` — 1 byte, implementation-defined signedness (signed for GCC compat).
    Char,
    /// `signed char` — explicitly signed, 1 byte.
    SignedChar,
    /// `unsigned char` — 1 byte.
    UnsignedChar,
    /// `short` / `signed short` — 2 bytes on all targets.
    Short,
    /// `unsigned short` — 2 bytes on all targets.
    UnsignedShort,
    /// `int` / `signed int` — 4 bytes on all targets.
    Int,
    /// `unsigned int` — 4 bytes on all targets.
    UnsignedInt,
    /// `long` / `signed long` — 4 bytes on i686 (ILP32), 8 bytes on LP64 targets.
    Long,
    /// `unsigned long` — same size as `long`.
    UnsignedLong,
    /// `long long` / `signed long long` — 8 bytes on all targets.
    LongLong,
    /// `unsigned long long` — 8 bytes on all targets.
    UnsignedLongLong,
}

impl IntegerKind {
    /// Returns `true` if this integer type is a signed variant.
    ///
    /// `Char` is treated as signed (GCC default behavior).
    /// `Bool` is unsigned (it only holds 0 or 1).
    #[inline]
    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            IntegerKind::Char
                | IntegerKind::SignedChar
                | IntegerKind::Short
                | IntegerKind::Int
                | IntegerKind::Long
                | IntegerKind::LongLong
        )
    }

    /// Returns `true` if this integer type is an unsigned variant.
    ///
    /// `Bool` is considered unsigned per C11 §6.2.5.
    #[inline]
    pub fn is_unsigned(&self) -> bool {
        !self.is_signed()
    }

    /// Returns the integer conversion rank per C11 §6.3.1.1.
    ///
    /// The rank determines promotion and conversion precedence:
    /// `_Bool` < `char`/`signed char`/`unsigned char` < `short`/`unsigned short`
    /// < `int`/`unsigned int` < `long`/`unsigned long` < `long long`/`unsigned long long`
    ///
    /// Signed and unsigned variants of the same base type share the same rank.
    #[inline]
    pub fn rank(&self) -> u8 {
        match self {
            IntegerKind::Bool => 1,
            IntegerKind::Char | IntegerKind::SignedChar | IntegerKind::UnsignedChar => 2,
            IntegerKind::Short | IntegerKind::UnsignedShort => 3,
            IntegerKind::Int | IntegerKind::UnsignedInt => 4,
            IntegerKind::Long | IntegerKind::UnsignedLong => 5,
            IntegerKind::LongLong | IntegerKind::UnsignedLongLong => 6,
        }
    }

    /// Converts this integer kind to its unsigned counterpart.
    ///
    /// Already-unsigned kinds are returned unchanged. `Char` maps to `UnsignedChar`.
    #[inline]
    pub fn to_unsigned(&self) -> IntegerKind {
        match self {
            IntegerKind::Bool => IntegerKind::Bool,
            IntegerKind::Char | IntegerKind::SignedChar | IntegerKind::UnsignedChar => {
                IntegerKind::UnsignedChar
            }
            IntegerKind::Short | IntegerKind::UnsignedShort => IntegerKind::UnsignedShort,
            IntegerKind::Int | IntegerKind::UnsignedInt => IntegerKind::UnsignedInt,
            IntegerKind::Long | IntegerKind::UnsignedLong => IntegerKind::UnsignedLong,
            IntegerKind::LongLong | IntegerKind::UnsignedLongLong => IntegerKind::UnsignedLongLong,
        }
    }

    /// Converts this integer kind to its signed counterpart.
    ///
    /// Already-signed kinds are returned unchanged. `Bool` maps to `SignedChar`
    /// (smallest signed type that can represent Bool's range).
    #[inline]
    pub fn to_signed(&self) -> IntegerKind {
        match self {
            IntegerKind::Bool => IntegerKind::SignedChar,
            IntegerKind::Char | IntegerKind::SignedChar | IntegerKind::UnsignedChar => {
                IntegerKind::SignedChar
            }
            IntegerKind::Short | IntegerKind::UnsignedShort => IntegerKind::Short,
            IntegerKind::Int | IntegerKind::UnsignedInt => IntegerKind::Int,
            IntegerKind::Long | IntegerKind::UnsignedLong => IntegerKind::Long,
            IntegerKind::LongLong | IntegerKind::UnsignedLongLong => IntegerKind::LongLong,
        }
    }

    /// Returns the size in bytes of this integer type on the given target.
    ///
    /// Most integer sizes are fixed across targets. The exception is
    /// `Long`/`UnsignedLong`, which is 4 bytes on i686 (ILP32) and
    /// 8 bytes on x86-64, AArch64, and RISC-V 64 (LP64).
    #[inline]
    pub fn size(&self, target: &TargetConfig) -> usize {
        match self {
            IntegerKind::Bool => 1,
            IntegerKind::Char | IntegerKind::SignedChar | IntegerKind::UnsignedChar => 1,
            IntegerKind::Short | IntegerKind::UnsignedShort => 2,
            IntegerKind::Int | IntegerKind::UnsignedInt => 4,
            IntegerKind::Long | IntegerKind::UnsignedLong => target.long_size as usize,
            IntegerKind::LongLong | IntegerKind::UnsignedLongLong => 8,
        }
    }

    /// Returns the alignment in bytes of this integer type on the given target.
    ///
    /// Integer alignment equals `min(size, target.max_alignment)`. In practice,
    /// all integer sizes are ≤16 bytes (the max alignment on all supported
    /// targets), so alignment equals size.
    #[inline]
    pub fn alignment(&self, target: &TargetConfig) -> usize {
        let sz = self.size(target) as u32;
        target.alignment_of(sz) as usize
    }
}

impl fmt::Display for IntegerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntegerKind::Bool => write!(f, "_Bool"),
            IntegerKind::Char => write!(f, "char"),
            IntegerKind::SignedChar => write!(f, "signed char"),
            IntegerKind::UnsignedChar => write!(f, "unsigned char"),
            IntegerKind::Short => write!(f, "short"),
            IntegerKind::UnsignedShort => write!(f, "unsigned short"),
            IntegerKind::Int => write!(f, "int"),
            IntegerKind::UnsignedInt => write!(f, "unsigned int"),
            IntegerKind::Long => write!(f, "long"),
            IntegerKind::UnsignedLong => write!(f, "unsigned long"),
            IntegerKind::LongLong => write!(f, "long long"),
            IntegerKind::UnsignedLongLong => write!(f, "unsigned long long"),
        }
    }
}

// ===========================================================================
// FloatKind
// ===========================================================================

/// Enumeration of C floating-point type variants.
///
/// `Float` and `Double` have fixed sizes across all targets (IEEE 754).
/// `LongDouble` is target-dependent:
/// - x86-64: 16 bytes (80-bit x87 extended, padded to 16 per SysV AMD64 ABI)
/// - i686:   12 bytes (80-bit x87 extended, padded to 12 per SysV i386 ABI)
/// - AArch64: 16 bytes (IEEE 754 binary128 quad precision)
/// - RISC-V 64: 16 bytes (IEEE 754 binary128 quad precision)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatKind {
    /// `float` — 4 bytes, IEEE 754 binary32 single precision.
    Float,
    /// `double` — 8 bytes, IEEE 754 binary64 double precision.
    Double,
    /// `long double` — target-dependent size (12 or 16 bytes).
    LongDouble,
}

impl FloatKind {
    /// Returns the float conversion rank for usual arithmetic conversions.
    ///
    /// `Float` < `Double` < `LongDouble` per C11 §6.3.1.8.
    #[inline]
    pub fn rank(&self) -> u8 {
        match self {
            FloatKind::Float => 1,
            FloatKind::Double => 2,
            FloatKind::LongDouble => 3,
        }
    }

    /// Returns the size in bytes of this float type on the given target.
    #[inline]
    pub fn size(&self, target: &TargetConfig) -> usize {
        match self {
            FloatKind::Float => 4,
            FloatKind::Double => 8,
            FloatKind::LongDouble => target.long_double_size as usize,
        }
    }

    /// Returns the alignment in bytes of this float type on the given target.
    ///
    /// Float alignment follows the same rule as integer alignment:
    /// `min(size, target.max_alignment)`.
    #[inline]
    pub fn alignment(&self, target: &TargetConfig) -> usize {
        let sz = self.size(target) as u32;
        target.alignment_of(sz) as usize
    }
}

impl fmt::Display for FloatKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FloatKind::Float => write!(f, "float"),
            FloatKind::Double => write!(f, "double"),
            FloatKind::LongDouble => write!(f, "long double"),
        }
    }
}

// ===========================================================================
// StructField, StructType
// ===========================================================================

/// A single field within a struct or union type.
///
/// The `offset` field is computed during layout (see [`StructType::compute_layout`])
/// and defaults to 0 before layout has been performed.
#[derive(Debug, Clone)]
pub struct StructField {
    /// Field name, or `None` for anonymous struct/union members and unnamed
    /// bit-fields (e.g., `int : 0;` for forcing alignment).
    pub name: Option<String>,
    /// The C type of this field.
    pub ty: CType,
    /// Bit-field width in bits, or `None` for regular (non-bit-field) members.
    pub bit_width: Option<u32>,
    /// Byte offset from the start of the containing struct/union.
    /// Set to 0 by default; computed by [`StructType::compute_layout`].
    pub offset: usize,
}

/// A struct or union type definition.
///
/// Supports GCC `__attribute__((packed))` (no padding) and
/// `__attribute__((aligned(N)))` (custom minimum alignment).
/// Forward declarations are represented with `is_complete = false`.
#[derive(Debug, Clone)]
pub struct StructType {
    /// The tag name (e.g., `"foo"` in `struct foo { ... }`), or `None`
    /// for anonymous struct/union types.
    pub tag: Option<String>,
    /// Ordered list of fields. Empty for forward declarations.
    pub fields: Vec<StructField>,
    /// `true` if this is a `union`, `false` if this is a `struct`.
    pub is_union: bool,
    /// `true` if `__attribute__((packed))` is applied — suppresses padding.
    pub is_packed: bool,
    /// Explicit minimum alignment from `__attribute__((aligned(N)))`, or `None`
    /// for natural alignment.
    pub custom_alignment: Option<usize>,
    /// `false` for forward declarations (`struct foo;`), `true` once the
    /// definition with fields has been seen.
    pub is_complete: bool,
}

// ===========================================================================
// EnumType
// ===========================================================================

/// An enumeration type definition.
///
/// In C11, the underlying type of an enum is always `int` (C11 §6.7.2.2).
/// Forward declarations are represented with `is_complete = false`.
#[derive(Debug, Clone)]
pub struct EnumType {
    /// The tag name (e.g., `"color"` in `enum color { ... }`), or `None`
    /// for anonymous enums.
    pub tag: Option<String>,
    /// Ordered list of `(enumerator_name, value)` pairs.
    pub variants: Vec<(String, i64)>,
    /// `false` for forward declarations, `true` once the definition is seen.
    pub is_complete: bool,
}

// ===========================================================================
// FunctionParam, FunctionType
// ===========================================================================

/// A single parameter in a function type.
#[derive(Debug, Clone)]
pub struct FunctionParam {
    /// Parameter name, or `None` in abstract declarators / prototypes
    /// without names (e.g., `void f(int, float)`).
    pub name: Option<String>,
    /// The C type of this parameter (after parameter type adjustment:
    /// arrays decay to pointers, functions decay to function pointers).
    pub ty: CType,
}

/// A function type representing the signature of a C function.
///
/// Distinguishes between C11 prototyped functions and K&R-style
/// (old-style) function definitions without prototypes.
#[derive(Debug, Clone)]
pub struct FunctionType {
    /// The return type of the function (boxed to avoid infinite recursion).
    pub return_type: Box<CType>,
    /// The parameter list. Empty for `void f(void)` and K&R functions.
    pub params: Vec<FunctionParam>,
    /// `true` if the function accepts variadic arguments (`...` at end).
    pub is_variadic: bool,
    /// `true` for K&R-style function definitions without a prototype.
    /// K&R functions skip parameter type checking for calls made before
    /// the definition is visible.
    pub is_old_style: bool,
}

// ===========================================================================
// ArraySize
// ===========================================================================

/// Discriminates fixed-size, variable-length, and incomplete array dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArraySize {
    /// Fixed-size array: `int arr[10]` — element count known at compile time.
    Fixed(usize),
    /// Variable-length array (VLA): `int arr[n]` — size determined at runtime.
    /// C11 §6.7.6.2.
    Variable,
    /// Incomplete (unsized) array: `int arr[]` — used for flexible array members
    /// (C11 §6.7.2.1), extern declarations, and function parameters.
    Incomplete,
}

// ===========================================================================
// CType — the central type representation
// ===========================================================================

/// The complete C type representation for all C11 types plus GCC extensions.
///
/// This enum is the central type in the semantic analysis module. Every expression,
/// declaration, and symbol table entry carries a `CType`. The type system is
/// designed to be target-parametric: size and alignment queries take a
/// [`TargetConfig`] reference to support all four architectures.
///
/// # Variants
///
/// - Primitive types: [`Void`](CType::Void), [`Integer`](CType::Integer),
///   [`Float`](CType::Float)
/// - Derived types: [`Pointer`](CType::Pointer), [`Array`](CType::Array),
///   [`Function`](CType::Function)
/// - Composite types: [`Struct`](CType::Struct), [`Enum`](CType::Enum)
/// - Type system wrappers: [`Typedef`](CType::Typedef),
///   [`Qualified`](CType::Qualified), [`TypeOf`](CType::TypeOf)
/// - Error sentinel: [`Error`](CType::Error)
#[derive(Debug, Clone)]
pub enum CType {
    /// `void` — incomplete type, cannot be used as a value type.
    Void,
    /// Integer types: `_Bool`, `char`, `short`, `int`, `long`, `long long`
    /// and their unsigned variants.
    Integer(IntegerKind),
    /// Floating-point types: `float`, `double`, `long double`.
    Float(FloatKind),
    /// Pointer type with optional qualifiers on the pointer itself
    /// (e.g., `int * const p` — pointer is const, pointee is int).
    Pointer {
        /// The type being pointed to.
        pointee: Box<CType>,
        /// Qualifiers on the pointer itself (not on the pointee).
        qualifiers: TypeQualifiers,
    },
    /// Array type with element type and dimension.
    Array {
        /// The type of each array element.
        element: Box<CType>,
        /// The array dimension: fixed size, VLA, or incomplete.
        size: ArraySize,
    },
    /// Struct or union type.
    Struct(StructType),
    /// Enum type (underlying type is always `int` in C11).
    Enum(EnumType),
    /// Function type (not a function pointer — that would be `Pointer` to `Function`).
    Function(FunctionType),
    /// Typedef alias: preserves the user-given name alongside the resolved type.
    Typedef {
        /// The typedef name (e.g., `"size_t"`).
        name: String,
        /// The resolved underlying type.
        underlying: Box<CType>,
    },
    /// Qualified type: wraps a base type with `const`/`volatile`/`restrict`/`_Atomic`.
    Qualified {
        /// The base type being qualified.
        base: Box<CType>,
        /// The qualifiers applied.
        qualifiers: TypeQualifiers,
    },
    /// GCC `typeof` / `__typeof__` — resolved to the actual type during
    /// semantic analysis. Present in the AST before full resolution.
    TypeOf(Box<CType>),
    /// Error/poison type — used for error recovery after type errors.
    /// Propagates silently through subsequent checks to avoid cascading errors.
    Error,
}

// ===========================================================================
// CType — Size and alignment computation
// ===========================================================================

impl CType {
    /// Returns the size in bytes of this type on the given target, or `None`
    /// for incomplete types (void, incomplete arrays, forward-declared structs,
    /// function types, and error types).
    ///
    /// This is the primary entry point for `sizeof` evaluation and struct layout.
    pub fn size(&self, target: &TargetConfig) -> Option<usize> {
        match self {
            CType::Void => None,
            CType::Integer(kind) => Some(kind.size(target)),
            CType::Float(kind) => Some(kind.size(target)),
            CType::Pointer { .. } => Some(target.pointer_size as usize),
            CType::Array { element, size } => match size {
                ArraySize::Fixed(n) => element.size(target).map(|elem_sz| elem_sz * n),
                ArraySize::Variable => None,
                ArraySize::Incomplete => None,
            },
            CType::Struct(s) => {
                if !s.is_complete {
                    return None;
                }
                Some(struct_total_size(s, target))
            }
            CType::Enum(e) => {
                if !e.is_complete {
                    return None;
                }
                // Enum underlying type is always int in C11 (section 6.7.2.2).
                Some(4)
            }
            CType::Function(_) => None,
            CType::Typedef { underlying, .. } => underlying.size(target),
            CType::Qualified { base, .. } => base.size(target),
            CType::TypeOf(inner) => inner.size(target),
            CType::Error => None,
        }
    }

    /// Returns the alignment in bytes of this type on the given target, or `None`
    /// for types that have no meaningful alignment (void, function types, error).
    pub fn alignment(&self, target: &TargetConfig) -> Option<usize> {
        match self {
            CType::Void => None,
            CType::Integer(kind) => Some(kind.alignment(target)),
            CType::Float(kind) => Some(kind.alignment(target)),
            CType::Pointer { .. } => Some(target.pointer_size as usize),
            CType::Array { element, .. } => element.alignment(target),
            CType::Struct(s) => {
                if !s.is_complete {
                    return None;
                }
                Some(struct_alignment(s, target))
            }
            CType::Enum(_) => {
                // Enum alignment = int alignment = 4 on all targets.
                Some(target.alignment_of(4) as usize)
            }
            CType::Function(_) => None,
            CType::Typedef { underlying, .. } => underlying.alignment(target),
            CType::Qualified { base, .. } => base.alignment(target),
            CType::TypeOf(inner) => inner.alignment(target),
            CType::Error => None,
        }
    }
}

// ===========================================================================
// CType — Type classification helpers
// ===========================================================================

impl CType {
    /// Returns `true` if this is an integer type (including `_Bool`).
    #[inline]
    pub fn is_integer(&self) -> bool {
        matches!(self.canonical(), CType::Integer(_))
    }

    /// Returns `true` if this is a floating-point type.
    #[inline]
    pub fn is_float(&self) -> bool {
        matches!(self.canonical(), CType::Float(_))
    }

    /// Returns `true` if this is an arithmetic type (integer or float).
    /// Enum types are also arithmetic (underlying type is `int`).
    #[inline]
    pub fn is_arithmetic(&self) -> bool {
        match self.canonical() {
            CType::Integer(_) | CType::Float(_) | CType::Enum(_) => true,
            _ => false,
        }
    }

    /// Returns `true` if this is a scalar type (arithmetic or pointer).
    /// Scalar types can be used in boolean contexts (`if`, `while`, `&&`, etc.).
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.is_arithmetic() || self.is_pointer()
    }

    /// Returns `true` if this is a pointer type.
    #[inline]
    pub fn is_pointer(&self) -> bool {
        matches!(self.canonical(), CType::Pointer { .. })
    }

    /// Returns `true` if this is an array type.
    #[inline]
    pub fn is_array(&self) -> bool {
        matches!(self.canonical(), CType::Array { .. })
    }

    /// Returns `true` if this is a function type (not a function pointer).
    #[inline]
    pub fn is_function(&self) -> bool {
        matches!(self.canonical(), CType::Function(_))
    }

    /// Returns `true` if this is a struct type (not a union).
    #[inline]
    pub fn is_struct(&self) -> bool {
        match self.canonical() {
            CType::Struct(s) => !s.is_union,
            _ => false,
        }
    }

    /// Returns `true` if this is a union type.
    #[inline]
    pub fn is_union(&self) -> bool {
        match self.canonical() {
            CType::Struct(s) => s.is_union,
            _ => false,
        }
    }

    /// Returns `true` if this is `void`.
    #[inline]
    pub fn is_void(&self) -> bool {
        matches!(self.canonical(), CType::Void)
    }

    /// Returns `true` if this type is complete (has a known size).
    ///
    /// Incomplete types include: `void`, incomplete arrays, forward-declared
    /// structs/unions, incomplete enums, function types, and error types.
    pub fn is_complete(&self) -> bool {
        match self.canonical() {
            CType::Void => false,
            CType::Integer(_) | CType::Float(_) | CType::Pointer { .. } => true,
            CType::Array { size, element, .. } => {
                matches!(size, ArraySize::Fixed(_)) && element.is_complete()
            }
            CType::Struct(s) => s.is_complete,
            CType::Enum(e) => e.is_complete,
            CType::Function(_) => false,
            CType::Error => false,
            // These should be resolved before completeness checks, but handle gracefully.
            CType::Typedef { underlying, .. } => underlying.is_complete(),
            CType::Qualified { base, .. } => base.is_complete(),
            CType::TypeOf(inner) => inner.is_complete(),
        }
    }

    /// Returns `true` if this is a pointer to `void` (`void *`).
    #[inline]
    pub fn is_void_pointer(&self) -> bool {
        match self.canonical() {
            CType::Pointer { pointee, .. } => pointee.canonical().is_void(),
            _ => false,
        }
    }

    /// Returns `true` if this type represents a null pointer constant.
    ///
    /// A null pointer constant is an integer constant expression with value 0,
    /// or such an expression cast to `void *` (C11 section 6.3.2.3).
    /// This is a simplified check; the full check requires expression-level
    /// analysis in the semantic analyzer.
    #[inline]
    pub fn is_null_pointer_constant(&self) -> bool {
        // At the type level, we can identify `void *` and integer types.
        // The full check (integer value == 0) is done in the expression analyzer.
        self.is_void_pointer()
    }

    /// Returns `true` if this is the error/poison type.
    #[inline]
    pub fn is_error(&self) -> bool {
        matches!(self, CType::Error)
    }

    /// Returns the base type with the outermost `Qualified` wrapper stripped.
    ///
    /// If this type is not qualified, returns `self`. Used for type
    /// comparison where qualifiers are not relevant (e.g., compatibility checks).
    pub fn unqualified(&self) -> &CType {
        match self {
            CType::Qualified { base, .. } => base.unqualified(),
            CType::Typedef { underlying, .. } => underlying.unqualified(),
            other => other,
        }
    }

    /// Returns the qualifiers on this type, or empty qualifiers if not qualified.
    pub fn qualifiers(&self) -> TypeQualifiers {
        match self {
            CType::Qualified { qualifiers, .. } => *qualifiers,
            CType::Pointer { qualifiers, .. } => *qualifiers,
            CType::Typedef { underlying, .. } => underlying.qualifiers(),
            _ => TypeQualifiers::default(),
        }
    }

    /// Performs array-to-pointer and function-to-pointer decay per C11 section 6.3.2.1.
    ///
    /// - `T[N]` -> `T *`  (array to pointer to first element)
    /// - `T[]`  -> `T *`  (incomplete array to pointer)
    /// - `fn(params) -> ret` -> pointer to `fn(params) -> ret`
    /// - All other types -> returned unchanged (cloned).
    pub fn decay(&self) -> CType {
        match self.canonical() {
            CType::Array { element, .. } => CType::Pointer {
                pointee: element.clone(),
                qualifiers: TypeQualifiers::default(),
            },
            CType::Function(ft) => CType::Pointer {
                pointee: Box::new(CType::Function(ft.clone())),
                qualifiers: TypeQualifiers::default(),
            },
            _ => self.clone(),
        }
    }

    /// Resolves through `Typedef` and `TypeOf` wrappers to the canonical type.
    /// Does NOT strip `Qualified` -- use [`unqualified`](CType::unqualified) for that.
    fn canonical(&self) -> &CType {
        match self {
            CType::Typedef { underlying, .. } => underlying.canonical(),
            CType::TypeOf(inner) => inner.canonical(),
            other => other,
        }
    }
}

// ===========================================================================
// CType — Type compatibility and comparison (C11 section 6.2.7)
// ===========================================================================

impl CType {
    /// Returns `true` if two types are compatible per C11 section 6.2.7.
    ///
    /// Compatible types are structurally equivalent after stripping qualifiers
    /// and resolving typedefs. Key rules:
    /// - Same basic type (after qualifier removal) -> compatible
    /// - Two pointers: compatible if pointee types are compatible
    /// - Two arrays: compatible if element types are compatible and sizes match
    ///   (or at least one is incomplete)
    /// - Two functions: compatible if return types and all parameter types are
    ///   compatible, and variadic status matches
    /// - Struct/union: compatible only if they refer to the same definition
    ///   (tag-based identity, not structural equality)
    pub fn is_compatible(&self, other: &CType) -> bool {
        let a = self.unqualified();
        let b = other.unqualified();

        // Error types are compatible with everything to prevent cascading errors.
        if a.is_error() || b.is_error() {
            return true;
        }

        match (a, b) {
            (CType::Void, CType::Void) => true,
            (CType::Integer(ak), CType::Integer(bk)) => ak == bk,
            (CType::Float(ak), CType::Float(bk)) => ak == bk,
            (CType::Pointer { pointee: ap, .. }, CType::Pointer { pointee: bp, .. }) => {
                ap.is_compatible(bp)
            }
            (
                CType::Array {
                    element: ae,
                    size: as_,
                },
                CType::Array {
                    element: be,
                    size: bs_,
                },
            ) => {
                if !ae.is_compatible(be) {
                    return false;
                }
                match (as_, bs_) {
                    (ArraySize::Fixed(an), ArraySize::Fixed(bn)) => an == bn,
                    // One or both incomplete -> compatible if element types match.
                    _ => true,
                }
            }
            (CType::Function(af), CType::Function(bf)) => {
                if !af.return_type.is_compatible(&bf.return_type) {
                    return false;
                }
                if af.is_variadic != bf.is_variadic {
                    return false;
                }
                // If either is old-style (no prototype), they are compatible
                // with any parameter list (C11 section 6.7.6.3 p15).
                if af.is_old_style || bf.is_old_style {
                    return true;
                }
                if af.params.len() != bf.params.len() {
                    return false;
                }
                af.params
                    .iter()
                    .zip(bf.params.iter())
                    .all(|(ap, bp)| ap.ty.is_compatible(&bp.ty))
            }
            (CType::Struct(as_), CType::Struct(bs_)) => {
                // Struct/union identity: same tag and same union-ness.
                // In a real compiler, identity would be by pointer to definition,
                // but for this representation we use tag comparison.
                as_.is_union == bs_.is_union && as_.tag == bs_.tag && as_.tag.is_some()
            }
            (CType::Enum(ae), CType::Enum(be)) => ae.tag == be.tag && ae.tag.is_some(),
            // Enum is compatible with int (underlying type).
            (CType::Enum(_), CType::Integer(IntegerKind::Int))
            | (CType::Integer(IntegerKind::Int), CType::Enum(_)) => true,
            _ => false,
        }
    }

    /// Returns `true` if two types are the same type including qualifiers.
    ///
    /// Stricter than [`is_compatible`](CType::is_compatible): qualifiers
    /// must match exactly. Used for redeclaration checking.
    pub fn is_same_type(&self, other: &CType) -> bool {
        // Check qualifiers match.
        if self.qualifiers() != other.qualifiers() {
            return false;
        }
        let a = self.unqualified();
        let b = other.unqualified();

        if a.is_error() || b.is_error() {
            return true;
        }

        match (a, b) {
            (CType::Void, CType::Void) => true,
            (CType::Integer(ak), CType::Integer(bk)) => ak == bk,
            (CType::Float(ak), CType::Float(bk)) => ak == bk,
            (
                CType::Pointer {
                    pointee: ap,
                    qualifiers: aq,
                },
                CType::Pointer {
                    pointee: bp,
                    qualifiers: bq,
                },
            ) => aq == bq && ap.is_same_type(bp),
            (
                CType::Array {
                    element: ae,
                    size: as_,
                },
                CType::Array {
                    element: be,
                    size: bs_,
                },
            ) => as_ == bs_ && ae.is_same_type(be),
            (CType::Function(af), CType::Function(bf)) => {
                if !af.return_type.is_same_type(&bf.return_type) {
                    return false;
                }
                if af.is_variadic != bf.is_variadic || af.is_old_style != bf.is_old_style {
                    return false;
                }
                if af.params.len() != bf.params.len() {
                    return false;
                }
                af.params
                    .iter()
                    .zip(bf.params.iter())
                    .all(|(ap, bp)| ap.ty.is_same_type(&bp.ty))
            }
            (CType::Struct(as_), CType::Struct(bs_)) => {
                as_.is_union == bs_.is_union && as_.tag == bs_.tag && as_.tag.is_some()
            }
            (CType::Enum(ae), CType::Enum(be)) => ae.tag == be.tag && ae.tag.is_some(),
            _ => false,
        }
    }

    /// Computes the composite type of two compatible types per C11 section 6.2.7.
    ///
    /// Used when combining a forward declaration with a definition, or when
    /// merging two compatible declarations. The composite type contains the
    /// most complete information from both types.
    pub fn composite_type(&self, other: &CType) -> CType {
        let a = self.unqualified();
        let b = other.unqualified();

        match (a, b) {
            (
                CType::Array {
                    element: ae,
                    size: as_,
                },
                CType::Array {
                    element: be,
                    size: bs_,
                },
            ) => {
                let elem = ae.composite_type(be);
                let sz = match (as_, bs_) {
                    (ArraySize::Fixed(n), _) | (_, ArraySize::Fixed(n)) => ArraySize::Fixed(*n),
                    (ArraySize::Variable, _) | (_, ArraySize::Variable) => ArraySize::Variable,
                    _ => ArraySize::Incomplete,
                };
                CType::Array {
                    element: Box::new(elem),
                    size: sz,
                }
            }
            (CType::Function(af), CType::Function(bf)) => {
                let ret = af.return_type.composite_type(&bf.return_type);
                let params = if af.is_old_style && !bf.is_old_style {
                    bf.params.clone()
                } else if bf.is_old_style && !af.is_old_style {
                    af.params.clone()
                } else {
                    // Both have prototypes: composite parameters.
                    af.params
                        .iter()
                        .zip(bf.params.iter())
                        .map(|(ap, bp)| FunctionParam {
                            name: ap.name.clone().or_else(|| bp.name.clone()),
                            ty: ap.ty.composite_type(&bp.ty),
                        })
                        .collect()
                };
                CType::Function(FunctionType {
                    return_type: Box::new(ret),
                    params,
                    is_variadic: af.is_variadic || bf.is_variadic,
                    is_old_style: false,
                })
            }
            (
                CType::Pointer {
                    pointee: ap,
                    qualifiers: aq,
                },
                CType::Pointer {
                    pointee: bp,
                    qualifiers: bq,
                },
            ) => CType::Pointer {
                pointee: Box::new(ap.composite_type(bp)),
                qualifiers: aq.merge(bq),
            },
            // For non-composite cases, prefer the more complete type.
            (CType::Struct(as_), CType::Struct(bs_)) => {
                if as_.is_complete {
                    CType::Struct(as_.clone())
                } else {
                    CType::Struct(bs_.clone())
                }
            }
            // Default: return self.
            _ => self.clone(),
        }
    }
}

// ===========================================================================
// StructType — layout computation
// ===========================================================================

impl StructType {
    /// Computes the byte offset for each field and updates `self.fields[i].offset`.
    ///
    /// For **structs**: Fields are laid out sequentially with padding for alignment.
    /// For **unions**: All fields are at offset 0.
    /// For **packed structs** (`__attribute__((packed))`): No padding is inserted.
    /// For **aligned structs** (`__attribute__((aligned(N)))`): The final alignment
    /// is `max(natural_alignment, N)`.
    ///
    /// Bit-field layout: bit-fields are packed into the current storage unit of
    /// the underlying type. A zero-width bit-field forces alignment to the next
    /// storage unit boundary.
    pub fn compute_layout(&mut self, target: &TargetConfig) {
        if !self.is_complete {
            return;
        }

        if self.is_union {
            // Union: all fields at offset 0.
            for field in &mut self.fields {
                field.offset = 0;
            }
            return;
        }

        // Struct layout computation.
        let mut current_offset: usize = 0;
        // Track current bit-field position within its storage unit.
        let mut bit_offset: u32 = 0;
        let mut bit_storage_size: u32 = 0;

        for field in &mut self.fields {
            if let Some(bw) = field.bit_width {
                // Bit-field handling.
                let storage_size = field.ty.size(target).unwrap_or(4);
                let storage_bits = (storage_size as u32) * 8;

                if bw == 0 {
                    // Zero-width bit-field: align to next storage unit boundary.
                    if bit_offset > 0 {
                        current_offset += bit_storage_size as usize;
                        bit_offset = 0;
                        bit_storage_size = 0;
                    }
                    field.offset = current_offset;
                    continue;
                }

                // Check if the bit-field fits in the current storage unit.
                if bit_offset + bw > storage_bits || bit_storage_size != storage_size as u32 {
                    // Does not fit or different storage type: start a new unit.
                    if bit_offset > 0 {
                        current_offset += bit_storage_size as usize;
                    }
                    if !self.is_packed {
                        let align = field.ty.alignment(target).unwrap_or(1);
                        current_offset = align_up(current_offset, align);
                    }
                    bit_offset = 0;
                    bit_storage_size = storage_size as u32;
                }

                field.offset = current_offset;
                bit_offset += bw;

                // If we filled the storage unit, advance.
                if bit_offset >= storage_bits {
                    current_offset += storage_size;
                    bit_offset = 0;
                    bit_storage_size = 0;
                }
            } else {
                // Non-bit-field: flush any pending bit-field storage.
                if bit_offset > 0 {
                    current_offset += bit_storage_size as usize;
                    bit_offset = 0;
                    bit_storage_size = 0;
                }

                let field_align = if self.is_packed {
                    1
                } else {
                    field.ty.alignment(target).unwrap_or(1)
                };
                current_offset = align_up(current_offset, field_align);
                field.offset = current_offset;

                let field_size = field.ty.size(target).unwrap_or(0);
                current_offset += field_size;
            }
        }

        // Flush final bit-field storage if any.
        if bit_offset > 0 {
            current_offset += bit_storage_size as usize;
        }
        let _ = current_offset; // suppress unused warning
    }
}

// ===========================================================================
// Helper functions for struct size/alignment
// ===========================================================================

/// Computes the total size of a struct/union including trailing padding.
fn struct_total_size(s: &StructType, target: &TargetConfig) -> usize {
    if s.is_union {
        // Union size = max of field sizes, aligned to union alignment.
        let max_field_size = s
            .fields
            .iter()
            .filter_map(|f| f.ty.size(target))
            .max()
            .unwrap_or(0);
        let alignment = struct_alignment(s, target);
        align_up(max_field_size, alignment)
    } else {
        // Struct size = last field offset + last field size + trailing padding.
        let raw_size = if s.fields.is_empty() {
            0
        } else {
            let mut current_offset: usize = 0;
            let mut bit_offset: u32 = 0;
            let mut bit_storage_size: u32 = 0;

            for field in &s.fields {
                if let Some(bw) = field.bit_width {
                    let storage_size = field.ty.size(target).unwrap_or(4);
                    let storage_bits = (storage_size as u32) * 8;

                    if bw == 0 {
                        if bit_offset > 0 {
                            current_offset += bit_storage_size as usize;
                            bit_offset = 0;
                            bit_storage_size = 0;
                        }
                        continue;
                    }

                    if bit_offset + bw > storage_bits || bit_storage_size != storage_size as u32 {
                        if bit_offset > 0 {
                            current_offset += bit_storage_size as usize;
                        }
                        if !s.is_packed {
                            let align = field.ty.alignment(target).unwrap_or(1);
                            current_offset = align_up(current_offset, align);
                        }
                        bit_offset = 0;
                        bit_storage_size = storage_size as u32;
                    }

                    bit_offset += bw;
                    if bit_offset >= storage_bits {
                        current_offset += storage_size;
                        bit_offset = 0;
                        bit_storage_size = 0;
                    }
                } else {
                    if bit_offset > 0 {
                        current_offset += bit_storage_size as usize;
                        bit_offset = 0;
                        bit_storage_size = 0;
                    }
                    let field_align = if s.is_packed {
                        1
                    } else {
                        field.ty.alignment(target).unwrap_or(1)
                    };
                    current_offset = align_up(current_offset, field_align);
                    let field_size = field.ty.size(target).unwrap_or(0);
                    current_offset += field_size;
                }
            }
            if bit_offset > 0 {
                current_offset += bit_storage_size as usize;
            }
            current_offset
        };

        let alignment = struct_alignment(s, target);
        align_up(raw_size, alignment)
    }
}

/// Computes the alignment requirement of a struct/union.
fn struct_alignment(s: &StructType, target: &TargetConfig) -> usize {
    if s.is_packed && s.custom_alignment.is_none() {
        return 1;
    }

    let natural_align = s
        .fields
        .iter()
        .filter_map(|f| {
            if s.is_packed {
                Some(1)
            } else {
                f.ty.alignment(target)
            }
        })
        .max()
        .unwrap_or(1);

    match s.custom_alignment {
        Some(custom) => std::cmp::max(natural_align, custom),
        None => natural_align,
    }
}

/// Rounds `offset` up to the next multiple of `alignment`.
#[inline]
fn align_up(offset: usize, alignment: usize) -> usize {
    if alignment == 0 {
        return offset;
    }
    (offset + alignment - 1) & !(alignment - 1)
}

// ===========================================================================
// Display implementations for diagnostics
// ===========================================================================

impl fmt::Display for CType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CType::Void => write!(f, "void"),
            CType::Integer(kind) => write!(f, "{}", kind),
            CType::Float(kind) => write!(f, "{}", kind),
            CType::Pointer {
                pointee,
                qualifiers,
            } => {
                write!(f, "{} *", pointee)?;
                if !qualifiers.is_empty() {
                    write!(f, " {}", qualifiers)?;
                }
                Ok(())
            }
            CType::Array { element, size } => match size {
                ArraySize::Fixed(n) => write!(f, "{}[{}]", element, n),
                ArraySize::Variable => write!(f, "{}[*]", element),
                ArraySize::Incomplete => write!(f, "{}[]", element),
            },
            CType::Struct(s) => {
                let keyword = if s.is_union { "union" } else { "struct" };
                match &s.tag {
                    Some(tag) => write!(f, "{} {}", keyword, tag),
                    None => write!(f, "<anonymous {}>", keyword),
                }
            }
            CType::Enum(e) => match &e.tag {
                Some(tag) => write!(f, "enum {}", tag),
                None => write!(f, "<anonymous enum>"),
            },
            CType::Function(ft) => {
                write!(f, "{}(", ft.return_type)?;
                for (i, param) in ft.params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", param.ty)?;
                }
                if ft.is_variadic {
                    if !ft.params.is_empty() {
                        write!(f, ", ")?;
                    }
                    write!(f, "...")?;
                }
                write!(f, ")")
            }
            CType::Typedef { name, .. } => write!(f, "{}", name),
            CType::Qualified { base, qualifiers } => {
                if !qualifiers.is_empty() {
                    write!(f, "{} {}", qualifiers, base)
                } else {
                    write!(f, "{}", base)
                }
            }
            CType::TypeOf(inner) => write!(f, "typeof({})", inner),
            CType::Error => write!(f, "<error>"),
        }
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::target::TargetConfig;

    /// Creates a TargetConfig for x86-64 (LP64: pointer=8, long=8, long_double=16).
    fn target_x86_64() -> TargetConfig {
        TargetConfig::x86_64()
    }

    /// Creates a TargetConfig for i686 (ILP32: pointer=4, long=4, long_double=12).
    fn target_i686() -> TargetConfig {
        TargetConfig::i686()
    }

    /// Creates a TargetConfig for AArch64 (LP64: pointer=8, long=8, long_double=16).
    fn target_aarch64() -> TargetConfig {
        TargetConfig::aarch64()
    }

    /// Creates a TargetConfig for RISC-V 64 (LP64: pointer=8, long=8, long_double=16).
    fn target_riscv64() -> TargetConfig {
        TargetConfig::riscv64()
    }

    // -----------------------------------------------------------------------
    // TypeQualifiers tests
    // -----------------------------------------------------------------------

    #[test]
    fn qualifiers_default_is_empty() {
        let q = TypeQualifiers::default();
        assert!(q.is_empty());
        assert!(!q.is_const);
        assert!(!q.is_volatile);
        assert!(!q.is_restrict);
        assert!(!q.is_atomic);
    }

    #[test]
    fn qualifiers_const_not_empty() {
        let q = TypeQualifiers {
            is_const: true,
            ..TypeQualifiers::default()
        };
        assert!(!q.is_empty());
        assert!(q.is_const);
    }

    #[test]
    fn qualifiers_merge_union() {
        let a = TypeQualifiers {
            is_const: true,
            ..TypeQualifiers::default()
        };
        let b = TypeQualifiers {
            is_volatile: true,
            ..TypeQualifiers::default()
        };
        let merged = a.merge(&b);
        assert!(merged.is_const);
        assert!(merged.is_volatile);
        assert!(!merged.is_restrict);
        assert!(!merged.is_atomic);
    }

    #[test]
    fn qualifiers_contains() {
        let cv = TypeQualifiers {
            is_const: true,
            is_volatile: true,
            ..TypeQualifiers::default()
        };
        let c = TypeQualifiers {
            is_const: true,
            ..TypeQualifiers::default()
        };
        let v = TypeQualifiers {
            is_volatile: true,
            ..TypeQualifiers::default()
        };
        assert!(cv.contains(&c));
        assert!(cv.contains(&v));
        assert!(cv.contains(&cv));
        assert!(!c.contains(&cv));
        assert!(c.contains(&TypeQualifiers::default()));
    }

    #[test]
    fn qualifiers_display() {
        let q = TypeQualifiers {
            is_const: true,
            is_volatile: true,
            ..TypeQualifiers::default()
        };
        let s = format!("{}", q);
        assert!(s.contains("const"));
        assert!(s.contains("volatile"));
    }

    // -----------------------------------------------------------------------
    // IntegerKind tests
    // -----------------------------------------------------------------------

    #[test]
    fn integer_signedness() {
        assert!(IntegerKind::Char.is_signed());
        assert!(IntegerKind::SignedChar.is_signed());
        assert!(IntegerKind::Short.is_signed());
        assert!(IntegerKind::Int.is_signed());
        assert!(IntegerKind::Long.is_signed());
        assert!(IntegerKind::LongLong.is_signed());

        assert!(IntegerKind::Bool.is_unsigned());
        assert!(IntegerKind::UnsignedChar.is_unsigned());
        assert!(IntegerKind::UnsignedShort.is_unsigned());
        assert!(IntegerKind::UnsignedInt.is_unsigned());
        assert!(IntegerKind::UnsignedLong.is_unsigned());
        assert!(IntegerKind::UnsignedLongLong.is_unsigned());
    }

    #[test]
    fn integer_rank_ordering() {
        assert!(IntegerKind::Bool.rank() < IntegerKind::Char.rank());
        assert!(IntegerKind::Char.rank() < IntegerKind::Short.rank());
        assert!(IntegerKind::Short.rank() < IntegerKind::Int.rank());
        assert!(IntegerKind::Int.rank() < IntegerKind::Long.rank());
        assert!(IntegerKind::Long.rank() < IntegerKind::LongLong.rank());

        // Signed and unsigned variants share the same rank.
        assert_eq!(IntegerKind::Int.rank(), IntegerKind::UnsignedInt.rank());
        assert_eq!(IntegerKind::Long.rank(), IntegerKind::UnsignedLong.rank());
    }

    #[test]
    fn integer_to_unsigned() {
        assert_eq!(IntegerKind::Int.to_unsigned(), IntegerKind::UnsignedInt);
        assert_eq!(IntegerKind::Long.to_unsigned(), IntegerKind::UnsignedLong);
        assert_eq!(
            IntegerKind::LongLong.to_unsigned(),
            IntegerKind::UnsignedLongLong
        );
        assert_eq!(IntegerKind::Char.to_unsigned(), IntegerKind::UnsignedChar);
        // Already unsigned: unchanged.
        assert_eq!(
            IntegerKind::UnsignedInt.to_unsigned(),
            IntegerKind::UnsignedInt
        );
    }

    #[test]
    fn integer_to_signed() {
        assert_eq!(IntegerKind::UnsignedInt.to_signed(), IntegerKind::Int);
        assert_eq!(IntegerKind::UnsignedLong.to_signed(), IntegerKind::Long);
        assert_eq!(IntegerKind::Int.to_signed(), IntegerKind::Int);
    }

    #[test]
    fn integer_sizes_x86_64() {
        let t = target_x86_64();
        assert_eq!(IntegerKind::Bool.size(&t), 1);
        assert_eq!(IntegerKind::Char.size(&t), 1);
        assert_eq!(IntegerKind::Short.size(&t), 2);
        assert_eq!(IntegerKind::Int.size(&t), 4);
        assert_eq!(IntegerKind::Long.size(&t), 8);
        assert_eq!(IntegerKind::LongLong.size(&t), 8);
    }

    #[test]
    fn integer_sizes_i686() {
        let t = target_i686();
        assert_eq!(IntegerKind::Bool.size(&t), 1);
        assert_eq!(IntegerKind::Int.size(&t), 4);
        assert_eq!(IntegerKind::Long.size(&t), 4); // ILP32: long is 4 bytes
        assert_eq!(IntegerKind::LongLong.size(&t), 8);
    }

    #[test]
    fn integer_sizes_all_targets() {
        for target in &[
            target_x86_64(),
            target_i686(),
            target_aarch64(),
            target_riscv64(),
        ] {
            assert_eq!(IntegerKind::Int.size(target), 4);
            assert_eq!(IntegerKind::LongLong.size(target), 8);
            assert_eq!(IntegerKind::Bool.size(target), 1);
            assert_eq!(IntegerKind::Short.size(target), 2);
        }
    }

    #[test]
    fn integer_alignment() {
        let t = target_x86_64();
        assert_eq!(IntegerKind::Int.alignment(&t), 4);
        assert_eq!(IntegerKind::Long.alignment(&t), 8);
        assert_eq!(IntegerKind::Char.alignment(&t), 1);
    }

    #[test]
    fn integer_display() {
        assert_eq!(format!("{}", IntegerKind::Int), "int");
        assert_eq!(format!("{}", IntegerKind::UnsignedLong), "unsigned long");
        assert_eq!(format!("{}", IntegerKind::Bool), "_Bool");
    }

    // -----------------------------------------------------------------------
    // FloatKind tests
    // -----------------------------------------------------------------------

    #[test]
    fn float_rank_ordering() {
        assert!(FloatKind::Float.rank() < FloatKind::Double.rank());
        assert!(FloatKind::Double.rank() < FloatKind::LongDouble.rank());
    }

    #[test]
    fn float_sizes() {
        let t64 = target_x86_64();
        let t32 = target_i686();
        assert_eq!(FloatKind::Float.size(&t64), 4);
        assert_eq!(FloatKind::Double.size(&t64), 8);
        assert_eq!(FloatKind::LongDouble.size(&t64), 16);
        assert_eq!(FloatKind::LongDouble.size(&t32), 12);
    }

    #[test]
    fn float_alignment() {
        let t = target_x86_64();
        assert_eq!(FloatKind::Float.alignment(&t), 4);
        assert_eq!(FloatKind::Double.alignment(&t), 8);
        assert_eq!(FloatKind::LongDouble.alignment(&t), 16);
    }

    #[test]
    fn float_display() {
        assert_eq!(format!("{}", FloatKind::Float), "float");
        assert_eq!(format!("{}", FloatKind::Double), "double");
        assert_eq!(format!("{}", FloatKind::LongDouble), "long double");
    }

    // -----------------------------------------------------------------------
    // CType — size and alignment tests
    // -----------------------------------------------------------------------

    #[test]
    fn ctype_void_size() {
        let t = target_x86_64();
        assert_eq!(CType::Void.size(&t), None);
        assert_eq!(CType::Void.alignment(&t), None);
    }

    #[test]
    fn ctype_integer_size() {
        let t = target_x86_64();
        assert_eq!(CType::Integer(IntegerKind::Int).size(&t), Some(4));
        assert_eq!(CType::Integer(IntegerKind::Long).size(&t), Some(8));
    }

    #[test]
    fn ctype_pointer_size() {
        let t64 = target_x86_64();
        let t32 = target_i686();
        let ptr = CType::Pointer {
            pointee: Box::new(CType::Integer(IntegerKind::Int)),
            qualifiers: TypeQualifiers::default(),
        };
        assert_eq!(ptr.size(&t64), Some(8));
        assert_eq!(ptr.size(&t32), Some(4));
    }

    #[test]
    fn ctype_array_size() {
        let t = target_x86_64();
        let arr = CType::Array {
            element: Box::new(CType::Integer(IntegerKind::Int)),
            size: ArraySize::Fixed(10),
        };
        assert_eq!(arr.size(&t), Some(40)); // 4 * 10

        let incomplete = CType::Array {
            element: Box::new(CType::Integer(IntegerKind::Int)),
            size: ArraySize::Incomplete,
        };
        assert_eq!(incomplete.size(&t), None);
    }

    #[test]
    fn ctype_enum_size() {
        let t = target_x86_64();
        let e = CType::Enum(EnumType {
            tag: Some("color".to_string()),
            variants: vec![("RED".to_string(), 0), ("GREEN".to_string(), 1)],
            is_complete: true,
        });
        assert_eq!(e.size(&t), Some(4)); // underlying int
    }

    #[test]
    fn ctype_function_size() {
        let t = target_x86_64();
        let func = CType::Function(FunctionType {
            return_type: Box::new(CType::Integer(IntegerKind::Int)),
            params: vec![],
            is_variadic: false,
            is_old_style: false,
        });
        assert_eq!(func.size(&t), None);
    }

    #[test]
    fn ctype_typedef_size() {
        let t = target_x86_64();
        let td = CType::Typedef {
            name: "size_t".to_string(),
            underlying: Box::new(CType::Integer(IntegerKind::UnsignedLong)),
        };
        assert_eq!(td.size(&t), Some(8));
    }

    #[test]
    fn ctype_qualified_size() {
        let t = target_x86_64();
        let cq = CType::Qualified {
            base: Box::new(CType::Integer(IntegerKind::Int)),
            qualifiers: TypeQualifiers {
                is_const: true,
                ..TypeQualifiers::default()
            },
        };
        assert_eq!(cq.size(&t), Some(4));
    }

    #[test]
    fn ctype_error_size() {
        let t = target_x86_64();
        assert_eq!(CType::Error.size(&t), None);
    }

    // -----------------------------------------------------------------------
    // CType — type classification tests
    // -----------------------------------------------------------------------

    #[test]
    fn classification_integer() {
        let ty = CType::Integer(IntegerKind::Int);
        assert!(ty.is_integer());
        assert!(ty.is_arithmetic());
        assert!(ty.is_scalar());
        assert!(!ty.is_float());
        assert!(!ty.is_pointer());
        assert!(!ty.is_array());
        assert!(!ty.is_function());
        assert!(!ty.is_void());
    }

    #[test]
    fn classification_float() {
        let ty = CType::Float(FloatKind::Double);
        assert!(ty.is_float());
        assert!(ty.is_arithmetic());
        assert!(ty.is_scalar());
        assert!(!ty.is_integer());
        assert!(!ty.is_pointer());
    }

    #[test]
    fn classification_pointer() {
        let ty = CType::Pointer {
            pointee: Box::new(CType::Integer(IntegerKind::Int)),
            qualifiers: TypeQualifiers::default(),
        };
        assert!(ty.is_pointer());
        assert!(ty.is_scalar());
        assert!(!ty.is_arithmetic());
        assert!(!ty.is_integer());
        assert!(!ty.is_float());
    }

    #[test]
    fn classification_void() {
        assert!(CType::Void.is_void());
        assert!(!CType::Void.is_complete());
    }

    #[test]
    fn classification_array() {
        let arr = CType::Array {
            element: Box::new(CType::Integer(IntegerKind::Int)),
            size: ArraySize::Fixed(5),
        };
        assert!(arr.is_array());
        assert!(arr.is_complete());

        let incomplete_arr = CType::Array {
            element: Box::new(CType::Integer(IntegerKind::Int)),
            size: ArraySize::Incomplete,
        };
        assert!(incomplete_arr.is_array());
        assert!(!incomplete_arr.is_complete());
    }

    #[test]
    fn classification_function() {
        let func = CType::Function(FunctionType {
            return_type: Box::new(CType::Void),
            params: vec![],
            is_variadic: false,
            is_old_style: false,
        });
        assert!(func.is_function());
        assert!(!func.is_complete());
    }

    #[test]
    fn classification_struct_union() {
        let st = CType::Struct(StructType {
            tag: Some("foo".to_string()),
            fields: vec![],
            is_union: false,
            is_packed: false,
            custom_alignment: None,
            is_complete: true,
        });
        assert!(st.is_struct());
        assert!(!st.is_union());

        let un = CType::Struct(StructType {
            tag: Some("bar".to_string()),
            fields: vec![],
            is_union: true,
            is_packed: false,
            custom_alignment: None,
            is_complete: true,
        });
        assert!(un.is_union());
        assert!(!un.is_struct());
    }

    #[test]
    fn classification_void_pointer() {
        let vp = CType::Pointer {
            pointee: Box::new(CType::Void),
            qualifiers: TypeQualifiers::default(),
        };
        assert!(vp.is_void_pointer());

        let ip = CType::Pointer {
            pointee: Box::new(CType::Integer(IntegerKind::Int)),
            qualifiers: TypeQualifiers::default(),
        };
        assert!(!ip.is_void_pointer());
    }

    #[test]
    fn classification_error() {
        assert!(CType::Error.is_error());
        assert!(!CType::Integer(IntegerKind::Int).is_error());
    }

    #[test]
    fn classification_enum_is_arithmetic() {
        let e = CType::Enum(EnumType {
            tag: Some("color".to_string()),
            variants: vec![],
            is_complete: true,
        });
        assert!(e.is_arithmetic());
        assert!(e.is_scalar());
    }

    // -----------------------------------------------------------------------
    // CType — unqualified and qualifiers tests
    // -----------------------------------------------------------------------

    #[test]
    fn unqualified_strips_qualifiers() {
        let q = CType::Qualified {
            base: Box::new(CType::Integer(IntegerKind::Int)),
            qualifiers: TypeQualifiers {
                is_const: true,
                ..TypeQualifiers::default()
            },
        };
        assert!(matches!(q.unqualified(), CType::Integer(IntegerKind::Int)));
    }

    #[test]
    fn qualifiers_from_qualified() {
        let q = CType::Qualified {
            base: Box::new(CType::Integer(IntegerKind::Int)),
            qualifiers: TypeQualifiers {
                is_const: true,
                is_volatile: true,
                ..TypeQualifiers::default()
            },
        };
        let quals = q.qualifiers();
        assert!(quals.is_const);
        assert!(quals.is_volatile);
    }

    #[test]
    fn qualifiers_from_unqualified() {
        let ty = CType::Integer(IntegerKind::Int);
        assert!(ty.qualifiers().is_empty());
    }

    // -----------------------------------------------------------------------
    // CType — decay tests
    // -----------------------------------------------------------------------

    #[test]
    fn decay_array_to_pointer() {
        let arr = CType::Array {
            element: Box::new(CType::Integer(IntegerKind::Int)),
            size: ArraySize::Fixed(10),
        };
        let decayed = arr.decay();
        match &decayed {
            CType::Pointer {
                pointee,
                qualifiers,
            } => {
                assert!(matches!(pointee.as_ref(), CType::Integer(IntegerKind::Int)));
                assert!(qualifiers.is_empty());
            }
            _ => panic!("Array did not decay to pointer"),
        }
    }

    #[test]
    fn decay_function_to_pointer() {
        let func = CType::Function(FunctionType {
            return_type: Box::new(CType::Integer(IntegerKind::Int)),
            params: vec![FunctionParam {
                name: Some("x".to_string()),
                ty: CType::Integer(IntegerKind::Int),
            }],
            is_variadic: false,
            is_old_style: false,
        });
        let decayed = func.decay();
        match &decayed {
            CType::Pointer { pointee, .. } => {
                assert!(matches!(pointee.as_ref(), CType::Function(_)));
            }
            _ => panic!("Function did not decay to pointer"),
        }
    }

    #[test]
    fn decay_int_unchanged() {
        let ty = CType::Integer(IntegerKind::Int);
        let decayed = ty.decay();
        assert!(matches!(decayed, CType::Integer(IntegerKind::Int)));
    }

    // -----------------------------------------------------------------------
    // CType — compatibility tests
    // -----------------------------------------------------------------------

    #[test]
    fn compatible_same_integer() {
        assert!(CType::Integer(IntegerKind::Int).is_compatible(&CType::Integer(IntegerKind::Int)));
    }

    #[test]
    fn compatible_different_integer() {
        assert!(!CType::Integer(IntegerKind::Int).is_compatible(&CType::Integer(IntegerKind::Long)));
    }

    #[test]
    fn compatible_pointer_to_int() {
        let p1 = CType::Pointer {
            pointee: Box::new(CType::Integer(IntegerKind::Int)),
            qualifiers: TypeQualifiers::default(),
        };
        let p2 = CType::Pointer {
            pointee: Box::new(CType::Integer(IntegerKind::Int)),
            qualifiers: TypeQualifiers::default(),
        };
        assert!(p1.is_compatible(&p2));
    }

    #[test]
    fn incompatible_pointer_types() {
        let p1 = CType::Pointer {
            pointee: Box::new(CType::Integer(IntegerKind::Int)),
            qualifiers: TypeQualifiers::default(),
        };
        let p2 = CType::Pointer {
            pointee: Box::new(CType::Integer(IntegerKind::Char)),
            qualifiers: TypeQualifiers::default(),
        };
        assert!(!p1.is_compatible(&p2));
    }

    #[test]
    fn compatible_enum_with_int() {
        let e = CType::Enum(EnumType {
            tag: Some("color".to_string()),
            variants: vec![],
            is_complete: true,
        });
        assert!(e.is_compatible(&CType::Integer(IntegerKind::Int)));
    }

    #[test]
    fn compatible_error_with_anything() {
        assert!(CType::Error.is_compatible(&CType::Integer(IntegerKind::Int)));
        assert!(CType::Integer(IntegerKind::Int).is_compatible(&CType::Error));
    }

    #[test]
    fn compatible_function_types() {
        let f1 = CType::Function(FunctionType {
            return_type: Box::new(CType::Integer(IntegerKind::Int)),
            params: vec![FunctionParam {
                name: None,
                ty: CType::Integer(IntegerKind::Int),
            }],
            is_variadic: false,
            is_old_style: false,
        });
        let f2 = CType::Function(FunctionType {
            return_type: Box::new(CType::Integer(IntegerKind::Int)),
            params: vec![FunctionParam {
                name: Some("x".to_string()),
                ty: CType::Integer(IntegerKind::Int),
            }],
            is_variadic: false,
            is_old_style: false,
        });
        assert!(f1.is_compatible(&f2));
    }

    #[test]
    fn incompatible_function_variadic_mismatch() {
        let f1 = CType::Function(FunctionType {
            return_type: Box::new(CType::Integer(IntegerKind::Int)),
            params: vec![],
            is_variadic: false,
            is_old_style: false,
        });
        let f2 = CType::Function(FunctionType {
            return_type: Box::new(CType::Integer(IntegerKind::Int)),
            params: vec![],
            is_variadic: true,
            is_old_style: false,
        });
        assert!(!f1.is_compatible(&f2));
    }

    // -----------------------------------------------------------------------
    // CType — is_same_type tests
    // -----------------------------------------------------------------------

    #[test]
    fn same_type_basic() {
        assert!(CType::Integer(IntegerKind::Int).is_same_type(&CType::Integer(IntegerKind::Int)));
        assert!(!CType::Integer(IntegerKind::Int).is_same_type(&CType::Integer(IntegerKind::Long)));
    }

    #[test]
    fn same_type_qualified() {
        let q1 = CType::Qualified {
            base: Box::new(CType::Integer(IntegerKind::Int)),
            qualifiers: TypeQualifiers {
                is_const: true,
                ..TypeQualifiers::default()
            },
        };
        let q2 = CType::Qualified {
            base: Box::new(CType::Integer(IntegerKind::Int)),
            qualifiers: TypeQualifiers {
                is_const: true,
                ..TypeQualifiers::default()
            },
        };
        let q3 = CType::Qualified {
            base: Box::new(CType::Integer(IntegerKind::Int)),
            qualifiers: TypeQualifiers {
                is_volatile: true,
                ..TypeQualifiers::default()
            },
        };
        assert!(q1.is_same_type(&q2));
        assert!(!q1.is_same_type(&q3));
    }

    // -----------------------------------------------------------------------
    // Struct layout computation tests
    // -----------------------------------------------------------------------

    #[test]
    fn struct_layout_simple() {
        // struct { int a; char b; int c; }
        // Expected: a@0, b@4, c@8, total size=12 (with 3 bytes padding after b)
        let t = target_x86_64();
        let mut s = StructType {
            tag: None,
            fields: vec![
                StructField {
                    name: Some("a".to_string()),
                    ty: CType::Integer(IntegerKind::Int),
                    bit_width: None,
                    offset: 0,
                },
                StructField {
                    name: Some("b".to_string()),
                    ty: CType::Integer(IntegerKind::Char),
                    bit_width: None,
                    offset: 0,
                },
                StructField {
                    name: Some("c".to_string()),
                    ty: CType::Integer(IntegerKind::Int),
                    bit_width: None,
                    offset: 0,
                },
            ],
            is_union: false,
            is_packed: false,
            custom_alignment: None,
            is_complete: true,
        };
        s.compute_layout(&t);
        assert_eq!(s.fields[0].offset, 0);
        assert_eq!(s.fields[1].offset, 4);
        assert_eq!(s.fields[2].offset, 8);

        let ty = CType::Struct(s);
        assert_eq!(ty.size(&t), Some(12));
    }

    #[test]
    fn union_layout() {
        // union { int a; double b; }
        // Expected: both at offset 0, size = 8 (max of 4, 8)
        let t = target_x86_64();
        let mut s = StructType {
            tag: None,
            fields: vec![
                StructField {
                    name: Some("a".to_string()),
                    ty: CType::Integer(IntegerKind::Int),
                    bit_width: None,
                    offset: 0,
                },
                StructField {
                    name: Some("b".to_string()),
                    ty: CType::Float(FloatKind::Double),
                    bit_width: None,
                    offset: 0,
                },
            ],
            is_union: true,
            is_packed: false,
            custom_alignment: None,
            is_complete: true,
        };
        s.compute_layout(&t);
        assert_eq!(s.fields[0].offset, 0);
        assert_eq!(s.fields[1].offset, 0);

        let ty = CType::Struct(s);
        assert_eq!(ty.size(&t), Some(8));
    }

    #[test]
    fn packed_struct_layout() {
        // __attribute__((packed)) struct { int a; char b; int c; }
        // Expected: a@0, b@4, c@5, total size=9 (no padding)
        let t = target_x86_64();
        let mut s = StructType {
            tag: None,
            fields: vec![
                StructField {
                    name: Some("a".to_string()),
                    ty: CType::Integer(IntegerKind::Int),
                    bit_width: None,
                    offset: 0,
                },
                StructField {
                    name: Some("b".to_string()),
                    ty: CType::Integer(IntegerKind::Char),
                    bit_width: None,
                    offset: 0,
                },
                StructField {
                    name: Some("c".to_string()),
                    ty: CType::Integer(IntegerKind::Int),
                    bit_width: None,
                    offset: 0,
                },
            ],
            is_union: false,
            is_packed: true,
            custom_alignment: None,
            is_complete: true,
        };
        s.compute_layout(&t);
        assert_eq!(s.fields[0].offset, 0);
        assert_eq!(s.fields[1].offset, 4);
        assert_eq!(s.fields[2].offset, 5);

        let ty = CType::Struct(s);
        assert_eq!(ty.size(&t), Some(9));
    }

    #[test]
    fn struct_with_custom_alignment() {
        // __attribute__((aligned(16))) struct { int a; }
        // Expected: size=16 (padded to alignment), alignment=16
        let t = target_x86_64();
        let mut s = StructType {
            tag: None,
            fields: vec![StructField {
                name: Some("a".to_string()),
                ty: CType::Integer(IntegerKind::Int),
                bit_width: None,
                offset: 0,
            }],
            is_union: false,
            is_packed: false,
            custom_alignment: Some(16),
            is_complete: true,
        };
        s.compute_layout(&t);
        assert_eq!(s.fields[0].offset, 0);

        let ty = CType::Struct(s);
        assert_eq!(ty.size(&t), Some(16));
        assert_eq!(ty.alignment(&t), Some(16));
    }

    #[test]
    fn forward_declared_struct_incomplete() {
        let t = target_x86_64();
        let s = CType::Struct(StructType {
            tag: Some("forward".to_string()),
            fields: vec![],
            is_union: false,
            is_packed: false,
            custom_alignment: None,
            is_complete: false,
        });
        assert_eq!(s.size(&t), None);
        assert!(!s.is_complete());
    }

    // -----------------------------------------------------------------------
    // CType — composite_type tests
    // -----------------------------------------------------------------------

    #[test]
    fn composite_array_incomplete_and_fixed() {
        let a = CType::Array {
            element: Box::new(CType::Integer(IntegerKind::Int)),
            size: ArraySize::Incomplete,
        };
        let b = CType::Array {
            element: Box::new(CType::Integer(IntegerKind::Int)),
            size: ArraySize::Fixed(10),
        };
        let composite = a.composite_type(&b);
        match composite {
            CType::Array { size, .. } => assert_eq!(size, ArraySize::Fixed(10)),
            _ => panic!("Expected array composite type"),
        }
    }

    // -----------------------------------------------------------------------
    // CType — Display tests
    // -----------------------------------------------------------------------

    #[test]
    fn display_basic_types() {
        assert_eq!(format!("{}", CType::Void), "void");
        assert_eq!(format!("{}", CType::Integer(IntegerKind::Int)), "int");
        assert_eq!(format!("{}", CType::Float(FloatKind::Double)), "double");
        assert_eq!(format!("{}", CType::Error), "<error>");
    }

    #[test]
    fn display_pointer() {
        let p = CType::Pointer {
            pointee: Box::new(CType::Integer(IntegerKind::Int)),
            qualifiers: TypeQualifiers::default(),
        };
        assert_eq!(format!("{}", p), "int *");
    }

    #[test]
    fn display_struct() {
        let s = CType::Struct(StructType {
            tag: Some("point".to_string()),
            fields: vec![],
            is_union: false,
            is_packed: false,
            custom_alignment: None,
            is_complete: true,
        });
        assert_eq!(format!("{}", s), "struct point");
    }

    #[test]
    fn display_enum() {
        let e = CType::Enum(EnumType {
            tag: Some("color".to_string()),
            variants: vec![],
            is_complete: true,
        });
        assert_eq!(format!("{}", e), "enum color");
    }

    #[test]
    fn display_function() {
        let f = CType::Function(FunctionType {
            return_type: Box::new(CType::Integer(IntegerKind::Int)),
            params: vec![
                FunctionParam {
                    name: None,
                    ty: CType::Integer(IntegerKind::Int),
                },
                FunctionParam {
                    name: None,
                    ty: CType::Float(FloatKind::Float),
                },
            ],
            is_variadic: false,
            is_old_style: false,
        });
        assert_eq!(format!("{}", f), "int(int, float)");
    }

    #[test]
    fn display_variadic_function() {
        let f = CType::Function(FunctionType {
            return_type: Box::new(CType::Integer(IntegerKind::Int)),
            params: vec![FunctionParam {
                name: None,
                ty: CType::Integer(IntegerKind::Int),
            }],
            is_variadic: true,
            is_old_style: false,
        });
        assert_eq!(format!("{}", f), "int(int, ...)");
    }

    #[test]
    fn display_qualified() {
        let q = CType::Qualified {
            base: Box::new(CType::Integer(IntegerKind::Int)),
            qualifiers: TypeQualifiers {
                is_const: true,
                ..TypeQualifiers::default()
            },
        };
        assert_eq!(format!("{}", q), "const int");
    }

    #[test]
    fn display_typedef() {
        let td = CType::Typedef {
            name: "size_t".to_string(),
            underlying: Box::new(CType::Integer(IntegerKind::UnsignedLong)),
        };
        assert_eq!(format!("{}", td), "size_t");
    }

    // -----------------------------------------------------------------------
    // align_up helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 4), 0);
        assert_eq!(align_up(1, 4), 4);
        assert_eq!(align_up(3, 4), 4);
        assert_eq!(align_up(4, 4), 4);
        assert_eq!(align_up(5, 4), 8);
        assert_eq!(align_up(5, 8), 8);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
    }

    // -----------------------------------------------------------------------
    // CType — typedef resolution through canonical
    // -----------------------------------------------------------------------

    #[test]
    fn typedef_is_integer() {
        let td = CType::Typedef {
            name: "my_int".to_string(),
            underlying: Box::new(CType::Integer(IntegerKind::Int)),
        };
        assert!(td.is_integer());
        assert!(td.is_arithmetic());
        assert!(td.is_scalar());
    }

    #[test]
    fn typedef_size() {
        let t = target_x86_64();
        let td = CType::Typedef {
            name: "my_long".to_string(),
            underlying: Box::new(CType::Integer(IntegerKind::Long)),
        };
        assert_eq!(td.size(&t), Some(8));
    }

    #[test]
    fn typeof_resolves() {
        let to = CType::TypeOf(Box::new(CType::Float(FloatKind::Double)));
        assert!(to.is_float());
        assert!(to.is_arithmetic());
    }
}
