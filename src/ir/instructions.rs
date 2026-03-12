//! IR instruction set definition for the `bcc` compiler's intermediate representation.
//!
//! This module defines all IR instruction types with typed operands and source
//! location tracking. The instruction set covers:
//!
//! - **Arithmetic**: `Add`, `Sub`, `Mul`, `Div`, `Mod`
//! - **Bitwise**: `And`, `Or`, `Xor`, `Shl`, `Shr`
//! - **Comparison**: `ICmp`, `FCmp`
//! - **Memory**: `Alloca`, `Load`, `Store`, `GetElementPtr`
//! - **Function**: `Call`
//! - **SSA**: `Phi`
//! - **Type Conversion**: `Cast`, `BitCast`
//! - **Miscellaneous**: `Select`, `Const`, `Copy`, `Nop`
//!
//! # SSA Value Model
//!
//! Every instruction that produces a result stores it in a [`Value`] — a lightweight
//! `u32` newtype that serves as a unique SSA value identifier. Values are assigned
//! sequentially within a function and referenced by subsequent instructions.
//!
//! # Source Location Tracking
//!
//! The [`LocatedInstruction`] wrapper pairs each instruction with a [`SourceLocation`]
//! from the original C source, enabling DWARF v4 debug info correlation.
//!
//! # Display Format
//!
//! Instructions are formatted in an LLVM-IR-inspired textual form:
//! ```text
//! %3 = add i32 %1, %2
//! %5 = load i32, i32* %4
//! store i32 %3, i32* %4
//! %7 = icmp eq i32 %5, %6
//! ```

use std::fmt;

use crate::common::source_map::SourceLocation;
use crate::ir::types::IrType;

// ---------------------------------------------------------------------------
// Value — SSA value reference
// ---------------------------------------------------------------------------

/// A unique identifier for an SSA value produced by an IR instruction.
///
/// `Value` is a lightweight `u32` newtype designed for cheap copying, hashing,
/// and embedding in instruction operands. Each instruction that produces a
/// result is assigned a unique `Value` within its containing function.
///
/// # Display
///
/// Values are formatted as `%N` (e.g., `%0`, `%1`, `%42`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Value(pub u32);

impl Value {
    /// Returns a sentinel value representing an undefined/poison SSA value.
    ///
    /// The undefined value uses `u32::MAX` as its index, which is guaranteed
    /// to never collide with a real value in any practical function (no function
    /// will have 4 billion+ SSA values).
    #[inline]
    pub fn undef() -> Self {
        Value(u32::MAX)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Value::undef() {
            write!(f, "undef")
        } else {
            write!(f, "%{}", self.0)
        }
    }
}

// ---------------------------------------------------------------------------
// BlockId — basic block identifier
// ---------------------------------------------------------------------------

/// A unique identifier for a basic block within a function's control flow graph.
///
/// `BlockId` is defined here (rather than in `cfg.rs`) to avoid circular
/// dependencies: instructions (specifically `Phi`) reference blocks, and
/// blocks contain instructions. By placing `BlockId` in the instruction
/// module, `cfg.rs` can import it without a cycle.
///
/// # Display
///
/// Blocks are formatted as `bb0`, `bb1`, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// CompareOp — integer comparison operations
// ---------------------------------------------------------------------------

/// Integer comparison operation for `ICmp` instructions.
///
/// Each variant corresponds to a specific comparison predicate. The `Signed*`
/// variants perform signed comparison, while `Unsigned*` variants treat
/// operands as unsigned integers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompareOp {
    /// Equal: `==`
    Equal,
    /// Not equal: `!=`
    NotEqual,
    /// Signed less than: `<`
    SignedLess,
    /// Signed less than or equal: `<=`
    SignedLessEqual,
    /// Signed greater than: `>`
    SignedGreater,
    /// Signed greater than or equal: `>=`
    SignedGreaterEqual,
    /// Unsigned less than: `<` (unsigned)
    UnsignedLess,
    /// Unsigned less than or equal: `<=` (unsigned)
    UnsignedLessEqual,
    /// Unsigned greater than: `>` (unsigned)
    UnsignedGreater,
    /// Unsigned greater than or equal: `>=` (unsigned)
    UnsignedGreaterEqual,
}

impl fmt::Display for CompareOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mnemonic = match self {
            CompareOp::Equal => "eq",
            CompareOp::NotEqual => "ne",
            CompareOp::SignedLess => "slt",
            CompareOp::SignedLessEqual => "sle",
            CompareOp::SignedGreater => "sgt",
            CompareOp::SignedGreaterEqual => "sge",
            CompareOp::UnsignedLess => "ult",
            CompareOp::UnsignedLessEqual => "ule",
            CompareOp::UnsignedGreater => "ugt",
            CompareOp::UnsignedGreaterEqual => "uge",
        };
        write!(f, "{}", mnemonic)
    }
}

// ---------------------------------------------------------------------------
// FloatCompareOp — floating-point comparison operations
// ---------------------------------------------------------------------------

/// Floating-point comparison operation for `FCmp` instructions.
///
/// IEEE 754 distinguishes between ordered and unordered comparisons:
/// - **Ordered** comparisons return `false` if either operand is NaN.
/// - **Unordered** comparisons return `true` if either operand is NaN.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatCompareOp {
    /// Ordered equal: `==` (false if NaN)
    OrderedEqual,
    /// Ordered not equal: `!=` (false if NaN)
    OrderedNotEqual,
    /// Ordered less than: `<` (false if NaN)
    OrderedLess,
    /// Ordered less than or equal: `<=` (false if NaN)
    OrderedLessEqual,
    /// Ordered greater than: `>` (false if NaN)
    OrderedGreater,
    /// Ordered greater than or equal: `>=` (false if NaN)
    OrderedGreaterEqual,
    /// Unordered: true if either operand is NaN
    Unordered,
    /// Unordered equal: `==` (true if NaN)
    UnorderedEqual,
}

impl fmt::Display for FloatCompareOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mnemonic = match self {
            FloatCompareOp::OrderedEqual => "oeq",
            FloatCompareOp::OrderedNotEqual => "one",
            FloatCompareOp::OrderedLess => "olt",
            FloatCompareOp::OrderedLessEqual => "ole",
            FloatCompareOp::OrderedGreater => "ogt",
            FloatCompareOp::OrderedGreaterEqual => "oge",
            FloatCompareOp::Unordered => "uno",
            FloatCompareOp::UnorderedEqual => "ueq",
        };
        write!(f, "{}", mnemonic)
    }
}

// ---------------------------------------------------------------------------
// CastOp — type conversion operations
// ---------------------------------------------------------------------------

/// Type conversion operation for `Cast` instructions.
///
/// Each variant describes the specific kind of type conversion being performed,
/// following the LLVM IR cast taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CastOp {
    /// Truncate integer to narrower type (e.g., i64 → i32).
    Trunc,
    /// Zero-extend integer to wider type (e.g., u8 → u32).
    ZExt,
    /// Sign-extend integer to wider type (e.g., i8 → i32).
    SExt,
    /// Convert floating-point to unsigned integer.
    FPToUI,
    /// Convert floating-point to signed integer.
    FPToSI,
    /// Convert unsigned integer to floating-point.
    UIToFP,
    /// Convert signed integer to floating-point.
    SIToFP,
    /// Truncate floating-point (e.g., double → float).
    FPTrunc,
    /// Extend floating-point (e.g., float → double).
    FPExt,
    /// Convert pointer to integer.
    PtrToInt,
    /// Convert integer to pointer.
    IntToPtr,
}

impl fmt::Display for CastOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            CastOp::Trunc => "trunc",
            CastOp::ZExt => "zext",
            CastOp::SExt => "sext",
            CastOp::FPToUI => "fptoui",
            CastOp::FPToSI => "fptosi",
            CastOp::UIToFP => "uitofp",
            CastOp::SIToFP => "sitofp",
            CastOp::FPTrunc => "fptrunc",
            CastOp::FPExt => "fpext",
            CastOp::PtrToInt => "ptrtoint",
            CastOp::IntToPtr => "inttoptr",
        };
        write!(f, "{}", name)
    }
}

// ---------------------------------------------------------------------------
// Callee — function call target
// ---------------------------------------------------------------------------

/// Represents the target of a function call instruction.
///
/// Calls can be either direct (to a named function) or indirect (through a
/// function pointer value).
#[derive(Debug, Clone, PartialEq)]
pub enum Callee {
    /// Direct call to a named function (e.g., `@printf`).
    Direct(String),
    /// Indirect call through a function pointer SSA value.
    Indirect(Value),
}

impl fmt::Display for Callee {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Callee::Direct(name) => write!(f, "@{}", name),
            Callee::Indirect(val) => write!(f, "{}", val),
        }
    }
}

// ---------------------------------------------------------------------------
// Constant — compile-time known values
// ---------------------------------------------------------------------------

/// A compile-time constant value used in `Const` instructions.
///
/// Constants represent values that are fully known at compile time, including
/// integer and floating-point literals, null pointers, undefined values,
/// zero-initialized aggregates, string literals, and references to global symbols.
#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    /// Integer constant with an explicit IR type (e.g., `i32 42`).
    Integer {
        /// The integer value (stored as i64 to accommodate all C integer types).
        value: i64,
        /// The IR type of this constant (must be an integer type).
        ty: IrType,
    },
    /// Floating-point constant with an explicit IR type (e.g., `f64 3.14`).
    Float {
        /// The floating-point value (stored as f64 to accommodate both float and double).
        value: f64,
        /// The IR type of this constant (must be `F32` or `F64`).
        ty: IrType,
    },
    /// Boolean constant (`true` or `false`).
    Bool(bool),
    /// Null pointer of a given pointer type.
    Null(IrType),
    /// Undefined value of a given type (analogous to LLVM `undef`).
    Undef(IrType),
    /// Zero-initialized aggregate of a given type.
    ZeroInit(IrType),
    /// String literal bytes, including null terminator.
    String(Vec<u8>),
    /// Reference to a global symbol by name.
    GlobalRef(String),
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Constant::Integer { value, ty } => write!(f, "{} {}", ty, value),
            Constant::Float { value, ty } => {
                // Format with enough precision to round-trip through parsing.
                write!(f, "{} {:.17e}", ty, value)
            }
            Constant::Bool(true) => write!(f, "i1 true"),
            Constant::Bool(false) => write!(f, "i1 false"),
            Constant::Null(ty) => write!(f, "{} null", ty),
            Constant::Undef(ty) => write!(f, "{} undef", ty),
            Constant::ZeroInit(ty) => write!(f, "{} zeroinitializer", ty),
            Constant::String(bytes) => {
                write!(f, "c\"")?;
                for &b in bytes {
                    if b.is_ascii_graphic() || b == b' ' {
                        write!(f, "{}", b as char)?;
                    } else {
                        write!(f, "\\{:02X}", b)?;
                    }
                }
                write!(f, "\"")
            }
            Constant::GlobalRef(name) => write!(f, "@{}", name),
        }
    }
}

// ---------------------------------------------------------------------------
// Instruction — the core IR instruction enum
// ---------------------------------------------------------------------------

/// The central IR instruction enum for the `bcc` compiler.
///
/// Each variant represents a single SSA instruction with typed operands. Most
/// variants produce a `result` value (the SSA definition) and consume one or
/// more operand values.
///
/// # Categories
///
/// | Category       | Variants                                         |
/// |----------------|--------------------------------------------------|
/// | Arithmetic     | `Add`, `Sub`, `Mul`, `Div`, `Mod`                |
/// | Bitwise        | `And`, `Or`, `Xor`, `Shl`, `Shr`                |
/// | Comparison     | `ICmp`, `FCmp`                                   |
/// | Memory         | `Alloca`, `Load`, `Store`, `GetElementPtr`       |
/// | Function       | `Call`                                           |
/// | SSA            | `Phi`                                            |
/// | Conversion     | `Cast`, `BitCast`                                |
/// | Miscellaneous  | `Select`, `Const`, `Copy`, `Nop`                 |
///
/// # Terminators
///
/// Branch, conditional branch, return, switch, and unreachable are **not**
/// represented here — they are defined separately as `Terminator` in `cfg.rs`.
/// `Instruction::is_terminator()` always returns `false`.
#[derive(Debug, Clone)]
pub enum Instruction {
    // === Arithmetic Operations ===
    /// Integer/float addition: `result = lhs + rhs`.
    Add {
        result: Value,
        lhs: Value,
        rhs: Value,
        ty: IrType,
    },
    /// Integer/float subtraction: `result = lhs - rhs`.
    Sub {
        result: Value,
        lhs: Value,
        rhs: Value,
        ty: IrType,
    },
    /// Integer/float multiplication: `result = lhs * rhs`.
    Mul {
        result: Value,
        lhs: Value,
        rhs: Value,
        ty: IrType,
    },
    /// Integer division: `result = lhs / rhs`.
    ///
    /// `is_signed` controls whether signed or unsigned division is performed.
    Div {
        result: Value,
        lhs: Value,
        rhs: Value,
        ty: IrType,
        is_signed: bool,
    },
    /// Integer modulo: `result = lhs % rhs`.
    ///
    /// `is_signed` controls whether signed or unsigned remainder is computed.
    Mod {
        result: Value,
        lhs: Value,
        rhs: Value,
        ty: IrType,
        is_signed: bool,
    },

    // === Bitwise Operations ===
    /// Bitwise AND: `result = lhs & rhs`.
    And {
        result: Value,
        lhs: Value,
        rhs: Value,
        ty: IrType,
    },
    /// Bitwise OR: `result = lhs | rhs`.
    Or {
        result: Value,
        lhs: Value,
        rhs: Value,
        ty: IrType,
    },
    /// Bitwise XOR: `result = lhs ^ rhs`.
    Xor {
        result: Value,
        lhs: Value,
        rhs: Value,
        ty: IrType,
    },
    /// Left shift: `result = lhs << rhs`.
    Shl {
        result: Value,
        lhs: Value,
        rhs: Value,
        ty: IrType,
    },
    /// Right shift: `result = lhs >> rhs`.
    ///
    /// When `is_arithmetic` is `true`, sign bits are preserved (arithmetic shift).
    /// When `false`, zeros are shifted in (logical shift).
    Shr {
        result: Value,
        lhs: Value,
        rhs: Value,
        ty: IrType,
        is_arithmetic: bool,
    },

    // === Comparison Operations ===
    /// Integer comparison producing an `i1` result.
    ///
    /// `ty` is the type of the *operands* being compared (not the result,
    /// which is always `i1`).
    ICmp {
        result: Value,
        op: CompareOp,
        lhs: Value,
        rhs: Value,
        ty: IrType,
    },
    /// Floating-point comparison producing an `i1` result.
    ///
    /// `ty` is the type of the *operands* being compared.
    FCmp {
        result: Value,
        op: FloatCompareOp,
        lhs: Value,
        rhs: Value,
        ty: IrType,
    },

    // === Memory Operations ===
    /// Stack allocation: `result = alloca ty [, count]`.
    ///
    /// Allocates space for one (or `count`) elements of type `ty` on the stack.
    /// The result is a pointer to the allocated memory.
    Alloca {
        result: Value,
        ty: IrType,
        count: Option<Value>,
    },
    /// Memory load: `result = load ty, ptr`.
    ///
    /// Reads a value of type `ty` from the memory location pointed to by `ptr`.
    Load {
        result: Value,
        ty: IrType,
        ptr: Value,
    },
    /// Memory store: `store value, ptr`.
    ///
    /// Writes `value` to the memory location pointed to by `ptr`. This
    /// instruction produces no result value.
    Store {
        value: Value,
        ptr: Value,
        /// Optional store type hint. When present, the code generator should
        /// use this type's width for the memory write (e.g., 32-bit store for
        /// I32 vs 64-bit store for I64/Pointer). When absent, the code
        /// generator uses its default width (typically 64-bit on x86-64).
        store_ty: Option<IrType>,
    },
    /// Element pointer computation: `result = getelementptr base_ty, ptr, indices...`.
    ///
    /// Computes the address of a sub-element of an aggregate type (struct field
    /// or array element) without actually performing a memory access.
    GetElementPtr {
        result: Value,
        base_ty: IrType,
        ptr: Value,
        indices: Vec<Value>,
        in_bounds: bool,
    },

    // === Function Operations ===
    /// Function call: `[result =] call return_ty callee(args...)`.
    ///
    /// `result` is `None` for void-returning functions.
    Call {
        result: Option<Value>,
        callee: Callee,
        args: Vec<Value>,
        return_ty: IrType,
    },

    // === SSA Operations ===
    /// Phi node: `result = phi ty [val1, block1], [val2, block2], ...`.
    ///
    /// Merges values from different control-flow predecessors at a join point.
    /// Each `(Value, BlockId)` pair specifies the value contributed by that
    /// predecessor block.
    Phi {
        result: Value,
        ty: IrType,
        incoming: Vec<(Value, BlockId)>,
    },

    // === Type Conversion Operations ===
    /// Type cast: `result = op value from from_ty to to_ty`.
    ///
    /// Performs a value-changing type conversion (e.g., integer truncation,
    /// sign extension, float-to-int).
    Cast {
        result: Value,
        op: CastOp,
        value: Value,
        from_ty: IrType,
        to_ty: IrType,
    },
    /// Bit-level reinterpretation: `result = bitcast value from from_ty to to_ty`.
    ///
    /// Reinterprets the bit pattern of a value as a different type without
    /// changing the bits. Used primarily for pointer casts.
    BitCast {
        result: Value,
        value: Value,
        from_ty: IrType,
        to_ty: IrType,
    },

    // === Miscellaneous ===
    /// Select (ternary): `result = select condition, true_val, false_val`.
    ///
    /// Evaluates to `true_val` if `condition` is nonzero, otherwise `false_val`.
    /// This is the IR equivalent of the C ternary operator `?:` when both arms
    /// are simple values.
    Select {
        result: Value,
        condition: Value,
        true_val: Value,
        false_val: Value,
        ty: IrType,
    },
    /// Load a constant value: `result = const value`.
    ///
    /// Materializes a compile-time constant as an SSA value.
    Const { result: Value, value: Constant },
    /// Copy instruction: `result = copy source`.
    ///
    /// Used during SSA destruction (phi elimination) to insert parallel copies
    /// on predecessor edges. The `ty` field records the type being copied.
    Copy {
        result: Value,
        source: Value,
        ty: IrType,
    },
    /// No-op placeholder instruction.
    ///
    /// May be inserted temporarily during IR construction and removed by
    /// optimization passes. Produces no result and has no effect.
    Nop,
}

// ---------------------------------------------------------------------------
// Instruction — helper methods
// ---------------------------------------------------------------------------

impl Instruction {
    /// Returns the result value produced by this instruction, or `None` if the
    /// instruction does not produce a value (`Store`, `Nop`).
    pub fn result(&self) -> Option<Value> {
        match self {
            Instruction::Add { result, .. }
            | Instruction::Sub { result, .. }
            | Instruction::Mul { result, .. }
            | Instruction::Div { result, .. }
            | Instruction::Mod { result, .. }
            | Instruction::And { result, .. }
            | Instruction::Or { result, .. }
            | Instruction::Xor { result, .. }
            | Instruction::Shl { result, .. }
            | Instruction::Shr { result, .. }
            | Instruction::ICmp { result, .. }
            | Instruction::FCmp { result, .. }
            | Instruction::Alloca { result, .. }
            | Instruction::Load { result, .. }
            | Instruction::GetElementPtr { result, .. }
            | Instruction::Phi { result, .. }
            | Instruction::Cast { result, .. }
            | Instruction::BitCast { result, .. }
            | Instruction::Select { result, .. }
            | Instruction::Const { result, .. }
            | Instruction::Copy { result, .. } => Some(*result),
            Instruction::Call { result, .. } => *result,
            Instruction::Store { .. } | Instruction::Nop => None,
        }
    }

    /// Returns a vector of all operand values consumed by this instruction.
    ///
    /// This does **not** include the result value. For `Phi` instructions, the
    /// incoming values (but not block IDs) are included.
    pub fn operands(&self) -> Vec<Value> {
        match self {
            Instruction::Add { lhs, rhs, .. }
            | Instruction::Sub { lhs, rhs, .. }
            | Instruction::Mul { lhs, rhs, .. }
            | Instruction::Div { lhs, rhs, .. }
            | Instruction::Mod { lhs, rhs, .. }
            | Instruction::And { lhs, rhs, .. }
            | Instruction::Or { lhs, rhs, .. }
            | Instruction::Xor { lhs, rhs, .. }
            | Instruction::Shl { lhs, rhs, .. }
            | Instruction::Shr { lhs, rhs, .. }
            | Instruction::ICmp { lhs, rhs, .. }
            | Instruction::FCmp { lhs, rhs, .. } => {
                vec![*lhs, *rhs]
            }
            Instruction::Alloca { count, .. } => count.iter().copied().collect(),
            Instruction::Load { ptr, .. } => vec![*ptr],
            Instruction::Store { value, ptr, .. } => vec![*value, *ptr],
            Instruction::GetElementPtr { ptr, indices, .. } => {
                let mut ops = vec![*ptr];
                ops.extend(indices.iter().copied());
                ops
            }
            Instruction::Call { callee, args, .. } => {
                let mut ops: Vec<Value> = args.clone();
                if let Callee::Indirect(v) = callee {
                    ops.push(*v);
                }
                ops
            }
            Instruction::Phi { incoming, .. } => incoming.iter().map(|(v, _)| *v).collect(),
            Instruction::Cast { value, .. } | Instruction::BitCast { value, .. } => vec![*value],
            Instruction::Select {
                condition,
                true_val,
                false_val,
                ..
            } => vec![*condition, *true_val, *false_val],
            Instruction::Const { .. } => vec![],
            Instruction::Copy { source, .. } => vec![*source],
            Instruction::Nop => vec![],
        }
    }

    /// Returns mutable references to all operand values consumed by this
    /// instruction.
    ///
    /// This is used during SSA renaming and value replacement passes to update
    /// operands in-place.
    pub fn operands_mut(&mut self) -> Vec<&mut Value> {
        match self {
            Instruction::Add { lhs, rhs, .. }
            | Instruction::Sub { lhs, rhs, .. }
            | Instruction::Mul { lhs, rhs, .. }
            | Instruction::Div { lhs, rhs, .. }
            | Instruction::Mod { lhs, rhs, .. }
            | Instruction::And { lhs, rhs, .. }
            | Instruction::Or { lhs, rhs, .. }
            | Instruction::Xor { lhs, rhs, .. }
            | Instruction::Shl { lhs, rhs, .. }
            | Instruction::Shr { lhs, rhs, .. }
            | Instruction::ICmp { lhs, rhs, .. }
            | Instruction::FCmp { lhs, rhs, .. } => {
                vec![lhs, rhs]
            }
            Instruction::Alloca { count, .. } => count.iter_mut().collect(),
            Instruction::Load { ptr, .. } => vec![ptr],
            Instruction::Store { value, ptr, .. } => vec![value, ptr],
            Instruction::GetElementPtr { ptr, indices, .. } => {
                let mut ops = vec![ptr];
                ops.extend(indices.iter_mut());
                ops
            }
            Instruction::Call { callee, args, .. } => {
                let mut ops: Vec<&mut Value> = args.iter_mut().collect();
                if let Callee::Indirect(v) = callee {
                    ops.push(v);
                }
                ops
            }
            Instruction::Phi { incoming, .. } => incoming.iter_mut().map(|(v, _)| v).collect(),
            Instruction::Cast { value, .. } | Instruction::BitCast { value, .. } => vec![value],
            Instruction::Select {
                condition,
                true_val,
                false_val,
                ..
            } => vec![condition, true_val, false_val],
            Instruction::Const { .. } => vec![],
            Instruction::Copy { source, .. } => vec![source],
            Instruction::Nop => vec![],
        }
    }

    /// Returns the IR type of the result value, or `None` if the instruction
    /// does not produce a value.
    ///
    /// For comparison instructions (`ICmp`, `FCmp`), the result type is always
    /// `I1` (boolean), not the operand type.
    pub fn result_type(&self) -> Option<&IrType> {
        match self {
            Instruction::Add { ty, .. }
            | Instruction::Sub { ty, .. }
            | Instruction::Mul { ty, .. }
            | Instruction::Div { ty, .. }
            | Instruction::Mod { ty, .. }
            | Instruction::And { ty, .. }
            | Instruction::Or { ty, .. }
            | Instruction::Xor { ty, .. }
            | Instruction::Shl { ty, .. }
            | Instruction::Shr { ty, .. }
            | Instruction::Phi { ty, .. }
            | Instruction::Select { ty, .. }
            | Instruction::Copy { ty, .. } => Some(ty),
            // ICmp and FCmp produce i1 — we return the operand type field
            // but callers should know the result is always i1.
            Instruction::ICmp { ty, .. } | Instruction::FCmp { ty, .. } => Some(ty),
            Instruction::Alloca { ty, .. } => Some(ty),
            Instruction::Load { ty, .. } => Some(ty),
            Instruction::GetElementPtr { base_ty, .. } => Some(base_ty),
            Instruction::Cast { to_ty, .. } | Instruction::BitCast { to_ty, .. } => Some(to_ty),
            Instruction::Call { return_ty, .. } => Some(return_ty),
            Instruction::Const { value, .. } => match value {
                Constant::Integer { ty, .. } => Some(ty),
                Constant::Float { ty, .. } => Some(ty),
                Constant::Null(ty) | Constant::Undef(ty) | Constant::ZeroInit(ty) => Some(ty),
                // Bool, String, GlobalRef don't carry a direct IrType field.
                Constant::Bool(_) | Constant::String(_) | Constant::GlobalRef(_) => None,
            },
            Instruction::Store { .. } | Instruction::Nop => None,
        }
    }

    /// Returns `false` — `Instruction` variants are never terminators.
    ///
    /// Terminators (branch, conditional branch, return, switch, unreachable)
    /// are defined separately as `Terminator` in the `cfg` module.
    #[inline]
    pub fn is_terminator(&self) -> bool {
        false
    }

    /// Returns `true` if this instruction has observable side effects beyond
    /// producing a result value.
    ///
    /// Instructions with side effects cannot be removed by dead code elimination
    /// even if their result is unused.
    ///
    /// Side-effecting instructions: `Store`, `Call`.
    pub fn has_side_effects(&self) -> bool {
        matches!(self, Instruction::Store { .. } | Instruction::Call { .. })
    }

    /// Returns `true` if this instruction performs a memory operation.
    ///
    /// Memory operations: `Alloca`, `Load`, `Store`.
    pub fn is_memory_operation(&self) -> bool {
        matches!(
            self,
            Instruction::Alloca { .. } | Instruction::Load { .. } | Instruction::Store { .. }
        )
    }

    /// Returns `true` if this instruction uses (reads) the given value as an
    /// operand.
    pub fn uses_value(&self, value: Value) -> bool {
        self.operands().contains(&value)
    }

    /// Replaces all occurrences of `old` with `new` in this instruction's
    /// operands.
    ///
    /// This is used during SSA renaming, value substitution, and copy
    /// propagation passes.
    pub fn replace_use(&mut self, old: Value, new: Value) {
        for operand in self.operands_mut() {
            if *operand == old {
                *operand = new;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Instruction — Display
// ---------------------------------------------------------------------------

impl fmt::Display for Instruction {
    /// Formats the instruction in LLVM-IR-inspired textual form.
    ///
    /// # Examples
    ///
    /// ```text
    /// %3 = add i32 %1, %2
    /// %5 = load i32, i32* %4
    /// store i32 %3, i32* %4
    /// %7 = icmp eq i32 %5, %6
    /// %9 = call i32 @printf(%8)
    /// %10 = phi i32 [%3, bb1], [%5, bb2]
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // --- Arithmetic (basic binary with shared format) ---
            Instruction::Add {
                result,
                lhs,
                rhs,
                ty,
            } => {
                write!(f, "{} = add {} {}, {}", result, ty, lhs, rhs)
            }
            Instruction::Sub {
                result,
                lhs,
                rhs,
                ty,
            } => {
                write!(f, "{} = sub {} {}, {}", result, ty, lhs, rhs)
            }
            Instruction::Mul {
                result,
                lhs,
                rhs,
                ty,
            } => {
                write!(f, "{} = mul {} {}, {}", result, ty, lhs, rhs)
            }
            Instruction::Div {
                result,
                lhs,
                rhs,
                ty,
                is_signed,
            } => {
                let op = if *is_signed { "sdiv" } else { "udiv" };
                write!(f, "{} = {} {} {}, {}", result, op, ty, lhs, rhs)
            }
            Instruction::Mod {
                result,
                lhs,
                rhs,
                ty,
                is_signed,
            } => {
                let op = if *is_signed { "srem" } else { "urem" };
                write!(f, "{} = {} {} {}, {}", result, op, ty, lhs, rhs)
            }

            // --- Bitwise ---
            Instruction::And {
                result,
                lhs,
                rhs,
                ty,
            } => {
                write!(f, "{} = and {} {}, {}", result, ty, lhs, rhs)
            }
            Instruction::Or {
                result,
                lhs,
                rhs,
                ty,
            } => {
                write!(f, "{} = or {} {}, {}", result, ty, lhs, rhs)
            }
            Instruction::Xor {
                result,
                lhs,
                rhs,
                ty,
            } => {
                write!(f, "{} = xor {} {}, {}", result, ty, lhs, rhs)
            }
            Instruction::Shl {
                result,
                lhs,
                rhs,
                ty,
            } => {
                write!(f, "{} = shl {} {}, {}", result, ty, lhs, rhs)
            }
            Instruction::Shr {
                result,
                lhs,
                rhs,
                ty,
                is_arithmetic,
            } => {
                let op = if *is_arithmetic { "ashr" } else { "lshr" };
                write!(f, "{} = {} {} {}, {}", result, op, ty, lhs, rhs)
            }

            // --- Comparisons ---
            Instruction::ICmp {
                result,
                op,
                lhs,
                rhs,
                ty,
            } => {
                write!(f, "{} = icmp {} {} {}, {}", result, op, ty, lhs, rhs)
            }
            Instruction::FCmp {
                result,
                op,
                lhs,
                rhs,
                ty,
            } => {
                write!(f, "{} = fcmp {} {} {}, {}", result, op, ty, lhs, rhs)
            }

            // --- Memory ---
            Instruction::Alloca { result, ty, count } => {
                write!(f, "{} = alloca {}", result, ty)?;
                if let Some(c) = count {
                    write!(f, ", {}", c)?;
                }
                Ok(())
            }
            Instruction::Load { result, ty, ptr } => {
                write!(f, "{} = load {}, {}* {}", result, ty, ty, ptr)
            }
            Instruction::Store { value, ptr, .. } => {
                write!(f, "store {}, {}", value, ptr)
            }
            Instruction::GetElementPtr {
                result,
                base_ty,
                ptr,
                indices,
                in_bounds,
            } => {
                write!(f, "{} = getelementptr ", result)?;
                if *in_bounds {
                    write!(f, "inbounds ")?;
                }
                write!(f, "{}, {}* {}", base_ty, base_ty, ptr)?;
                for idx in indices {
                    write!(f, ", {}", idx)?;
                }
                Ok(())
            }

            // --- Function call ---
            Instruction::Call {
                result,
                callee,
                args,
                return_ty,
            } => {
                if let Some(r) = result {
                    write!(f, "{} = ", r)?;
                }
                write!(f, "call {} {}(", return_ty, callee)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }

            // --- SSA phi ---
            Instruction::Phi {
                result,
                ty,
                incoming,
            } => {
                write!(f, "{} = phi {}", result, ty)?;
                for (i, (val, block)) in incoming.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, " [{}, {}]", val, block)?;
                }
                Ok(())
            }

            // --- Type conversions ---
            Instruction::Cast {
                result,
                op,
                value,
                from_ty,
                to_ty,
            } => {
                write!(f, "{} = {} {} {} to {}", result, op, from_ty, value, to_ty)
            }
            Instruction::BitCast {
                result,
                value,
                from_ty,
                to_ty,
            } => {
                write!(f, "{} = bitcast {} {} to {}", result, from_ty, value, to_ty)
            }

            // --- Miscellaneous ---
            Instruction::Select {
                result,
                condition,
                true_val,
                false_val,
                ty,
            } => {
                write!(
                    f,
                    "{} = select i1 {}, {} {}, {} {}",
                    result, condition, ty, true_val, ty, false_val
                )
            }
            Instruction::Const { result, value } => {
                write!(f, "{} = const {}", result, value)
            }
            Instruction::Copy { result, source, ty } => {
                write!(f, "{} = copy {} {}", result, ty, source)
            }
            Instruction::Nop => write!(f, "nop"),
        }
    }
}

// ---------------------------------------------------------------------------
// LocatedInstruction — instruction with source location
// ---------------------------------------------------------------------------

/// An instruction paired with its originating source location.
///
/// `LocatedInstruction` enables DWARF v4 debug info generation by mapping
/// each IR instruction back to the C source file position from which it was
/// generated.
#[derive(Debug, Clone)]
pub struct LocatedInstruction {
    /// The IR instruction.
    pub instruction: Instruction,
    /// The source position in the original C file.
    pub location: SourceLocation,
}

impl LocatedInstruction {
    /// Creates a new located instruction.
    #[inline]
    pub fn new(instruction: Instruction, location: SourceLocation) -> Self {
        LocatedInstruction {
            instruction,
            location,
        }
    }

    /// Creates a located instruction with a dummy (synthetic) source location.
    #[inline]
    pub fn synthetic(instruction: Instruction) -> Self {
        LocatedInstruction {
            instruction,
            location: SourceLocation::dummy(),
        }
    }
}

impl fmt::Display for LocatedInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.instruction)
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // -----------------------------------------------------------------------
    // Value tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_value_equality_same_id() {
        let v1 = Value(42);
        let v2 = Value(42);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_value_inequality_different_id() {
        let v1 = Value(1);
        let v2 = Value(2);
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_value_as_hashmap_key() {
        let mut map: HashMap<Value, &str> = HashMap::new();
        map.insert(Value(0), "zero");
        map.insert(Value(1), "one");
        assert_eq!(map.get(&Value(0)), Some(&"zero"));
        assert_eq!(map.get(&Value(1)), Some(&"one"));
        assert_eq!(map.get(&Value(2)), None);
    }

    #[test]
    fn test_value_copy_semantics() {
        let v1 = Value(10);
        let v2 = v1; // Copy
        assert_eq!(v1, v2); // v1 still valid
    }

    #[test]
    fn test_value_undef() {
        let undef = Value::undef();
        assert_eq!(undef, Value(u32::MAX));
        assert_ne!(undef, Value(0));
    }

    #[test]
    fn test_value_display() {
        assert_eq!(format!("{}", Value(0)), "%0");
        assert_eq!(format!("{}", Value(42)), "%42");
        assert_eq!(format!("{}", Value::undef()), "undef");
    }

    // -----------------------------------------------------------------------
    // BlockId tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_block_id_equality() {
        assert_eq!(BlockId(0), BlockId(0));
        assert_ne!(BlockId(0), BlockId(1));
    }

    #[test]
    fn test_block_id_display() {
        assert_eq!(format!("{}", BlockId(0)), "bb0");
        assert_eq!(format!("{}", BlockId(5)), "bb5");
    }

    // -----------------------------------------------------------------------
    // Constant tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_constant_integer_construction_and_equality() {
        let c1 = Constant::Integer {
            value: 42,
            ty: IrType::I32,
        };
        let c2 = Constant::Integer {
            value: 42,
            ty: IrType::I32,
        };
        assert_eq!(c1, c2);

        let c3 = Constant::Integer {
            value: 100,
            ty: IrType::I32,
        };
        assert_ne!(c1, c3);
    }

    #[test]
    fn test_constant_float_construction() {
        let c = Constant::Float {
            value: 3.14,
            ty: IrType::F64,
        };
        if let Constant::Float { value, ty } = &c {
            assert!((*value - 3.14).abs() < 1e-15);
            assert_eq!(*ty, IrType::F64);
        } else {
            panic!("Expected Float constant");
        }
    }

    #[test]
    fn test_constant_null_pointer() {
        let c = Constant::Null(IrType::Pointer(Box::new(IrType::I8)));
        if let Constant::Null(ty) = &c {
            assert!(ty.is_pointer());
        } else {
            panic!("Expected Null constant");
        }
    }

    #[test]
    fn test_constant_bool() {
        let t = Constant::Bool(true);
        let f = Constant::Bool(false);
        assert_ne!(t, f);
        assert_eq!(format!("{}", t), "i1 true");
        assert_eq!(format!("{}", f), "i1 false");
    }

    #[test]
    fn test_constant_display() {
        let int_c = Constant::Integer {
            value: 42,
            ty: IrType::I32,
        };
        assert_eq!(format!("{}", int_c), "i32 42");

        let null_c = Constant::Null(IrType::Pointer(Box::new(IrType::I8)));
        assert_eq!(format!("{}", null_c), "i8* null");

        let global_c = Constant::GlobalRef("main".to_string());
        assert_eq!(format!("{}", global_c), "@main");
    }

    // -----------------------------------------------------------------------
    // CompareOp tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compare_op_all_distinct() {
        let ops = [
            CompareOp::Equal,
            CompareOp::NotEqual,
            CompareOp::SignedLess,
            CompareOp::SignedLessEqual,
            CompareOp::SignedGreater,
            CompareOp::SignedGreaterEqual,
            CompareOp::UnsignedLess,
            CompareOp::UnsignedLessEqual,
            CompareOp::UnsignedGreater,
            CompareOp::UnsignedGreaterEqual,
        ];
        for i in 0..ops.len() {
            for j in (i + 1)..ops.len() {
                assert_ne!(ops[i], ops[j], "ops[{}] == ops[{}]", i, j);
            }
        }
    }

    #[test]
    fn test_compare_op_display() {
        assert_eq!(format!("{}", CompareOp::Equal), "eq");
        assert_eq!(format!("{}", CompareOp::NotEqual), "ne");
        assert_eq!(format!("{}", CompareOp::SignedLess), "slt");
        assert_eq!(format!("{}", CompareOp::SignedGreaterEqual), "sge");
        assert_eq!(format!("{}", CompareOp::UnsignedLess), "ult");
        assert_eq!(format!("{}", CompareOp::UnsignedGreaterEqual), "uge");
    }

    // -----------------------------------------------------------------------
    // FloatCompareOp tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_float_compare_op_all_distinct() {
        let ops = [
            FloatCompareOp::OrderedEqual,
            FloatCompareOp::OrderedNotEqual,
            FloatCompareOp::OrderedLess,
            FloatCompareOp::OrderedLessEqual,
            FloatCompareOp::OrderedGreater,
            FloatCompareOp::OrderedGreaterEqual,
            FloatCompareOp::Unordered,
            FloatCompareOp::UnorderedEqual,
        ];
        for i in 0..ops.len() {
            for j in (i + 1)..ops.len() {
                assert_ne!(ops[i], ops[j], "ops[{}] == ops[{}]", i, j);
            }
        }
    }

    #[test]
    fn test_float_compare_op_display() {
        assert_eq!(format!("{}", FloatCompareOp::OrderedEqual), "oeq");
        assert_eq!(format!("{}", FloatCompareOp::Unordered), "uno");
        assert_eq!(format!("{}", FloatCompareOp::UnorderedEqual), "ueq");
    }

    // -----------------------------------------------------------------------
    // CastOp tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cast_op_all_distinct() {
        let ops = [
            CastOp::Trunc,
            CastOp::ZExt,
            CastOp::SExt,
            CastOp::FPToUI,
            CastOp::FPToSI,
            CastOp::UIToFP,
            CastOp::SIToFP,
            CastOp::FPTrunc,
            CastOp::FPExt,
            CastOp::PtrToInt,
            CastOp::IntToPtr,
        ];
        for i in 0..ops.len() {
            for j in (i + 1)..ops.len() {
                assert_ne!(ops[i], ops[j], "ops[{}] == ops[{}]", i, j);
            }
        }
    }

    #[test]
    fn test_cast_op_display() {
        assert_eq!(format!("{}", CastOp::Trunc), "trunc");
        assert_eq!(format!("{}", CastOp::ZExt), "zext");
        assert_eq!(format!("{}", CastOp::SExt), "sext");
        assert_eq!(format!("{}", CastOp::FPToSI), "fptosi");
        assert_eq!(format!("{}", CastOp::SIToFP), "sitofp");
        assert_eq!(format!("{}", CastOp::PtrToInt), "ptrtoint");
        assert_eq!(format!("{}", CastOp::IntToPtr), "inttoptr");
    }

    // -----------------------------------------------------------------------
    // Callee tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_callee_display() {
        let direct = Callee::Direct("printf".to_string());
        assert_eq!(format!("{}", direct), "@printf");

        let indirect = Callee::Indirect(Value(5));
        assert_eq!(format!("{}", indirect), "%5");
    }

    // -----------------------------------------------------------------------
    // Instruction construction and method tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_instruction_result_and_operands() {
        let inst = Instruction::Add {
            result: Value(3),
            lhs: Value(1),
            rhs: Value(2),
            ty: IrType::I32,
        };
        assert_eq!(inst.result(), Some(Value(3)));
        assert_eq!(inst.operands(), vec![Value(1), Value(2)]);
    }

    #[test]
    fn test_load_is_memory_operation() {
        let inst = Instruction::Load {
            result: Value(5),
            ty: IrType::I32,
            ptr: Value(4),
        };
        assert!(inst.is_memory_operation());
        assert!(!inst.has_side_effects());
        assert_eq!(inst.result(), Some(Value(5)));
    }

    #[test]
    fn test_store_result_and_side_effects() {
        let inst = Instruction::Store {
            value: Value(3),
            ptr: Value(4),
            store_ty: None,
        };
        assert_eq!(inst.result(), None);
        assert!(inst.has_side_effects());
        assert!(inst.is_memory_operation());
        assert_eq!(inst.operands(), vec![Value(3), Value(4)]);
    }

    #[test]
    fn test_call_has_side_effects() {
        let inst = Instruction::Call {
            result: Some(Value(9)),
            callee: Callee::Direct("printf".to_string()),
            args: vec![Value(8)],
            return_ty: IrType::I32,
        };
        assert!(inst.has_side_effects());
        assert_eq!(inst.result(), Some(Value(9)));
    }

    #[test]
    fn test_void_call_result_none() {
        let inst = Instruction::Call {
            result: None,
            callee: Callee::Direct("abort".to_string()),
            args: vec![],
            return_ty: IrType::Void,
        };
        assert_eq!(inst.result(), None);
        assert!(inst.has_side_effects());
    }

    #[test]
    fn test_phi_instruction_incoming_values() {
        let inst = Instruction::Phi {
            result: Value(10),
            ty: IrType::I32,
            incoming: vec![(Value(3), BlockId(1)), (Value(5), BlockId(2))],
        };
        assert_eq!(inst.result(), Some(Value(10)));
        // Operands should contain incoming values but not block IDs.
        assert_eq!(inst.operands(), vec![Value(3), Value(5)]);
    }

    #[test]
    fn test_alloca_is_memory_operation() {
        let inst = Instruction::Alloca {
            result: Value(0),
            ty: IrType::I32,
            count: None,
        };
        assert!(inst.is_memory_operation());
        assert_eq!(inst.result(), Some(Value(0)));
        assert!(inst.operands().is_empty());
    }

    #[test]
    fn test_alloca_with_count() {
        let inst = Instruction::Alloca {
            result: Value(0),
            ty: IrType::I32,
            count: Some(Value(1)),
        };
        assert_eq!(inst.operands(), vec![Value(1)]);
    }

    #[test]
    fn test_nop_properties() {
        let inst = Instruction::Nop;
        assert_eq!(inst.result(), None);
        assert!(inst.operands().is_empty());
        assert!(!inst.has_side_effects());
        assert!(!inst.is_memory_operation());
        assert!(!inst.is_terminator());
    }

    #[test]
    fn test_is_terminator_always_false() {
        let instructions = vec![
            Instruction::Add {
                result: Value(0),
                lhs: Value(1),
                rhs: Value(2),
                ty: IrType::I32,
            },
            Instruction::Store {
                value: Value(0),
                ptr: Value(1),
                store_ty: None,
            },
            Instruction::Nop,
        ];
        for inst in &instructions {
            assert!(!inst.is_terminator());
        }
    }

    // -----------------------------------------------------------------------
    // operands() comprehensive test
    // -----------------------------------------------------------------------

    #[test]
    fn test_operands_binary_ops() {
        let ops = vec![
            Instruction::Sub {
                result: Value(5),
                lhs: Value(3),
                rhs: Value(4),
                ty: IrType::I32,
            },
            Instruction::Mul {
                result: Value(6),
                lhs: Value(3),
                rhs: Value(4),
                ty: IrType::I32,
            },
            Instruction::Div {
                result: Value(7),
                lhs: Value(3),
                rhs: Value(4),
                ty: IrType::I32,
                is_signed: true,
            },
            Instruction::Mod {
                result: Value(8),
                lhs: Value(3),
                rhs: Value(4),
                ty: IrType::I32,
                is_signed: false,
            },
        ];
        for inst in &ops {
            assert_eq!(inst.operands(), vec![Value(3), Value(4)]);
        }
    }

    #[test]
    fn test_operands_gep() {
        let inst = Instruction::GetElementPtr {
            result: Value(10),
            base_ty: IrType::Struct {
                fields: vec![IrType::I32, IrType::I64],
                packed: false,
            },
            ptr: Value(5),
            indices: vec![Value(6), Value(7)],
            in_bounds: true,
        };
        assert_eq!(inst.operands(), vec![Value(5), Value(6), Value(7)]);
    }

    #[test]
    fn test_operands_select() {
        let inst = Instruction::Select {
            result: Value(10),
            condition: Value(1),
            true_val: Value(2),
            false_val: Value(3),
            ty: IrType::I32,
        };
        assert_eq!(inst.operands(), vec![Value(1), Value(2), Value(3)]);
    }

    #[test]
    fn test_operands_const() {
        let inst = Instruction::Const {
            result: Value(0),
            value: Constant::Integer {
                value: 42,
                ty: IrType::I32,
            },
        };
        assert!(inst.operands().is_empty());
    }

    #[test]
    fn test_operands_copy() {
        let inst = Instruction::Copy {
            result: Value(10),
            source: Value(5),
            ty: IrType::I32,
        };
        assert_eq!(inst.operands(), vec![Value(5)]);
    }

    #[test]
    fn test_operands_call_with_indirect() {
        let inst = Instruction::Call {
            result: Some(Value(10)),
            callee: Callee::Indirect(Value(20)),
            args: vec![Value(1), Value(2)],
            return_ty: IrType::I32,
        };
        let ops = inst.operands();
        assert!(ops.contains(&Value(1)));
        assert!(ops.contains(&Value(2)));
        assert!(ops.contains(&Value(20)));
    }

    // -----------------------------------------------------------------------
    // uses_value() tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_uses_value_found() {
        let inst = Instruction::Add {
            result: Value(3),
            lhs: Value(1),
            rhs: Value(2),
            ty: IrType::I32,
        };
        assert!(inst.uses_value(Value(1)));
        assert!(inst.uses_value(Value(2)));
    }

    #[test]
    fn test_uses_value_not_found() {
        let inst = Instruction::Add {
            result: Value(3),
            lhs: Value(1),
            rhs: Value(2),
            ty: IrType::I32,
        };
        assert!(!inst.uses_value(Value(3))); // result, not operand
        assert!(!inst.uses_value(Value(99)));
    }

    #[test]
    fn test_uses_value_store() {
        let inst = Instruction::Store {
            value: Value(5),
            ptr: Value(6),
            store_ty: None,
        };
        assert!(inst.uses_value(Value(5)));
        assert!(inst.uses_value(Value(6)));
        assert!(!inst.uses_value(Value(7)));
    }

    // -----------------------------------------------------------------------
    // replace_use() tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_replace_use_basic() {
        let mut inst = Instruction::Add {
            result: Value(3),
            lhs: Value(1),
            rhs: Value(2),
            ty: IrType::I32,
        };
        inst.replace_use(Value(1), Value(10));
        assert_eq!(inst.operands(), vec![Value(10), Value(2)]);
        assert!(!inst.uses_value(Value(1)));
        assert!(inst.uses_value(Value(10)));
    }

    #[test]
    fn test_replace_use_both_operands() {
        let mut inst = Instruction::Add {
            result: Value(3),
            lhs: Value(1),
            rhs: Value(1),
            ty: IrType::I32,
        };
        inst.replace_use(Value(1), Value(99));
        assert_eq!(inst.operands(), vec![Value(99), Value(99)]);
    }

    #[test]
    fn test_replace_use_store() {
        let mut inst = Instruction::Store {
            value: Value(5),
            ptr: Value(6),
            store_ty: None,
        };
        inst.replace_use(Value(5), Value(50));
        assert_eq!(inst.operands(), vec![Value(50), Value(6)]);
    }

    #[test]
    fn test_replace_use_phi() {
        let mut inst = Instruction::Phi {
            result: Value(10),
            ty: IrType::I32,
            incoming: vec![(Value(1), BlockId(0)), (Value(2), BlockId(1))],
        };
        inst.replace_use(Value(1), Value(100));
        assert_eq!(inst.operands(), vec![Value(100), Value(2)]);
    }

    #[test]
    fn test_replace_use_no_match() {
        let mut inst = Instruction::Add {
            result: Value(3),
            lhs: Value(1),
            rhs: Value(2),
            ty: IrType::I32,
        };
        inst.replace_use(Value(99), Value(100));
        assert_eq!(inst.operands(), vec![Value(1), Value(2)]);
    }

    // -----------------------------------------------------------------------
    // result_type() tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_result_type_arithmetic() {
        let inst = Instruction::Add {
            result: Value(3),
            lhs: Value(1),
            rhs: Value(2),
            ty: IrType::I64,
        };
        assert_eq!(inst.result_type(), Some(&IrType::I64));
    }

    #[test]
    fn test_result_type_store() {
        let inst = Instruction::Store {
            value: Value(1),
            ptr: Value(2),
            store_ty: None,
        };
        assert_eq!(inst.result_type(), None);
    }

    #[test]
    fn test_result_type_cast() {
        let inst = Instruction::Cast {
            result: Value(5),
            op: CastOp::SExt,
            value: Value(4),
            from_ty: IrType::I8,
            to_ty: IrType::I32,
        };
        assert_eq!(inst.result_type(), Some(&IrType::I32));
    }

    #[test]
    fn test_result_type_call() {
        let inst = Instruction::Call {
            result: Some(Value(10)),
            callee: Callee::Direct("foo".to_string()),
            args: vec![],
            return_ty: IrType::I32,
        };
        assert_eq!(inst.result_type(), Some(&IrType::I32));
    }

    #[test]
    fn test_result_type_const_integer() {
        let inst = Instruction::Const {
            result: Value(0),
            value: Constant::Integer {
                value: 42,
                ty: IrType::I32,
            },
        };
        assert_eq!(inst.result_type(), Some(&IrType::I32));
    }

    #[test]
    fn test_result_type_nop() {
        assert_eq!(Instruction::Nop.result_type(), None);
    }

    // -----------------------------------------------------------------------
    // Display tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_display_add() {
        let inst = Instruction::Add {
            result: Value(3),
            lhs: Value(1),
            rhs: Value(2),
            ty: IrType::I32,
        };
        assert_eq!(format!("{}", inst), "%3 = add i32 %1, %2");
    }

    #[test]
    fn test_display_load() {
        let inst = Instruction::Load {
            result: Value(5),
            ty: IrType::I32,
            ptr: Value(4),
        };
        assert_eq!(format!("{}", inst), "%5 = load i32, i32* %4");
    }

    #[test]
    fn test_display_store() {
        let inst = Instruction::Store {
            value: Value(3),
            ptr: Value(4),
            store_ty: None,
        };
        assert_eq!(format!("{}", inst), "store %3, %4");
    }

    #[test]
    fn test_display_icmp() {
        let inst = Instruction::ICmp {
            result: Value(7),
            op: CompareOp::Equal,
            lhs: Value(5),
            rhs: Value(6),
            ty: IrType::I32,
        };
        assert_eq!(format!("{}", inst), "%7 = icmp eq i32 %5, %6");
    }

    #[test]
    fn test_display_fcmp() {
        let inst = Instruction::FCmp {
            result: Value(7),
            op: FloatCompareOp::OrderedLess,
            lhs: Value(5),
            rhs: Value(6),
            ty: IrType::F64,
        };
        assert_eq!(format!("{}", inst), "%7 = fcmp olt f64 %5, %6");
    }

    #[test]
    fn test_display_call() {
        let inst = Instruction::Call {
            result: Some(Value(9)),
            callee: Callee::Direct("printf".to_string()),
            args: vec![Value(8)],
            return_ty: IrType::I32,
        };
        assert_eq!(format!("{}", inst), "%9 = call i32 @printf(%8)");
    }

    #[test]
    fn test_display_void_call() {
        let inst = Instruction::Call {
            result: None,
            callee: Callee::Direct("abort".to_string()),
            args: vec![],
            return_ty: IrType::Void,
        };
        assert_eq!(format!("{}", inst), "call void @abort()");
    }

    #[test]
    fn test_display_phi() {
        let inst = Instruction::Phi {
            result: Value(10),
            ty: IrType::I32,
            incoming: vec![(Value(3), BlockId(1)), (Value(5), BlockId(2))],
        };
        assert_eq!(format!("{}", inst), "%10 = phi i32 [%3, bb1], [%5, bb2]");
    }

    #[test]
    fn test_display_alloca() {
        let inst = Instruction::Alloca {
            result: Value(0),
            ty: IrType::I32,
            count: None,
        };
        assert_eq!(format!("{}", inst), "%0 = alloca i32");
    }

    #[test]
    fn test_display_alloca_with_count() {
        let inst = Instruction::Alloca {
            result: Value(0),
            ty: IrType::I32,
            count: Some(Value(1)),
        };
        assert_eq!(format!("{}", inst), "%0 = alloca i32, %1");
    }

    #[test]
    fn test_display_gep() {
        let inst = Instruction::GetElementPtr {
            result: Value(10),
            base_ty: IrType::I32,
            ptr: Value(5),
            indices: vec![Value(6)],
            in_bounds: true,
        };
        assert_eq!(
            format!("{}", inst),
            "%10 = getelementptr inbounds i32, i32* %5, %6"
        );
    }

    #[test]
    fn test_display_cast() {
        let inst = Instruction::Cast {
            result: Value(5),
            op: CastOp::SExt,
            value: Value(4),
            from_ty: IrType::I8,
            to_ty: IrType::I32,
        };
        assert_eq!(format!("{}", inst), "%5 = sext i8 %4 to i32");
    }

    #[test]
    fn test_display_bitcast() {
        let inst = Instruction::BitCast {
            result: Value(5),
            value: Value(4),
            from_ty: IrType::Pointer(Box::new(IrType::I32)),
            to_ty: IrType::Pointer(Box::new(IrType::I8)),
        };
        assert_eq!(format!("{}", inst), "%5 = bitcast i32* %4 to i8*");
    }

    #[test]
    fn test_display_select() {
        let inst = Instruction::Select {
            result: Value(10),
            condition: Value(1),
            true_val: Value(2),
            false_val: Value(3),
            ty: IrType::I32,
        };
        assert_eq!(format!("{}", inst), "%10 = select i1 %1, i32 %2, i32 %3");
    }

    #[test]
    fn test_display_copy() {
        let inst = Instruction::Copy {
            result: Value(10),
            source: Value(5),
            ty: IrType::I32,
        };
        assert_eq!(format!("{}", inst), "%10 = copy i32 %5");
    }

    #[test]
    fn test_display_nop() {
        assert_eq!(format!("{}", Instruction::Nop), "nop");
    }

    #[test]
    fn test_display_div_signed() {
        let inst = Instruction::Div {
            result: Value(3),
            lhs: Value(1),
            rhs: Value(2),
            ty: IrType::I32,
            is_signed: true,
        };
        assert_eq!(format!("{}", inst), "%3 = sdiv i32 %1, %2");
    }

    #[test]
    fn test_display_div_unsigned() {
        let inst = Instruction::Div {
            result: Value(3),
            lhs: Value(1),
            rhs: Value(2),
            ty: IrType::I32,
            is_signed: false,
        };
        assert_eq!(format!("{}", inst), "%3 = udiv i32 %1, %2");
    }

    #[test]
    fn test_display_shr_arithmetic() {
        let inst = Instruction::Shr {
            result: Value(3),
            lhs: Value(1),
            rhs: Value(2),
            ty: IrType::I32,
            is_arithmetic: true,
        };
        assert_eq!(format!("{}", inst), "%3 = ashr i32 %1, %2");
    }

    #[test]
    fn test_display_shr_logical() {
        let inst = Instruction::Shr {
            result: Value(3),
            lhs: Value(1),
            rhs: Value(2),
            ty: IrType::I32,
            is_arithmetic: false,
        };
        assert_eq!(format!("{}", inst), "%3 = lshr i32 %1, %2");
    }

    // -----------------------------------------------------------------------
    // LocatedInstruction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_located_instruction_new() {
        use crate::common::source_map::FileId;

        let loc = SourceLocation {
            file_id: FileId(1),
            byte_offset: 100,
            line: 10,
            column: 5,
        };
        let inst = Instruction::Add {
            result: Value(0),
            lhs: Value(1),
            rhs: Value(2),
            ty: IrType::I32,
        };
        let located = LocatedInstruction::new(inst, loc);
        assert_eq!(located.location.file_id, FileId(1));
        assert_eq!(located.location.byte_offset, 100);
        assert_eq!(located.location.line, 10);
        assert_eq!(located.location.column, 5);
    }

    #[test]
    fn test_located_instruction_synthetic() {
        let inst = Instruction::Nop;
        let located = LocatedInstruction::synthetic(inst);
        assert!(located.location.is_dummy());
    }

    #[test]
    fn test_located_instruction_display() {
        let inst = Instruction::Add {
            result: Value(3),
            lhs: Value(1),
            rhs: Value(2),
            ty: IrType::I32,
        };
        let located = LocatedInstruction::synthetic(inst);
        assert_eq!(format!("{}", located), "%3 = add i32 %1, %2");
    }

    // -----------------------------------------------------------------------
    // operands_mut() tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_operands_mut_add() {
        let mut inst = Instruction::Add {
            result: Value(3),
            lhs: Value(1),
            rhs: Value(2),
            ty: IrType::I32,
        };
        let ops = inst.operands_mut();
        assert_eq!(ops.len(), 2);
    }

    #[test]
    fn test_operands_mut_store() {
        let mut inst = Instruction::Store {
            value: Value(1),
            ptr: Value(2),
            store_ty: None,
        };
        let ops = inst.operands_mut();
        assert_eq!(ops.len(), 2);
    }

    #[test]
    fn test_operands_mut_nop() {
        let mut inst = Instruction::Nop;
        let ops = inst.operands_mut();
        assert!(ops.is_empty());
    }

    #[test]
    fn test_operands_mut_const() {
        let mut inst = Instruction::Const {
            result: Value(0),
            value: Constant::Integer {
                value: 42,
                ty: IrType::I32,
            },
        };
        let ops = inst.operands_mut();
        assert!(ops.is_empty());
    }
}
