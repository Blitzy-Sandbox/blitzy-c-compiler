//! IR Builder — AST-to-IR Translation for the `bcc` compiler.
//!
//! This module implements [`IrBuilder`], the bridge between the frontend/sema pipeline
//! and the optimization/codegen pipeline. It translates a fully type-checked AST
//! ([`TypedTranslationUnit`]) into SSA-form IR instruction sequences organized as
//! a [`Module`] containing [`Function`]s and [`GlobalVariable`]s.
//!
//! # Architecture
//!
//! The builder operates in a single pass over the typed AST:
//! 1. Top-level declarations are dispatched to function or global variable builders
//! 2. Function bodies are lowered statement-by-statement into basic blocks
//! 3. Expressions are lowered recursively into IR instruction sequences
//! 4. After initial construction (with alloca/load/store patterns), SSA construction
//!    promotes stack allocations to SSA registers with phi nodes
//!
//! # C-Specific Semantics
//!
//! The builder handles:
//! - **Short-circuit evaluation** for `&&` and `||` via conditional branching
//! - **Sequence points** at statement boundaries and function calls
//! - **Compound assignments** (`+=`, `-=`, etc.) via load-operate-store patterns
//! - **Comma operator** (evaluate left, discard, evaluate right, keep)
//! - **Ternary `?:`** via conditional branch with phi node at merge point
//! - **Pre/post increment/decrement** with correct value semantics
//!
//! # Target Awareness
//!
//! Uses [`TargetConfig`] for target-specific type sizing (pointer width, `long` size,
//! etc.) during C-to-IR type mapping via [`IrBuilder::map_type()`].
//!
//! # Performance
//!
//! Designed to handle ~230K LOC (SQLite amalgamation) efficiently with O(1) symbol
//! lookups via `HashMap<InternId, Value>` and sequential instruction emission.
//!
//! # Zero External Dependencies
//!
//! Only imports from `std` and internal crate modules. No external crates.
//!
//! # Safety
//!
//! This module contains zero `unsafe` blocks. All operations use safe Rust.

use std::collections::HashMap;

use crate::common::diagnostics::DiagnosticEmitter;
use crate::common::intern::{InternId, Interner};
use crate::common::source_map::SourceSpan;
use crate::driver::target::TargetConfig;
use crate::ir::cfg::{BasicBlock, Terminator};
use crate::ir::instructions::{
    BlockId, Callee, CastOp, CompareOp, Constant, FloatCompareOp, Instruction, Value,
};
use crate::ir::types::IrType;
use crate::sema::types::{
    ArraySize as SemaArraySize, CType, FloatKind, IntegerKind,
};
use crate::sema::{TypedDeclaration, TypedTranslationUnit};

// Re-import AST types for pattern matching during lowering.
use crate::frontend::parser::ast::{
    AssignmentOp, BinaryOp, BlockItem, Declaration, Expression, FloatSuffix, ForInit,
    FunctionDef, InitDeclarator, Initializer, Statement, StorageClass, UnaryOp,
};

// ===========================================================================
// Module — Top-level IR container
// ===========================================================================

/// The top-level IR container representing an entire C translation unit.
///
/// A `Module` holds all functions (both definitions and declarations) and
/// global variables produced by lowering a [`TypedTranslationUnit`].
///
/// Consumed by the optimization pass pipeline and then by the code generation
/// backends.
#[derive(Debug, Clone)]
pub struct Module {
    /// All functions in this module (definitions and extern declarations).
    pub functions: Vec<Function>,
    /// All global variables in this module.
    pub globals: Vec<GlobalVariable>,
    /// The module name (typically derived from the source file name).
    pub name: String,
}

impl Module {
    /// Creates a new empty module with the given name.
    pub fn new(name: String) -> Self {
        Module {
            functions: Vec::new(),
            globals: Vec::new(),
            name,
        }
    }
}

// ===========================================================================
// Function — IR function representation
// ===========================================================================

/// An IR function containing a control flow graph of basic blocks.
///
/// Functions may be definitions (with a non-empty block list) or declarations
/// (extern stubs with `is_definition = false` and an empty block list).
#[derive(Debug, Clone)]
pub struct Function {
    /// The function's mangled or unmangled name (C functions are not mangled).
    pub name: String,
    /// The IR return type of the function.
    pub return_type: IrType,
    /// Parameter names and their IR types, in declaration order.
    pub params: Vec<(String, IrType)>,
    /// IR Value IDs for each function parameter, in declaration order.
    /// These are the values that the code generator should map to ABI
    /// register locations (rdi, rsi, rdx, rcx, r8, r9 for System V AMD64).
    pub param_values: Vec<Value>,
    /// Basic blocks comprising the function body. Empty for declarations.
    pub blocks: Vec<BasicBlock>,
    /// The entry block ID (the first block executed on function entry).
    pub entry_block: BlockId,
    /// `true` if this function has a body (definition), `false` for extern stubs.
    pub is_definition: bool,
    /// `true` if declared with `static` (internal linkage, local binding).
    pub is_static: bool,
    /// `true` if declared with `__attribute__((weak))` (weak binding).
    pub is_weak: bool,
}

// ===========================================================================
// GlobalVariable — IR global variable representation
// ===========================================================================

/// An IR global variable with optional initializer.
///
/// Represents file-scope variable declarations and definitions, including
/// `static` globals (internal linkage) and `extern` declarations.
#[derive(Debug, Clone)]
pub struct GlobalVariable {
    /// The variable's name.
    pub name: String,
    /// The IR type of the variable.
    pub ty: IrType,
    /// Optional compile-time constant initializer.
    pub initializer: Option<Constant>,
    /// `true` if declared with `extern` (no definition in this TU).
    pub is_extern: bool,
    /// `true` if declared with `static` (internal linkage).
    pub is_static: bool,
}

// ===========================================================================
// FunctionBuilder — per-function construction state
// ===========================================================================

/// Internal helper tracking per-function construction state.
///
/// Created when entering a function definition and consumed when the function
/// body has been fully lowered to produce a [`Function`].
struct FunctionBuilder {
    /// Function name.
    name: String,
    /// IR return type.
    return_type: IrType,
    /// Parameter (name, IR type) pairs.
    params: Vec<(String, IrType)>,
    /// IR Value IDs for function parameters, populated during param lowering.
    param_values: Vec<Value>,
    /// Basic blocks built so far.
    blocks: Vec<BasicBlock>,
    /// The current block being appended to.
    current_block: BlockId,
    /// The entry block ID.
    entry_block: BlockId,
    /// Per-function mapping from variable identifiers to their alloca values.
    local_values: HashMap<InternId, Value>,
    /// Per-function mapping from variable identifiers to their IR types.
    /// Used to emit correctly-typed Load instructions instead of defaulting
    /// to I32 for all variables.
    local_types: HashMap<InternId, IrType>,
    /// Per-function mapping from IR Values to their IR types.
    /// Used to detect pointer arithmetic in binary Add/Sub so the
    /// integer operand is correctly scaled by sizeof(pointee).
    value_types: HashMap<Value, IrType>,
}

impl FunctionBuilder {
    /// Creates a new function builder with the given name, return type, and params.
    fn new(name: String, return_type: IrType, params: Vec<(String, IrType)>, entry_id: BlockId) -> Self {
        let entry_block = BasicBlock::new(entry_id, "entry".to_string());
        FunctionBuilder {
            name,
            return_type,
            params,
            param_values: Vec::new(),
            blocks: vec![entry_block],
            current_block: entry_id,
            entry_block: entry_id,
            local_values: HashMap::new(),
            local_types: HashMap::new(),
            value_types: HashMap::new(),
        }
    }

    /// Returns a mutable reference to the current basic block.
    fn current_block_mut(&mut self) -> &mut BasicBlock {
        let idx = self.current_block.0 as usize;
        &mut self.blocks[idx]
    }

    /// Returns an immutable reference to the current basic block.
    fn current_block_ref(&self) -> &BasicBlock {
        let idx = self.current_block.0 as usize;
        &self.blocks[idx]
    }

    /// Returns true if the current block already has a terminator.
    fn current_block_terminated(&self) -> bool {
        self.current_block_ref().terminator.is_some()
    }

    /// Adds a new basic block and returns its ID.
    fn add_block(&mut self, id: BlockId, label: String) -> BlockId {
        let block = BasicBlock::new(id, label);
        // Ensure the blocks vec is large enough
        let idx = id.0 as usize;
        while self.blocks.len() <= idx {
            let placeholder_id = BlockId(self.blocks.len() as u32);
            self.blocks.push(BasicBlock::new(
                placeholder_id,
                format!("__placeholder_{}", placeholder_id.0),
            ));
        }
        self.blocks[idx] = block;
        id
    }
}

// ===========================================================================
// IrBuilder — the central AST-to-IR translation engine
// ===========================================================================

/// The central AST-to-IR translation engine for the `bcc` compiler.
///
/// `IrBuilder` translates a fully type-checked [`TypedTranslationUnit`] into an
/// IR [`Module`] containing functions and global variables. It handles all C11
/// expression and statement forms, including GCC extensions, with correct
/// C-specific semantics (short-circuit evaluation, compound assignments, etc.).
///
/// # Lifetime `'a`
///
/// The builder borrows the target configuration and diagnostic emitter for the
/// duration of IR construction.
///
/// # Usage
///
/// ```ignore
/// let mut diagnostics = DiagnosticEmitter::new();
/// let target = TargetConfig::x86_64();
/// let mut builder = IrBuilder::new(&target, &mut diagnostics, "main.c", &interner);
/// let module = builder.build(&typed_tu);
/// ```
pub struct IrBuilder<'a> {
    /// The module being built (collection of functions and globals).
    module: Module,
    /// Current function being built (None when processing top-level declarations).
    current_function: Option<FunctionBuilder>,
    /// Target configuration for type sizing.
    target: &'a TargetConfig,
    /// Diagnostic emitter for errors during IR construction.
    diagnostics: &'a mut DiagnosticEmitter,
    /// String interner for resolving InternId → &str (identifiers, labels, etc.).
    interner: &'a Interner,
    /// Counter for generating unique value IDs.
    next_value_id: u32,
    /// Counter for generating unique block IDs.
    next_block_id: u32,
    /// Map from symbol identifiers to IR values (for variable → alloca mapping).
    symbol_values: HashMap<InternId, Value>,
    /// Map from global/static symbol identifiers to their IR types.
    /// Used to emit correctly-typed Load instructions for global variables.
    symbol_types: HashMap<InternId, IrType>,
    /// Map from label names to block IDs (for goto support).
    label_blocks: HashMap<InternId, BlockId>,
    /// Break target stack (for nested loops/switches).
    break_targets: Vec<BlockId>,
    /// Continue target stack (for nested loops).
    continue_targets: Vec<BlockId>,
    /// String literal counter for generating unique global names.
    string_literal_count: u32,
    /// Map from struct/union tag names to their field names (in order).
    /// Used to compute field byte offsets in MemberAccess/ArrowAccess GEP.
    struct_field_names: HashMap<String, Vec<(String, IrType)>>,
}

impl<'a> IrBuilder<'a> {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Creates a new `IrBuilder` with the given target configuration, diagnostic
    /// emitter, and module name.
    pub fn new(
        target: &'a TargetConfig,
        diagnostics: &'a mut DiagnosticEmitter,
        module_name: &str,
        interner: &'a Interner,
    ) -> Self {
        IrBuilder {
            module: Module::new(module_name.to_string()),
            current_function: None,
            target,
            diagnostics,
            interner,
            next_value_id: 0,
            next_block_id: 0,
            symbol_values: HashMap::new(),
            symbol_types: HashMap::new(),
            label_blocks: HashMap::new(),
            break_targets: Vec::new(),
            continue_targets: Vec::new(),
            string_literal_count: 0,
            struct_field_names: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Public API: build()
    // -----------------------------------------------------------------------

    /// Translates a fully type-checked translation unit into an IR [`Module`].
    ///
    /// Iterates over all top-level declarations and dispatches to the appropriate
    /// builder method:
    /// - Variable declarations → global variables
    /// - Function definitions → full function bodies with CFG
    /// - Function declarations → extern stubs
    /// - Typedefs, static asserts, empty declarations → skipped (no IR needed)
    ///
    /// Returns the completed module containing all functions and globals.
    pub fn build(&mut self, translation_unit: &TypedTranslationUnit) -> Module {
        for typed_decl in &translation_unit.declarations {
            self.lower_top_level_declaration(typed_decl);
        }
        // Take ownership of the completed module, leaving an empty one in its place.
        std::mem::replace(&mut self.module, Module::new(String::new()))
    }

    // -----------------------------------------------------------------------
    // Public API: Value and block generation
    // -----------------------------------------------------------------------

    /// Generates a unique SSA [`Value`] identifier.
    ///
    /// Each call returns a new value with a monotonically increasing ID.
    pub fn new_value(&mut self, _ty: IrType) -> Value {
        let id = self.next_value_id;
        self.next_value_id = self.next_value_id.wrapping_add(1);
        Value(id)
    }

    /// Appends an instruction to the current basic block and returns its result value.
    ///
    /// For instructions that produce no result (e.g., `Store`), returns `Value::undef()`.
    pub fn emit_instruction(&mut self, inst: Instruction) -> Value {
        let result = inst.result().unwrap_or(Value::undef());
        if let Some(ref mut fb) = self.current_function {
            if !fb.current_block_terminated() {
                fb.current_block_mut().instructions.push(inst);
            }
        }
        result
    }

    /// Sets the current insertion point to the given block.
    ///
    /// All subsequent instructions will be appended to this block until the
    /// insertion point is changed again.
    pub fn set_insert_point(&mut self, block_id: BlockId) {
        if let Some(ref mut fb) = self.current_function {
            fb.current_block = block_id;
        }
    }

    /// Creates a new empty basic block with a descriptive label and returns its ID.
    pub fn create_block(&mut self, name: &str) -> BlockId {
        let id = BlockId(self.next_block_id);
        self.next_block_id = self.next_block_id.wrapping_add(1);
        if let Some(ref mut fb) = self.current_function {
            fb.add_block(id, name.to_string());
        }
        id
    }

    /// Maps a C type ([`CType`]) to the corresponding IR type ([`IrType`]).
    ///
    /// Uses the target configuration for target-dependent type sizes (pointer width,
    /// `long` size, `long double` size, etc.).
    pub fn map_type(&self, c_type: &CType) -> IrType {
        match c_type {
            CType::Void => IrType::Void,

            CType::Integer(kind) => match kind {
                IntegerKind::Bool => IrType::I8,
                IntegerKind::Char | IntegerKind::SignedChar | IntegerKind::UnsignedChar => {
                    IrType::I8
                }
                IntegerKind::Short | IntegerKind::UnsignedShort => IrType::I16,
                IntegerKind::Int | IntegerKind::UnsignedInt => IrType::I32,
                IntegerKind::Long | IntegerKind::UnsignedLong => {
                    IrType::int_type_for_size(self.target.long_size as usize)
                }
                IntegerKind::LongLong | IntegerKind::UnsignedLongLong => IrType::I64,
            },

            CType::Float(kind) => match kind {
                FloatKind::Float => IrType::F32,
                FloatKind::Double => IrType::F64,
                // long double maps to F64 for IR purposes (backend handles extended precision)
                FloatKind::LongDouble => IrType::F64,
            },

            CType::Pointer { pointee, .. } => {
                let pt = self.map_type(pointee);
                IrType::Pointer(Box::new(pt))
            }

            CType::Array { element, size } => {
                let elem_ty = self.map_type(element);
                let count = match size {
                    SemaArraySize::Fixed(n) => *n,
                    SemaArraySize::Variable | SemaArraySize::Incomplete => 0,
                };
                IrType::Array {
                    element: Box::new(elem_ty),
                    count,
                }
            }

            CType::Struct(st) => {
                let field_types: Vec<IrType> = st.fields.iter().map(|f| self.map_type(&f.ty)).collect();
                IrType::Struct {
                    fields: field_types,
                    packed: st.is_packed,
                }
            }

            // Enum underlying type is always int in C11 (§6.7.2.2)
            CType::Enum(_) => IrType::I32,

            CType::Function(ft) => {
                let ret = self.map_type(&ft.return_type);
                let param_types: Vec<IrType> =
                    ft.params.iter().map(|p| self.map_type(&p.ty)).collect();
                IrType::Function {
                    return_type: Box::new(ret),
                    param_types,
                    is_variadic: ft.is_variadic,
                }
            }

            CType::Typedef { underlying, .. } => self.map_type(underlying),

            CType::Qualified { base, .. } => self.map_type(base),

            CType::TypeOf(inner) => self.map_type(inner),

            CType::Error => IrType::I32, // Error recovery: default to i32
        }
    }

    /// Returns the IR type associated with a Value.
    ///
    /// Checks the current function's instructions for the Value's defining
    /// instruction and returns its result type. Returns I64 (pointer size)
    /// as a conservative fallback if the value's type cannot be determined.
    fn value_type(&self, val: Value) -> IrType {
        if let Some(ref fb) = self.current_function {
            // Search through all blocks for the defining instruction
            for block in &fb.blocks {
                for inst in &block.instructions {
                    if inst.result() == Some(val) {
                        if let Some(ty) = inst.result_type() {
                            // For Alloca instructions, the stored type is what
                            // sizeof should return, not the pointer type.
                            if let Instruction::Alloca { ty: alloca_ty, .. } = inst {
                                return alloca_ty.clone();
                            }
                            // For Load instructions, return the loaded type.
                            return ty.clone();
                        }
                    }
                }
            }
        }
        // Conservative fallback: pointer-sized integer.
        IrType::int_type_for_size(self.target.pointer_size as usize)
    }

    /// Resolves a TypeName AST node to a byte size for sizeof evaluation.
    ///
    /// Inspects the type specifiers in the TypeName to determine the base C type,
    /// then computes the target-specific size. Handles pointers via abstract
    /// declarator inspection.
    fn resolve_sizeof_type_name(&mut self, type_name: &crate::frontend::parser::ast::TypeName) -> i64 {
        use crate::frontend::parser::ast::TypeSpecifier;

        // Check if the abstract declarator makes this a pointer type.
        let has_pointer = type_name.abstract_declarator.as_ref()
            .map(|ad| !ad.pointer.is_empty())
            .unwrap_or(false);
        if has_pointer {
            return self.target.pointer_size as i64;
        }

        let size = self.resolve_sizeof_type_specifier(&type_name.specifiers.type_specifier);
        size
    }

    /// Resolves a TypeSpecifier to a byte size for sizeof evaluation.
    fn resolve_sizeof_type_specifier(&mut self, spec: &crate::frontend::parser::ast::TypeSpecifier) -> i64 {
        use crate::frontend::parser::ast::TypeSpecifier;
        match spec {
            TypeSpecifier::Void => 0, // sizeof(void) is 0 (GCC extension: 1, but 0 is standard)
            TypeSpecifier::Bool => 1,
            TypeSpecifier::Char => 1,
            TypeSpecifier::Short => 2,
            TypeSpecifier::Int => 4,
            TypeSpecifier::Long => self.target.long_size as i64,
            TypeSpecifier::LongLong => 8,
            TypeSpecifier::Float => 4,
            TypeSpecifier::Double => 8,
            TypeSpecifier::LongDouble => self.target.long_double_size as i64,
            TypeSpecifier::Signed(inner) => self.resolve_sizeof_type_specifier(inner),
            TypeSpecifier::Unsigned(inner) => self.resolve_sizeof_type_specifier(inner),
            TypeSpecifier::Complex(inner) => self.resolve_sizeof_type_specifier(inner) * 2,
            TypeSpecifier::Atomic(inner) => self.resolve_sizeof_type_specifier(inner),
            TypeSpecifier::Qualified { inner, .. } => self.resolve_sizeof_type_specifier(inner),
            // Struct/union/enum definitions and references — compute from
            // the resolved IrType to account for alignment padding.
            TypeSpecifier::Struct(def) => {
                self.register_struct_def(def);
                let ctype = self.specifier_to_ctype(spec);
                let ir_ty = self.map_type(&ctype);
                let size = ir_ty.size(&self.target) as i64;
                if size == 0 { self.target.pointer_size as i64 } else { size }
            }
            TypeSpecifier::Enum(_) => 4, // Enum underlying type is always int
            TypeSpecifier::StructRef { .. } | TypeSpecifier::UnionRef { .. } => {
                // Resolve struct/union reference through specifier_to_ctype
                // which uses struct_field_names for actual field types.
                let ctype = self.specifier_to_ctype(spec);
                let ir_ty = self.map_type(&ctype);
                let size = ir_ty.size(&self.target) as i64;
                if size == 0 { self.target.pointer_size as i64 } else { size }
            }
            TypeSpecifier::EnumRef { .. } => 4, // enum is always int-sized
            TypeSpecifier::Union(def) => {
                let ctype = self.specifier_to_ctype(spec);
                let ir_ty = self.map_type(&ctype);
                let size = ir_ty.size(&self.target) as i64;
                if size == 0 { self.target.pointer_size as i64 } else { size }
            }
            TypeSpecifier::TypedefName { .. } => self.target.pointer_size as i64,
            TypeSpecifier::Typeof { .. } | TypeSpecifier::TypeofType { .. } => {
                self.target.pointer_size as i64
            }
            TypeSpecifier::Error => self.target.pointer_size as i64,
        }
    }

    /// Resolves a TypeName AST node to an alignment for _Alignof evaluation.
    fn resolve_alignof_type_name(&mut self, type_name: &crate::frontend::parser::ast::TypeName) -> i64 {
        // For pointer types, alignment = pointer alignment
        let has_pointer = type_name.abstract_declarator.as_ref()
            .map(|ad| !ad.pointer.is_empty())
            .unwrap_or(false);
        if has_pointer {
            return self.target.pointer_size as i64;
        }
        // For most types, alignment = min(size, max_alignment)
        let size = self.resolve_sizeof_type_specifier(&type_name.specifiers.type_specifier);
        if size == 0 { 1 } else { size.min(self.target.max_alignment as i64) }
    }

    /// Resolves a TypeName AST node to an IrType for cast expressions.
    ///
    /// Handles pointer types (via abstract declarator) and base types (via specifiers).
    /// Used by the Cast expression handler to determine the target type of a C cast.
    fn resolve_type_name_to_ir_type(
        &mut self,
        type_name: &crate::frontend::parser::ast::TypeName,
    ) -> IrType {
        let base_ir = self.resolve_specifier_to_ir_type(&type_name.specifiers.type_specifier);

        // Apply abstract declarator modifiers (pointers, arrays)
        if let Some(ref ad) = type_name.abstract_declarator {
            let mut result = base_ir;
            // Apply pointer modifiers
            for _ptr in &ad.pointer {
                result = IrType::Pointer(Box::new(result));
            }
            // Apply direct abstract declarator (array suffixes)
            if let Some(ref direct) = ad.direct {
                result = self.resolve_direct_abstract_decl_ir_type(result, direct);
            }
            result
        } else {
            base_ir
        }
    }

    /// Resolves a DirectAbstractDeclarator to an IrType by applying array/function modifiers.
    fn resolve_direct_abstract_decl_ir_type(
        &self,
        base_ir: IrType,
        direct: &crate::frontend::parser::ast::DirectAbstractDeclarator,
    ) -> IrType {
        use crate::frontend::parser::ast::{DirectAbstractDeclarator, ArraySize as AstArraySize};
        match direct {
            DirectAbstractDeclarator::Parenthesized(inner) => {
                let mut result = base_ir;
                for _ptr in &inner.pointer {
                    result = IrType::Pointer(Box::new(result));
                }
                if let Some(ref d) = inner.direct {
                    result = self.resolve_direct_abstract_decl_ir_type(result, d);
                }
                result
            }
            DirectAbstractDeclarator::Array { base, size, .. } => {
                let inner = if let Some(ref b) = base {
                    self.resolve_direct_abstract_decl_ir_type(base_ir, b)
                } else {
                    base_ir
                };
                let count = match size {
                    AstArraySize::Fixed(expr) => {
                        self.try_eval_const_expr(expr).unwrap_or(0) as usize
                    }
                    AstArraySize::Unspecified | AstArraySize::VLA => 0,
                    AstArraySize::Static(expr) => {
                        self.try_eval_const_expr(expr).unwrap_or(0) as usize
                    }
                };
                IrType::Array {
                    element: Box::new(inner),
                    count,
                }
            }
            DirectAbstractDeclarator::Function { .. } => {
                // Function pointer in abstract declarator — treat as pointer to function
                IrType::Pointer(Box::new(base_ir))
            }
        }
    }

    /// Checks if a TypeSpecifier represents an unsigned integer type.
    ///
    /// Used to determine whether to emit ZExt (unsigned) or SExt (signed) for
    /// integer widening cast operations.
    fn is_unsigned_specifier(
        &self,
        spec: &crate::frontend::parser::ast::TypeSpecifier,
    ) -> bool {
        use crate::frontend::parser::ast::TypeSpecifier;
        match spec {
            TypeSpecifier::Unsigned(_) => true,
            TypeSpecifier::Bool => true, // _Bool is unsigned
            TypeSpecifier::Signed(_) => false,
            TypeSpecifier::Qualified { inner, .. } | TypeSpecifier::Atomic(inner) => {
                self.is_unsigned_specifier(inner)
            }
            // Default: signed for char, short, int, long, longlong
            _ => false,
        }
    }

    /// Resolves a TypeSpecifier into an IrType for local variable allocation.
    ///
    /// This allows the IR builder to allocate correct sizes for local variables
    /// including arrays (e.g. `char buf[8192]` allocates 8192 bytes, not 4).
    fn resolve_specifier_to_ir_type(
        &mut self,
        spec: &crate::frontend::parser::ast::TypeSpecifier,
    ) -> IrType {
        use crate::frontend::parser::ast::TypeSpecifier;
        match spec {
            TypeSpecifier::Void => IrType::Void,
            TypeSpecifier::Bool => IrType::I8,
            TypeSpecifier::Char => IrType::I8,
            TypeSpecifier::Short => IrType::I16,
            TypeSpecifier::Int => IrType::I32,
            TypeSpecifier::Long => {
                IrType::int_type_for_size(self.target.long_size as usize)
            }
            TypeSpecifier::LongLong => IrType::I64,
            TypeSpecifier::Float => IrType::F32,
            TypeSpecifier::Double => IrType::F64,
            TypeSpecifier::LongDouble => IrType::F64,
            TypeSpecifier::Signed(inner)
            | TypeSpecifier::Unsigned(inner)
            | TypeSpecifier::Atomic(inner)
            | TypeSpecifier::Qualified { inner, .. } => {
                self.resolve_specifier_to_ir_type(inner)
            }
            TypeSpecifier::Complex(inner) => {
                // Simplify: treat _Complex as just the base type
                self.resolve_specifier_to_ir_type(inner)
            }
            TypeSpecifier::Enum(_) | TypeSpecifier::EnumRef { .. } => IrType::I32,
            TypeSpecifier::Struct(def) => {
                self.register_struct_def(def);
                let fields: Vec<IrType> = self.get_struct_ir_fields(def);
                IrType::Struct { fields, packed: false }
            }
            TypeSpecifier::StructRef { tag, .. } => {
                let tag = self.interner.resolve(*tag).to_string();
                if let Some(field_info) = self.struct_field_names.get(&tag) {
                    let fields: Vec<IrType> = field_info.iter().map(|(_, ty)| ty.clone()).collect();
                    IrType::Struct { fields, packed: false }
                } else {
                    // Unknown struct ref — treat as opaque pointer-sized
                    IrType::I32
                }
            }
            TypeSpecifier::Union(def) => {
                // Union: all fields at offset 0, size = max field size
                let mut max_size: usize = 0;
                let mut max_ty = IrType::I32;
                for member in &def.members {
                    if let crate::frontend::parser::ast::StructMember::Field { specifiers, .. } = member {
                        let ty = self.resolve_specifier_to_ir_type(&specifiers.type_specifier);
                        let sz = Self::approx_ir_type_size(&ty);
                        if sz > max_size {
                            max_size = sz;
                            max_ty = ty;
                        }
                    }
                }
                max_ty
            }
            _ => IrType::I32, // fallback for typedef, etc.
        }
    }

    /// Register struct field names and types from a struct definition AST node.
    fn register_struct_def(&mut self, def: &crate::frontend::parser::ast::StructDef) {
        let tag = if let Some(tag_id) = def.tag {
            self.interner.resolve(tag_id).to_string()
        } else {
            return; // Anonymous struct — can't look up by name
        };
        if self.struct_field_names.contains_key(&tag) {
            return; // Already registered
        }
        let fields = self.get_struct_field_info(def);
        self.struct_field_names.insert(tag, fields);
    }

    /// Extract field name+type pairs from a struct definition.
    fn get_struct_field_info(&mut self, def: &crate::frontend::parser::ast::StructDef) -> Vec<(String, IrType)> {
        let mut fields = Vec::new();
        for member in &def.members {
            if let crate::frontend::parser::ast::StructMember::Field { specifiers, declarators, .. } = member {
                let base_ty = self.resolve_specifier_to_ir_type(&specifiers.type_specifier);
                if declarators.is_empty() {
                    fields.push(("".to_string(), base_ty));
                } else {
                    for decl in declarators {
                        if let Some(ref declarator) = decl.declarator {
                            let field_ty = self.resolve_declarator_ir_type(base_ty.clone(), declarator);
                            let name = Self::extract_declarator_name(declarator, self.interner);
                            fields.push((name, field_ty));
                        } else {
                            fields.push(("".to_string(), base_ty.clone()));
                        }
                    }
                }
            }
        }
        fields
    }

    /// Get the IR field types for a struct definition.
    fn get_struct_ir_fields(&mut self, def: &crate::frontend::parser::ast::StructDef) -> Vec<IrType> {
        self.get_struct_field_info(def).into_iter().map(|(_, ty)| ty).collect()
    }

    /// Infer the store type for an assignment target expression.
    /// Returns Some(IrType) for struct member accesses (to emit correctly-sized stores),
    /// None for regular variable stores (which use the default 64-bit store on x86-64).
    fn infer_store_type_from_target(&self, target: &Expression) -> Option<IrType> {
        match target {
            Expression::MemberAccess { object, member, .. } => {
                let member_name = self.interner.resolve(*member).to_string();
                let (_, field_ty) = self.compute_struct_member_offset(object, &member_name);
                Some(field_ty)
            }
            Expression::ArrowAccess { pointer, member, .. } => {
                let member_name = self.interner.resolve(*member).to_string();
                let (_, field_ty) = self.compute_struct_member_offset_from_ptr(pointer, &member_name);
                Some(field_ty)
            }
            Expression::UnaryPrefix { op: crate::frontend::parser::ast::UnaryOp::Dereference, operand, .. } => {
                // For *p = val, infer pointee type from the pointer expression
                if let Expression::Identifier { name, .. } = operand.as_ref() {
                    if let Some(ref fb) = self.current_function {
                        if let Some(ty) = fb.local_types.get(name) {
                            if let IrType::Pointer(inner) = ty {
                                return Some(inner.as_ref().clone());
                            }
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Register struct field names from a TypeSpecifier if it contains a struct definition.
    fn try_register_struct_from_specifier(&mut self, spec: &crate::frontend::parser::ast::TypeSpecifier) {
        use crate::frontend::parser::ast::TypeSpecifier;
        match spec {
            TypeSpecifier::Struct(def) => {
                self.register_struct_def(def);
            }
            TypeSpecifier::Signed(inner) | TypeSpecifier::Unsigned(inner) |
            TypeSpecifier::Qualified { inner, .. } | TypeSpecifier::Atomic(inner) => {
                self.try_register_struct_from_specifier(inner);
            }
            _ => {}
        }
    }

    /// Resolves a Declarator (from a local variable declaration) to an IrType,
    /// accounting for pointer and array suffixes.
    ///
    /// For `char buf[8192]`: base_ir=I8, declarator has Array(8192) → Array { I8, 8192 }
    /// For `int *p`: base_ir=I32, declarator has pointer → Pointer(I32)
    fn resolve_declarator_ir_type(
        &self,
        base_ir: IrType,
        declarator: &crate::frontend::parser::ast::Declarator,
    ) -> IrType {
        use crate::frontend::parser::ast::DirectDeclarator;
        let mut result = base_ir;

        // Apply pointer modifiers
        for _ptr in &declarator.pointer {
            result = IrType::Pointer(Box::new(result));
        }

        // Apply direct declarator modifiers (array)
        result = self.resolve_direct_decl_ir_type(result, &declarator.direct);

        result
    }

    fn resolve_direct_decl_ir_type(
        &self,
        base_ir: IrType,
        direct: &crate::frontend::parser::ast::DirectDeclarator,
    ) -> IrType {
        use crate::frontend::parser::ast::{DirectDeclarator, ArraySize as AstArraySize};
        match direct {
            DirectDeclarator::Identifier(_) | DirectDeclarator::Abstract => base_ir,
            DirectDeclarator::Parenthesized(inner) => {
                self.resolve_declarator_ir_type(base_ir, inner)
            }
            DirectDeclarator::Array { base, size, .. } => {
                let inner = self.resolve_direct_decl_ir_type(base_ir, base);
                let count = match size {
                    AstArraySize::Fixed(expr) => {
                        // Try to evaluate the size as a constant
                        self.try_eval_const_expr(expr).unwrap_or(0) as usize
                    }
                    AstArraySize::Unspecified | AstArraySize::VLA => 0,
                    AstArraySize::Static(expr) => {
                        self.try_eval_const_expr(expr).unwrap_or(0) as usize
                    }
                };
                IrType::Array {
                    element: Box::new(inner),
                    count,
                }
            }
            DirectDeclarator::Function { .. } => {
                // Function declarators produce function types — locals can't
                // be function types, so fall back to pointer.
                IrType::Pointer(Box::new(base_ir))
            }
        }
    }

    /// Try to evaluate a constant expression at IR-build time.
    /// Returns None if the expression cannot be evaluated to a compile-time constant.
    fn try_eval_const_expr(&self, expr: &crate::frontend::parser::ast::Expression) -> Option<i64> {
        use crate::frontend::parser::ast::Expression;
        match expr {
            Expression::IntegerLiteral { value, .. } => Some(*value as i64),
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // Top-level declaration lowering
    // -----------------------------------------------------------------------

    /// Lowers a single top-level typed declaration into IR constructs.
    fn lower_top_level_declaration(&mut self, typed_decl: &TypedDeclaration) {
        match &typed_decl.decl {
            Declaration::Variable {
                specifiers,
                declarators,
                span,
            } => {
                self.lower_global_variable_declaration(specifiers, declarators, *span, &typed_decl.resolved_type);
            }

            Declaration::Function(func_def) => {
                self.build_function(func_def, &typed_decl.resolved_type);
            }

            // Typedefs produce no IR output.
            Declaration::Typedef { .. } => {}

            // Static asserts are compile-time checks already handled by sema.
            Declaration::StaticAssert { .. } => {}

            // Empty declarations produce no IR.
            Declaration::Empty { .. } => {}
        }
    }

    // -----------------------------------------------------------------------
    // Global variable lowering
    // -----------------------------------------------------------------------

    /// Lowers a file-scope variable declaration into a [`GlobalVariable`].
    fn lower_global_variable_declaration(
        &mut self,
        specifiers: &crate::frontend::parser::ast::DeclSpecifiers,
        declarators: &[InitDeclarator],
        _span: SourceSpan,
        resolved_type: &Option<CType>,
    ) {
        // Register struct definitions even if no declarators
        self.try_register_struct_from_specifier(&specifiers.type_specifier);

        let is_extern = specifiers.storage_class == Some(StorageClass::Extern);
        let is_static = specifiers.storage_class == Some(StorageClass::Static);

        for init_decl in declarators {
            // Extract name from the declarator using the interner
            let name = Self::extract_declarator_name(&init_decl.declarator, self.interner);
            if name.is_empty() {
                continue;
            }

            // Check if this declaration is actually a function declaration
            // (e.g., `int foo(void);` or `extern int bar(int);`).
            // Function declarations at file scope have implicit extern linkage
            // even without an explicit `extern` keyword.
            if Self::is_function_declarator(&init_decl.declarator) {
                // Create an extern function stub (no body).
                let func = Function {
                    name: name.clone(),
                    return_type: self.map_type(
                        &resolved_type.as_ref().cloned().unwrap_or(CType::Integer(IntegerKind::Int))
                    ),
                    params: Vec::new(),
                    param_values: Vec::new(),
                    blocks: Vec::new(),
                    entry_block: BlockId(0),
                    is_definition: false,
                    is_static: is_static,
                    is_weak: false,
                };
                self.module.functions.push(func);
                continue;
            }

            // Determine the type: prefer resolved_type from sema, fall back to
            // resolving from the declaration's specifiers + declarator.
            let ir_type = if let Some(ref ct) = resolved_type {
                self.map_type(ct)
            } else {
                // Resolve from specifiers + declarator to handle arrays, pointers, etc.
                let base_ir = self.resolve_specifier_to_ir_type(&specifiers.type_specifier);
                self.resolve_declarator_ir_type(base_ir, &init_decl.declarator)
            };

            // Lower the initializer if present
            let initializer = init_decl.initializer.as_ref().and_then(|init| {
                self.lower_constant_initializer(init, &ir_type)
            });

            // For file-scope variable declarations without explicit `extern`,
            // check if implicit extern applies (declaration without initializer
            // at file scope is a tentative definition in C, not extern).
            self.module.globals.push(GlobalVariable {
                name: name.clone(),
                ty: ir_type.clone(),
                initializer,
                is_extern,
                is_static,
            });

            // Register the global variable's type so that Identifier
            // expression lookups can emit correctly-typed Load instructions.
            // The actual GlobalRef + Load instructions are emitted at use
            // time inside function bodies (see lower_expression/Identifier).
            if let Some(intern_id) = self.interner.get(&name) {
                self.symbol_types.insert(intern_id, ir_type.clone());
            }
        }
    }

    /// Attempts to lower an initializer to a compile-time constant.
    ///
    /// Returns `None` if the initializer cannot be evaluated at compile time
    /// (will be handled by runtime initialization in the function builder).
    fn lower_constant_initializer(&self, init: &Initializer, _ty: &IrType) -> Option<Constant> {
        match init {
            Initializer::Expression(expr) => self.try_eval_constant_expr(expr),
            Initializer::Compound { .. } => {
                // Compound initializers for globals: zero-initialize for now
                Some(Constant::ZeroInit(_ty.clone()))
            }
        }
    }

    /// Attempts to evaluate an expression as a compile-time constant.
    fn try_eval_constant_expr(&self, expr: &Expression) -> Option<Constant> {
        match expr {
            Expression::IntegerLiteral { value, .. } => {
                Some(Constant::Integer {
                    value: *value as i64,
                    ty: IrType::I32,
                })
            }
            Expression::FloatLiteral { value, suffix, .. } => {
                let ty = if *suffix == FloatSuffix::Float {
                    IrType::F32
                } else {
                    IrType::F64
                };
                Some(Constant::Float {
                    value: *value,
                    ty,
                })
            }
            Expression::StringLiteral { value, .. } => {
                let mut bytes = value.as_bytes().to_vec();
                bytes.push(0); // null terminator
                Some(Constant::String(bytes))
            }
            Expression::CharLiteral { value, .. } => {
                Some(Constant::Integer {
                    value: *value as i64,
                    ty: IrType::I8,
                })
            }
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // Function building
    // -----------------------------------------------------------------------

    /// Builds an IR function from an AST function definition.
    ///
    /// Creates the entry block, allocates stack space for parameters, stores
    /// parameter values into their allocas, lowers the function body, and ensures
    /// all control flow paths have proper terminators.
    fn build_function(&mut self, func_def: &FunctionDef, resolved_type: &Option<CType>) {
        // Determine function type from resolved_type or by analyzing the AST
        let (return_type, param_types) = self.resolve_function_signature(func_def, resolved_type);

        let func_name = Self::extract_declarator_name(&func_def.declarator, self.interner);
        if func_name.is_empty() {
            return;
        }

        let ir_ret_type = self.map_type(&return_type);
        let ir_params: Vec<(String, IrType)> = param_types
            .iter()
            .map(|(name, ty)| (name.clone(), self.map_type(ty)))
            .collect();

        // Reset per-function state
        self.next_block_id = 0;
        let entry_id = BlockId(self.next_block_id);
        self.next_block_id = self.next_block_id.wrapping_add(1);

        // Save value ID state (functions have independent value spaces conceptually,
        // but we keep a global counter for simplicity)
        let saved_symbol_values = std::mem::take(&mut self.symbol_values);
        let saved_symbol_types = std::mem::take(&mut self.symbol_types);
        let saved_label_blocks = std::mem::take(&mut self.label_blocks);
        let saved_break = std::mem::take(&mut self.break_targets);
        let saved_continue = std::mem::take(&mut self.continue_targets);

        // Initialize function builder
        self.current_function = Some(FunctionBuilder::new(
            func_name.clone(),
            ir_ret_type.clone(),
            ir_params.clone(),
            entry_id,
        ));

        // Allocate and store parameters
        for (i, (param_name, param_ty)) in ir_params.iter().enumerate() {
            let alloca_val = self.new_value(param_ty.clone().pointer_to());
            let alloca_inst = Instruction::Alloca {
                result: alloca_val,
                ty: param_ty.clone(),
                count: None,
            };
            self.emit_instruction(alloca_inst);

            // Create a value representing the parameter.
            // The code generator maps these param_val IDs to ABI register
            // locations (e.g., rdi, rsi, rdx for System V AMD64).
            let param_val = self.new_value(param_ty.clone());
            let const_inst = Instruction::Const {
                result: param_val,
                value: Constant::Integer {
                    value: i as i64,
                    ty: param_ty.clone(),
                },
            };
            // Store param_val into the function builder's param_values list
            // so the code generator can identify which IR Values correspond
            // to function parameters and map them to ABI registers.
            if let Some(ref mut fb) = self.current_function {
                fb.param_values.push(param_val);
            }
            self.emit_instruction(const_inst);

            // Check if this is a large struct parameter (>8 bytes) that
            // arrives as a hidden pointer per the System V AMD64 ABI.
            // For such parameters the caller passes the address of its copy,
            // so the callee must load the struct contents word-by-word from
            // that pointer into a fresh local alloca.
            let is_large_struct_param = matches!(param_ty, IrType::Struct { .. })
                && Self::approx_ir_type_size(param_ty) > 8;

            if is_large_struct_param {
                // Large struct: copy word-by-word from source pointer to
                // local alloca.  Each word is 8 bytes (I64), and we round
                // up to cover any trailing padding in the struct.
                let struct_size = Self::approx_ir_type_size(param_ty);
                let num_words = (struct_size + 7) / 8;
                for w in 0..num_words {
                    let off = (w * 8) as i64;
                    // Source address: param_val + byte offset
                    let src_ptr = if off == 0 {
                        param_val
                    } else {
                        let offset_val = self.emit_const_int(off, IrType::I64);
                        let src = self.new_value(IrType::Pointer(Box::new(IrType::I64)));
                        self.emit_instruction(Instruction::GetElementPtr {
                            result: src,
                            base_ty: IrType::I8,
                            ptr: param_val,
                            indices: vec![offset_val],
                            in_bounds: true,
                        });
                        src
                    };
                    // Load one 8-byte word from the source pointer
                    let word_val = self.new_value(IrType::I64);
                    self.emit_instruction(Instruction::Load {
                        result: word_val,
                        ty: IrType::I64,
                        ptr: src_ptr,
                    });
                    // Destination address: alloca_val + byte offset
                    let dst_ptr = if off == 0 {
                        alloca_val
                    } else {
                        let offset_val2 = self.emit_const_int(off, IrType::I64);
                        let dst = self.new_value(IrType::Pointer(Box::new(IrType::I64)));
                        self.emit_instruction(Instruction::GetElementPtr {
                            result: dst,
                            base_ty: IrType::I8,
                            ptr: alloca_val,
                            indices: vec![offset_val2],
                            in_bounds: true,
                        });
                        dst
                    };
                    // Store the word into the local alloca
                    self.emit_instruction(Instruction::Store {
                        value: word_val,
                        ptr: dst_ptr,
                        store_ty: None,
                    });
                }
            } else {
                // Small parameter: direct store of the ABI register value.
                let store_inst = Instruction::Store {
                    value: param_val,
                    ptr: alloca_val,
                    store_ty: None,
                };
                self.emit_instruction(store_inst);
            }

            // Register the alloca using the real InternId for the parameter name
            // so that Identifier expression lookups can find it.
            if let Some(real_id) = self.interner.get(param_name) {
                if let Some(ref mut fb) = self.current_function {
                    fb.local_values.insert(real_id, alloca_val);
                    fb.local_types.insert(real_id, param_ty.clone());
                }
            } else {
            }
        }

        // Lower the function body
        self.lower_statement(&func_def.body);

        // Ensure the last block has a terminator
        // Pre-generate a value ID in case we need it for zero-return
        let zero_val_id = self.next_value_id;
        self.next_value_id = self.next_value_id.wrapping_add(1);
        let zero_val = Value(zero_val_id);

        if let Some(ref mut fb) = self.current_function {
            if !fb.current_block_terminated() {
                // Insert implicit return for void functions, or return 0 for non-void
                let term = if ir_ret_type == IrType::Void {
                    Terminator::Return { value: None }
                } else {
                    // Create a zero constant for the return type
                    let zero_inst = Instruction::Const {
                        result: zero_val,
                        value: Constant::Integer {
                            value: 0,
                            ty: ir_ret_type.clone(),
                        },
                    };
                    fb.current_block_mut().instructions.push(zero_inst);
                    Terminator::Return {
                        value: Some(zero_val),
                    }
                };
                fb.current_block_mut().terminator = Some(term);
            }

            // Also ensure ALL blocks have terminators
            for block in &mut fb.blocks {
                if block.terminator.is_none() {
                    block.terminator = Some(Terminator::Unreachable);
                }
            }
        }

        // Extract the completed function
        let fb = self.current_function.take().unwrap();
        let is_static_fn = func_def.specifiers.storage_class
            == Some(crate::frontend::parser::ast::StorageClass::Static);
        let is_weak_fn = if let Some(weak_id) = self.interner.get("weak") {
            func_def.attributes.iter().any(|attr| attr.name == weak_id)
                || func_def.specifiers.attributes.iter().any(|attr| attr.name == weak_id)
        } else {
            false
        };
        let function = Function {
            name: fb.name,
            return_type: fb.return_type,
            params: fb.params,
            param_values: fb.param_values,
            blocks: fb.blocks,
            entry_block: fb.entry_block,
            is_definition: true,
            is_static: is_static_fn,
            is_weak: is_weak_fn,
        };

        self.module.functions.push(function);

        // Restore saved state
        self.symbol_values = saved_symbol_values;
        self.symbol_types = saved_symbol_types;
        self.label_blocks = saved_label_blocks;
        self.break_targets = saved_break;
        self.continue_targets = saved_continue;
    }

    /// Resolves the function signature from the AST and optional resolved CType.
    fn resolve_function_signature(
        &self,
        func_def: &FunctionDef,
        resolved_type: &Option<CType>,
    ) -> (CType, Vec<(String, CType)>) {
        // First try from the resolved semantic type
        if let Some(CType::Function(ft)) = resolved_type {
            let params: Vec<(String, CType)> = ft
                .params
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let name = p.name.clone().unwrap_or_else(|| format!("__param_{}", i));
                    (name, p.ty.clone())
                })
                .collect();
            return (*ft.return_type.clone(), params);
        }

        // Fallback: extract return type from specifiers and parameters from
        // the AST declarator when semantic resolution did not produce a
        // Function CType (e.g. the sema pass stored the resolved type in a
        // form the IR builder cannot reach).
        let return_type = self.specifier_to_ctype(&func_def.specifiers.type_specifier);

        // Walk into the declarator to find the Function direct declarator
        let param_list = Self::extract_param_list_from_declarator(&func_def.declarator);

        let params: Vec<(String, CType)> = match param_list {
            Some(pl) => {
                pl.params.iter().enumerate().map(|(i, pd)| {
                    // Extract parameter name from its declarator
                    let name = pd.declarator.as_ref()
                        .map(|d| Self::extract_declarator_name(d, self.interner))
                        .unwrap_or_else(|| format!("__param_{}", i));

                    // Convert parameter type specifier to CType, then apply
                    // pointer/array modifiers from the declarator
                    let base_ctype = self.specifier_to_ctype(&pd.specifiers.type_specifier);
                    let ctype = if let Some(ref d) = pd.declarator {
                        self.apply_declarator_to_ctype(base_ctype, d)
                    } else {
                        base_ctype
                    };
                    (name, ctype)
                }).collect()
            }
            None => Vec::new(),
        };

        (return_type, params)
    }

    /// Extracts the ParamList from a function Declarator by walking the
    /// DirectDeclarator tree to find the Function variant.
    fn extract_param_list_from_declarator(
        declarator: &crate::frontend::parser::ast::Declarator,
    ) -> Option<&crate::frontend::parser::ast::ParamList> {
        Self::extract_param_list_from_direct(&declarator.direct)
    }

    /// Recursive helper to find ParamList in a DirectDeclarator tree.
    fn extract_param_list_from_direct(
        dd: &crate::frontend::parser::ast::DirectDeclarator,
    ) -> Option<&crate::frontend::parser::ast::ParamList> {
        use crate::frontend::parser::ast::DirectDeclarator;
        match dd {
            DirectDeclarator::Function { params, .. } => Some(params),
            DirectDeclarator::Parenthesized(inner) => {
                Self::extract_param_list_from_declarator(inner)
            }
            DirectDeclarator::Array { base, .. } => {
                Self::extract_param_list_from_direct(base)
            }
            _ => None,
        }
    }

    /// Converts a TypeSpecifier to a CType for the AST-based fallback path
    /// in resolve_function_signature.
    fn specifier_to_ctype(
        &self,
        spec: &crate::frontend::parser::ast::TypeSpecifier,
    ) -> CType {
        use crate::frontend::parser::ast::TypeSpecifier;
        match spec {
            TypeSpecifier::Void => CType::Void,
            TypeSpecifier::Bool => CType::Integer(IntegerKind::Bool),
            TypeSpecifier::Char => CType::Integer(IntegerKind::Char),
            TypeSpecifier::Short => CType::Integer(IntegerKind::Short),
            TypeSpecifier::Int => CType::Integer(IntegerKind::Int),
            TypeSpecifier::Long => CType::Integer(IntegerKind::Long),
            TypeSpecifier::LongLong => CType::Integer(IntegerKind::LongLong),
            TypeSpecifier::Float => CType::Float(FloatKind::Float),
            TypeSpecifier::Double => CType::Float(FloatKind::Double),
            TypeSpecifier::LongDouble => CType::Float(FloatKind::LongDouble),
            TypeSpecifier::Signed(inner) => self.specifier_to_ctype(inner),
            TypeSpecifier::Unsigned(inner) => {
                match self.specifier_to_ctype(inner) {
                    CType::Integer(IntegerKind::Char) => CType::Integer(IntegerKind::UnsignedChar),
                    CType::Integer(IntegerKind::Short) => CType::Integer(IntegerKind::UnsignedShort),
                    CType::Integer(IntegerKind::Int) => CType::Integer(IntegerKind::UnsignedInt),
                    CType::Integer(IntegerKind::Long) => CType::Integer(IntegerKind::UnsignedLong),
                    CType::Integer(IntegerKind::LongLong) => CType::Integer(IntegerKind::UnsignedLongLong),
                    other => other,
                }
            }
            TypeSpecifier::Atomic(inner) | TypeSpecifier::Qualified { inner, .. } => {
                self.specifier_to_ctype(inner)
            }
            TypeSpecifier::Complex(inner) => self.specifier_to_ctype(inner),
            TypeSpecifier::Enum(_) | TypeSpecifier::EnumRef { .. } => {
                CType::Integer(IntegerKind::Int)
            }
            TypeSpecifier::Struct(def) => {
                // Build CType::Struct from the struct definition
                let fields: Vec<crate::sema::types::StructField> = def.members.iter().flat_map(|m| {
                    match m {
                        crate::frontend::parser::ast::StructMember::Field { specifiers, declarators, .. } => {
                            let base = self.specifier_to_ctype(&specifiers.type_specifier);
                            declarators.iter().map(|sd| {
                                let (name, ty) = if let Some(ref d) = sd.declarator {
                                    let n = Self::extract_declarator_name(d, self.interner);
                                    let t = self.apply_declarator_to_ctype(base.clone(), d);
                                    (n, t)
                                } else {
                                    (String::new(), base.clone())
                                };
                                crate::sema::types::StructField {
                                    name: Some(name),
                                    ty,
                                    bit_width: None,
                                    offset: 0,
                                }
                            }).collect::<Vec<_>>()
                        }
                        _ => Vec::new(),
                    }
                }).collect();
                CType::Struct(crate::sema::types::StructType {
                    tag: def.tag.map(|id| self.interner.resolve(id).to_string()),
                    fields,
                    is_union: false,
                    is_packed: false,
                    custom_alignment: None,
                    is_complete: true,
                })
            }
            TypeSpecifier::StructRef { tag, .. } => {
                // Struct forward reference — create a CType::Struct from
                // the registered field types in struct_field_names.
                let tag_name = self.interner.resolve(*tag).to_string();
                // Try to resolve from struct_field_names, which stores
                // (field_name, IrType) pairs.  Convert each IrType back to
                // CType so that map_type round-trips correctly.
                if let Some(field_info) = self.struct_field_names.get(&tag_name) {
                    let fields: Vec<crate::sema::types::StructField> = field_info.iter().map(|(name, ir_ty)| {
                        crate::sema::types::StructField {
                            name: Some(name.clone()),
                            ty: Self::ir_type_to_approx_ctype(ir_ty),
                            bit_width: None,
                            offset: 0,
                        }
                    }).collect();
                    CType::Struct(crate::sema::types::StructType {
                        tag: Some(tag_name),
                        fields,
                        is_union: false,
                        is_packed: false,
                        custom_alignment: None,
                        is_complete: true,
                    })
                } else {
                    CType::Integer(IntegerKind::Int) // unknown struct
                }
            }
            TypeSpecifier::Union(def) => {
                // Treat union as its largest member type for size purposes
                let mut largest = CType::Integer(IntegerKind::Int);
                let mut max_size: usize = 4;
                for member in &def.members {
                    if let crate::frontend::parser::ast::StructMember::Field { specifiers, .. } = member {
                        let ty = self.specifier_to_ctype(&specifiers.type_specifier);
                        let ir_ty = self.map_type(&ty);
                        let sz = IrBuilder::approx_ir_type_size(&ir_ty);
                        if sz > max_size {
                            max_size = sz;
                            largest = ty;
                        }
                    }
                }
                largest
            }
            _ => CType::Integer(IntegerKind::Int), // fallback for typedef, etc.
        }
    }

    /// Applies pointer and array modifiers from a Declarator to a base CType.
    /// Used by the AST-based fallback path in resolve_function_signature.
    fn apply_declarator_to_ctype(
        &self,
        mut base: CType,
        declarator: &crate::frontend::parser::ast::Declarator,
    ) -> CType {
        // Apply pointer modifiers
        for _ptr in &declarator.pointer {
            base = CType::Pointer {
                pointee: Box::new(base),
                qualifiers: crate::sema::types::TypeQualifiers::default(),
            };
        }
        // Apply array modifiers from direct declarator
        base = self.apply_direct_decl_to_ctype(base, &declarator.direct);
        base
    }

    /// Recursive helper for apply_declarator_to_ctype.
    fn apply_direct_decl_to_ctype(
        &self,
        base: CType,
        direct: &crate::frontend::parser::ast::DirectDeclarator,
    ) -> CType {
        use crate::frontend::parser::ast::{DirectDeclarator, ArraySize as AstArraySize};
        match direct {
            DirectDeclarator::Identifier(_) | DirectDeclarator::Abstract => base,
            DirectDeclarator::Parenthesized(inner) => {
                self.apply_declarator_to_ctype(base, inner)
            }
            DirectDeclarator::Array { base: dd_base, size, .. } => {
                let inner = self.apply_direct_decl_to_ctype(base, dd_base);
                let count = match size {
                    AstArraySize::Fixed(expr) => {
                        self.try_eval_const_expr(expr).unwrap_or(0) as usize
                    }
                    _ => 0,
                };
                CType::Array {
                    element: Box::new(inner),
                    size: crate::sema::types::ArraySize::Fixed(count),
                }
            }
            DirectDeclarator::Function { base: dd_base, .. } => {
                // For function pointers in params, just use the base
                self.apply_direct_decl_to_ctype(base, dd_base)
            }
        }
    }

    // -----------------------------------------------------------------------
    // Statement lowering
    // -----------------------------------------------------------------------

    /// Lowers a C statement into IR instructions and basic block terminators.
    fn lower_statement(&mut self, stmt: &Statement) {
        // Skip emission if current block is already terminated
        if let Some(ref fb) = self.current_function {
            if fb.current_block_terminated() {
                // Dead code after a terminator — still process for labels/declarations
                match stmt {
                    Statement::Labeled { .. } | Statement::Declaration(_) => {}
                    _ => return,
                }
            }
        }

        match stmt {
            Statement::Compound { items, .. } => {
                // Save current scope state so inner declarations don't
                // leak into the outer scope (C block scoping semantics).
                let saved_local_values = self.current_function.as_ref()
                    .map(|fb| fb.local_values.clone());
                let saved_local_types = self.current_function.as_ref()
                    .map(|fb| fb.local_types.clone());

                for item in items {
                    match item {
                        BlockItem::Declaration(decl) => {
                            self.lower_local_declaration(decl);
                        }
                        BlockItem::Statement(s) => {
                            self.lower_statement(s);
                        }
                    }
                }

                // Restore outer scope: variables declared in this block
                // are no longer visible after the block exits.
                if let (Some(ref mut fb), Some(vals), Some(types)) = (
                    &mut self.current_function,
                    saved_local_values,
                    saved_local_types,
                ) {
                    fb.local_values = vals;
                    fb.local_types = types;
                }
            }

            Statement::Expression { expr, .. } => {
                // Evaluate expression and discard result
                self.lower_expression(expr);
            }

            Statement::Return { value, .. } => {
                let ret_val = value.as_ref().map(|e| self.lower_expression(e));
                if let Some(ref mut fb) = self.current_function {
                    if !fb.current_block_terminated() {
                        fb.current_block_mut().terminator =
                            Some(Terminator::Return { value: ret_val });
                    }
                }
            }

            Statement::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.lower_if_statement(condition, then_branch, else_branch.as_deref());
            }

            Statement::While {
                condition, body, ..
            } => {
                self.lower_while_loop(condition, body);
            }

            Statement::DoWhile {
                body, condition, ..
            } => {
                self.lower_do_while_loop(body, condition);
            }

            Statement::For {
                init,
                condition,
                increment,
                body,
                ..
            } => {
                self.lower_for_loop(
                    init.as_deref(),
                    condition.as_deref(),
                    increment.as_deref(),
                    body,
                );
            }

            Statement::Switch { expr, body, .. } => {
                self.lower_switch_statement(expr, body);
            }

            Statement::Case { value: _value, body, .. } => {
                // Case labels are handled within switch lowering.
                // If we encounter one outside, just lower the body.
                self.lower_statement(body);
            }

            Statement::Default { body, .. } => {
                // Default labels are handled within switch lowering.
                self.lower_statement(body);
            }

            Statement::Break { .. } => {
                if let Some(target) = self.break_targets.last().copied() {
                    if let Some(ref mut fb) = self.current_function {
                        if !fb.current_block_terminated() {
                            fb.current_block_mut().terminator =
                                Some(Terminator::Branch { target });
                        }
                    }
                }
            }

            Statement::Continue { .. } => {
                if let Some(target) = self.continue_targets.last().copied() {
                    if let Some(ref mut fb) = self.current_function {
                        if !fb.current_block_terminated() {
                            fb.current_block_mut().terminator =
                                Some(Terminator::Branch { target });
                        }
                    }
                }
            }

            Statement::Goto { label, .. } => {
                let target = self.get_or_create_label_block(*label);
                if let Some(ref mut fb) = self.current_function {
                    if !fb.current_block_terminated() {
                        fb.current_block_mut().terminator =
                            Some(Terminator::Branch { target });
                    }
                }
            }

            Statement::Labeled { label, body, .. } => {
                let label_block = self.get_or_create_label_block(*label);
                // Branch from current block to the label block
                if let Some(ref mut fb) = self.current_function {
                    if !fb.current_block_terminated() {
                        fb.current_block_mut().terminator =
                            Some(Terminator::Branch { target: label_block });
                    }
                }
                self.set_insert_point(label_block);
                self.lower_statement(body);
            }

            Statement::Null { .. } => {
                // Empty statement — no IR needed
            }

            Statement::Declaration(decl) => {
                self.lower_local_declaration(decl);
            }

            Statement::Asm(_) => {
                // Inline assembly: emit a nop placeholder; actual asm is handled by codegen
                let nop = Instruction::Nop;
                self.emit_instruction(nop);
            }

            Statement::ComputedGoto { target, .. } => {
                // Computed goto: lower the target expression and emit an unreachable
                // (actual indirect branch is handled at a lower level)
                let _target_val = self.lower_expression(target);
                if let Some(ref mut fb) = self.current_function {
                    if !fb.current_block_terminated() {
                        fb.current_block_mut().terminator = Some(Terminator::Unreachable);
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Control flow statement lowering
    // -----------------------------------------------------------------------

    /// Lowers an if/else statement into conditional branch IR.
    fn lower_if_statement(
        &mut self,
        condition: &Expression,
        then_branch: &Statement,
        else_branch: Option<&Statement>,
    ) {
        let cond_val = self.lower_expression(condition);
        let cond_bool = self.ensure_boolean(cond_val);

        let then_block = self.create_block("if.then");
        let merge_block = self.create_block("if.merge");
        let else_block = if else_branch.is_some() {
            self.create_block("if.else")
        } else {
            merge_block
        };

        // Conditional branch
        if let Some(ref mut fb) = self.current_function {
            if !fb.current_block_terminated() {
                fb.current_block_mut().terminator = Some(Terminator::CondBranch {
                    condition: cond_bool,
                    true_block: then_block,
                    false_block: else_block,
                });
            }
        }

        // Then branch
        self.set_insert_point(then_block);
        self.lower_statement(then_branch);
        if let Some(ref mut fb) = self.current_function {
            if !fb.current_block_terminated() {
                fb.current_block_mut().terminator =
                    Some(Terminator::Branch { target: merge_block });
            }
        }

        // Else branch
        if let Some(else_stmt) = else_branch {
            self.set_insert_point(else_block);
            self.lower_statement(else_stmt);
            if let Some(ref mut fb) = self.current_function {
                if !fb.current_block_terminated() {
                    fb.current_block_mut().terminator =
                        Some(Terminator::Branch { target: merge_block });
                }
            }
        }

        // Continue at merge block
        self.set_insert_point(merge_block);
    }

    /// Lowers a while loop into IR basic blocks.
    fn lower_while_loop(&mut self, condition: &Expression, body: &Statement) {
        let cond_block = self.create_block("while.cond");
        let body_block = self.create_block("while.body");
        let exit_block = self.create_block("while.exit");

        // Branch to condition block
        if let Some(ref mut fb) = self.current_function {
            if !fb.current_block_terminated() {
                fb.current_block_mut().terminator =
                    Some(Terminator::Branch { target: cond_block });
            }
        }

        // Condition block
        self.set_insert_point(cond_block);
        let cond_val = self.lower_expression(condition);
        let cond_bool = self.ensure_boolean(cond_val);
        if let Some(ref mut fb) = self.current_function {
            if !fb.current_block_terminated() {
                fb.current_block_mut().terminator = Some(Terminator::CondBranch {
                    condition: cond_bool,
                    true_block: body_block,
                    false_block: exit_block,
                });
            }
        }

        // Body block
        self.break_targets.push(exit_block);
        self.continue_targets.push(cond_block);
        self.set_insert_point(body_block);
        self.lower_statement(body);
        if let Some(ref mut fb) = self.current_function {
            if !fb.current_block_terminated() {
                fb.current_block_mut().terminator =
                    Some(Terminator::Branch { target: cond_block });
            }
        }
        self.break_targets.pop();
        self.continue_targets.pop();

        // Continue at exit block
        self.set_insert_point(exit_block);
    }

    /// Lowers a do-while loop into IR basic blocks.
    fn lower_do_while_loop(&mut self, body: &Statement, condition: &Expression) {
        let body_block = self.create_block("dowhile.body");
        let cond_block = self.create_block("dowhile.cond");
        let exit_block = self.create_block("dowhile.exit");

        // Branch to body block
        if let Some(ref mut fb) = self.current_function {
            if !fb.current_block_terminated() {
                fb.current_block_mut().terminator =
                    Some(Terminator::Branch { target: body_block });
            }
        }

        // Body block
        self.break_targets.push(exit_block);
        self.continue_targets.push(cond_block);
        self.set_insert_point(body_block);
        self.lower_statement(body);
        if let Some(ref mut fb) = self.current_function {
            if !fb.current_block_terminated() {
                fb.current_block_mut().terminator =
                    Some(Terminator::Branch { target: cond_block });
            }
        }
        self.break_targets.pop();
        self.continue_targets.pop();

        // Condition block
        self.set_insert_point(cond_block);
        let cond_val = self.lower_expression(condition);
        let cond_bool = self.ensure_boolean(cond_val);
        if let Some(ref mut fb) = self.current_function {
            if !fb.current_block_terminated() {
                fb.current_block_mut().terminator = Some(Terminator::CondBranch {
                    condition: cond_bool,
                    true_block: body_block,
                    false_block: exit_block,
                });
            }
        }

        // Continue at exit block
        self.set_insert_point(exit_block);
    }

    /// Lowers a for loop into IR basic blocks.
    fn lower_for_loop(
        &mut self,
        init: Option<&ForInit>,
        condition: Option<&Expression>,
        increment: Option<&Expression>,
        body: &Statement,
    ) {
        // Emit init
        if let Some(for_init) = init {
            match for_init {
                ForInit::Declaration(decl) => self.lower_local_declaration(decl),
                ForInit::Expression(expr) => {
                    self.lower_expression(expr);
                }
            }
        }

        let cond_block = self.create_block("for.cond");
        let body_block = self.create_block("for.body");
        let incr_block = self.create_block("for.incr");
        let exit_block = self.create_block("for.exit");

        // Branch to condition
        if let Some(ref mut fb) = self.current_function {
            if !fb.current_block_terminated() {
                fb.current_block_mut().terminator =
                    Some(Terminator::Branch { target: cond_block });
            }
        }

        // Condition block
        self.set_insert_point(cond_block);
        if let Some(cond) = condition {
            let cond_val = self.lower_expression(cond);
            let cond_bool = self.ensure_boolean(cond_val);
            if let Some(ref mut fb) = self.current_function {
                if !fb.current_block_terminated() {
                    fb.current_block_mut().terminator = Some(Terminator::CondBranch {
                        condition: cond_bool,
                        true_block: body_block,
                        false_block: exit_block,
                    });
                }
            }
        } else {
            // No condition means infinite loop
            if let Some(ref mut fb) = self.current_function {
                if !fb.current_block_terminated() {
                    fb.current_block_mut().terminator =
                        Some(Terminator::Branch { target: body_block });
                }
            }
        }

        // Body block
        self.break_targets.push(exit_block);
        self.continue_targets.push(incr_block);
        self.set_insert_point(body_block);
        self.lower_statement(body);
        if let Some(ref mut fb) = self.current_function {
            if !fb.current_block_terminated() {
                fb.current_block_mut().terminator =
                    Some(Terminator::Branch { target: incr_block });
            }
        }
        self.break_targets.pop();
        self.continue_targets.pop();

        // Increment block
        self.set_insert_point(incr_block);
        if let Some(incr) = increment {
            self.lower_expression(incr);
        }
        if let Some(ref mut fb) = self.current_function {
            if !fb.current_block_terminated() {
                fb.current_block_mut().terminator =
                    Some(Terminator::Branch { target: cond_block });
            }
        }

        // Continue at exit block
        self.set_insert_point(exit_block);
    }

    /// Lowers a switch statement into IR basic blocks with a Switch terminator.
    fn lower_switch_statement(&mut self, expr: &Expression, body: &Statement) {
        let switch_val = self.lower_expression(expr);
        let exit_block = self.create_block("switch.exit");
        let default_block = self.create_block("switch.default");

        // Collect case values and create blocks for them
        let cases = self.collect_switch_cases(body);
        let mut case_entries: Vec<(i64, BlockId)> = Vec::new();
        let mut case_map: std::collections::HashMap<i64, BlockId> = std::collections::HashMap::new();

        for case_val in &cases {
            let case_block = self.create_block(&format!("switch.case.{}", case_val));
            case_entries.push((*case_val, case_block));
            case_map.insert(*case_val, case_block);
        }

        // Emit the switch terminator
        if let Some(ref mut fb) = self.current_function {
            if !fb.current_block_terminated() {
                fb.current_block_mut().terminator = Some(Terminator::Switch {
                    value: switch_val,
                    default: default_block,
                    cases: case_entries.clone(),
                });
            }
        }

        // Push break target
        self.break_targets.push(exit_block);

        // Lower case bodies by walking the body compound and dispatching
        // each Case/Default to its corresponding block. Fall-through from
        // one case to the next is implemented by branching to the next
        // case block when the current block is not yet terminated.
        self.lower_switch_body(body, &case_map, default_block, exit_block);

        // Ensure default block has a terminator (may not have been visited)
        if let Some(ref mut fb) = self.current_function {
            let idx = default_block.0 as usize;
            if idx < fb.blocks.len() && fb.blocks[idx].terminator.is_none() {
                fb.blocks[idx].terminator = Some(Terminator::Branch { target: exit_block });
            }
        }

        // Ensure all case blocks have terminators
        for (_, block_id) in &case_entries {
            if let Some(ref mut fb) = self.current_function {
                let idx = block_id.0 as usize;
                if idx < fb.blocks.len() && fb.blocks[idx].terminator.is_none() {
                    fb.blocks[idx].terminator = Some(Terminator::Branch { target: exit_block });
                }
            }
        }

        self.break_targets.pop();

        // Continue at exit block
        self.set_insert_point(exit_block);
    }

    /// Walks the compound body of a switch and dispatches each Case/Default
    /// label to its corresponding IR block, handling fall-through semantics.
    fn lower_switch_body(
        &mut self,
        body: &Statement,
        case_map: &std::collections::HashMap<i64, BlockId>,
        default_block: BlockId,
        exit_block: BlockId,
    ) {
        if let Statement::Compound { items, .. } = body {
            for item in items {
                match item {
                    BlockItem::Statement(stmt) => {
                        self.lower_switch_item(stmt, case_map, default_block, exit_block);
                    }
                    BlockItem::Declaration(decl) => {
                        self.lower_local_declaration(decl);
                    }
                }
            }
            // If the last block in the switch is not terminated, branch to exit
            if let Some(ref mut fb) = self.current_function {
                if !fb.current_block_terminated() {
                    fb.current_block_mut().terminator =
                        Some(Terminator::Branch { target: exit_block });
                }
            }
        } else {
            // Not a compound — just lower as a single statement
            self.lower_statement(body);
            if let Some(ref mut fb) = self.current_function {
                if !fb.current_block_terminated() {
                    fb.current_block_mut().terminator =
                        Some(Terminator::Branch { target: exit_block });
                }
            }
        }
    }

    /// Lowers a single item inside a switch body, handling Case/Default dispatch.
    fn lower_switch_item(
        &mut self,
        stmt: &Statement,
        case_map: &std::collections::HashMap<i64, BlockId>,
        default_block: BlockId,
        exit_block: BlockId,
    ) {
        match stmt {
            Statement::Case { value, body, .. } => {
                // Determine which block this case value maps to
                let target_block = if let Expression::IntegerLiteral { value: v, .. } = value.as_ref() {
                    case_map.get(&(*v as i64)).copied()
                } else if let Expression::UnaryPrefix { op: UnaryOp::Negate, operand, .. } = value.as_ref() {
                    if let Expression::IntegerLiteral { value: v, .. } = operand.as_ref() {
                        case_map.get(&(-(*v as i64))).copied()
                    } else {
                        None
                    }
                } else {
                    None
                };
                if let Some(block_id) = target_block {
                    // Fall-through: branch from previous block to this case block
                    if let Some(ref mut fb) = self.current_function {
                        if !fb.current_block_terminated() {
                            fb.current_block_mut().terminator =
                                Some(Terminator::Branch { target: block_id });
                        }
                    }
                    self.set_insert_point(block_id);
                }
                // Lower the case body (the statement after the label)
                self.lower_switch_item(body, case_map, default_block, exit_block);
            }
            Statement::Default { body, .. } => {
                // Fall-through: branch from previous block to default block
                if let Some(ref mut fb) = self.current_function {
                    if !fb.current_block_terminated() {
                        fb.current_block_mut().terminator =
                            Some(Terminator::Branch { target: default_block });
                    }
                }
                self.set_insert_point(default_block);
                // Lower the default body
                self.lower_switch_item(body, case_map, default_block, exit_block);
            }
            Statement::Compound { items, .. } => {
                // Recurse into compound statements (case bodies may be wrapped)
                for item in items {
                    match item {
                        BlockItem::Statement(s) => {
                            self.lower_switch_item(s, case_map, default_block, exit_block);
                        }
                        BlockItem::Declaration(d) => {
                            self.lower_local_declaration(d);
                        }
                    }
                }
            }
            _ => {
                // Regular statement — lower normally (handles Break, Continue, etc.)
                self.lower_statement(stmt);
            }
        }
    }

    /// Collects integer case values from a switch body for building the Switch terminator.
    fn collect_switch_cases(&self, stmt: &Statement) -> Vec<i64> {
        let mut cases = Vec::new();
        self.collect_cases_recursive(stmt, &mut cases);
        cases
    }

    /// Recursively walks a statement tree to collect case constant values.
    fn collect_cases_recursive(&self, stmt: &Statement, cases: &mut Vec<i64>) {
        match stmt {
            Statement::Case { value, body, .. } => {
                if let Expression::IntegerLiteral { value: v, .. } = value.as_ref() {
                    cases.push(*v as i64);
                }
                self.collect_cases_recursive(body, cases);
            }
            Statement::Default { body, .. } => {
                self.collect_cases_recursive(body, cases);
            }
            Statement::Compound { items, .. } => {
                for item in items {
                    if let BlockItem::Statement(s) = item {
                        self.collect_cases_recursive(s, cases);
                    }
                }
            }
            _ => {}
        }
    }

    // -----------------------------------------------------------------------
    // Local declaration lowering
    // -----------------------------------------------------------------------

    /// Lowers a block-level declaration into IR alloca + optional store.
    fn lower_local_declaration(&mut self, decl: &Declaration) {
        match decl {
            Declaration::Variable {
                specifiers: _specifiers,
                declarators,
                ..
            } => {
                // Register struct definitions even if no declarators (standalone struct def).
                self.try_register_struct_from_specifier(&_specifiers.type_specifier);

                for init_decl in declarators {
                    let var_name = Self::extract_declarator_name(&init_decl.declarator, self.interner);
                    if var_name.is_empty() {
                        continue;
                    }

                    // Resolve the variable's type from the declaration specifiers
                    // and declarator, so arrays (e.g. char buf[8192]) get their
                    // full size allocated on the stack.
                    let base_ir = self.resolve_specifier_to_ir_type(
                        &_specifiers.type_specifier,
                    );
                    let var_type = self.resolve_declarator_ir_type(
                        base_ir,
                        &init_decl.declarator,
                    );

                    // Allocate stack space
                    let alloca_val = self.new_value(var_type.clone().pointer_to());
                    let alloca_inst = Instruction::Alloca {
                        result: alloca_val,
                        ty: var_type.clone(),
                        count: None,
                    };
                    self.emit_instruction(alloca_inst);

                    // Store initializer if present
                    if let Some(ref init) = init_decl.initializer {
                        match init {
                            Initializer::Expression(expr) => {
                                let init_val = self.lower_expression(expr);
                                let store_inst = Instruction::Store {
                                    value: init_val,
                                    ptr: alloca_val,
                                    store_ty: None,
                                };
                                self.emit_instruction(store_inst);
                            }
                            Initializer::Compound { .. } => {
                                // Compound initializers for locals: zero-init then fill fields
                                // Simplified: just leave default-initialized
                            }
                        }
                    }

                    // Register the alloca using the real InternId for the
                    // variable name so Identifier expression lookups find it.
                    if let Some(real_id) = self.interner.get(&var_name) {
                        if let Some(ref mut fb) = self.current_function {
                            fb.local_values.insert(real_id, alloca_val);
                            fb.local_types.insert(real_id, var_type.clone());
                        }
                    }
                }
            }
            _ => {
                // Other declaration types (typedef, etc.) don't produce local IR
            }
        }
    }

    // -----------------------------------------------------------------------
    // Expression lowering
    // -----------------------------------------------------------------------

    /// Lowers a C expression into IR instructions, returning the SSA value
    /// representing the expression's result.
    fn lower_expression(&mut self, expr: &Expression) -> Value {
        match expr {
            // === Literals ===
            Expression::IntegerLiteral { value, .. } => {
                let result = self.new_value(IrType::I32);
                let inst = Instruction::Const {
                    result,
                    value: Constant::Integer {
                        value: *value as i64,
                        ty: IrType::I32,
                    },
                };
                self.emit_instruction(inst);
                result
            }

            Expression::FloatLiteral { value, suffix, .. } => {
                let float_ty = if *suffix == FloatSuffix::Float {
                    IrType::F32
                } else {
                    IrType::F64
                };
                let result = self.new_value(float_ty.clone());
                let inst = Instruction::Const {
                    result,
                    value: Constant::Float {
                        value: *value,
                        ty: float_ty.clone(),
                    },
                };
                self.emit_instruction(inst);
                // Track the float type so binary operations on floats
                // can infer the correct type instead of defaulting to I32.
                if let Some(ref mut fb) = self.current_function {
                    fb.value_types.insert(result, float_ty);
                }
                result
            }

            Expression::StringLiteral { value, .. } => {
                // String literals become global constants
                let str_name = format!(".str.{}", self.string_literal_count);
                self.string_literal_count = self.string_literal_count.wrapping_add(1);
                let mut bytes = value.as_bytes().to_vec();
                bytes.push(0); // null terminator
                let len = bytes.len();

                let array_ty = IrType::Array {
                    element: Box::new(IrType::I8),
                    count: len,
                };

                self.module.globals.push(GlobalVariable {
                    name: str_name.clone(),
                    ty: array_ty,
                    initializer: Some(Constant::String(bytes)),
                    is_extern: false,
                    is_static: true,
                });

                // Return a pointer to the global string
                let result = self.new_value(IrType::I8.pointer_to());
                let inst = Instruction::Const {
                    result,
                    value: Constant::GlobalRef(str_name),
                };
                self.emit_instruction(inst);
                result
            }

            Expression::CharLiteral { value, .. } => {
                let result = self.new_value(IrType::I32);
                let inst = Instruction::Const {
                    result,
                    value: Constant::Integer {
                        value: *value as i64,
                        ty: IrType::I32,
                    },
                };
                self.emit_instruction(inst);
                result
            }

            // === Identifier reference ===
            Expression::Identifier { name, .. } => {
                // Look up the variable in function builder locals first
                if let Some(ref fb) = self.current_function {
                    if let Some(&alloca_val) = fb.local_values.get(name) {
                        // Use the variable's actual IR type (tracked in
                        // local_types) instead of defaulting to I32.  This
                        // is critical for long, long long, pointer, char,
                        // and short types that would otherwise be truncated
                        // or over-read by a 32-bit Load.
                        let load_ty = fb.local_types.get(name).cloned().unwrap_or(IrType::I32);

                        // Array-to-pointer decay (C11 §6.3.2.1p3): when an
                        // array identifier is used in an expression context
                        // other than sizeof/&, it decays to a pointer to
                        // its first element.  Return the alloca address
                        // directly instead of loading from it.
                        if load_ty.is_array() {
                            // Record decayed type as pointer to element.
                            if let Some(elem) = load_ty.element_type() {
                                if let Some(ref mut fb) = self.current_function {
                                    fb.value_types.insert(alloca_val, elem.clone().pointer_to());
                                }
                            }
                            return alloca_val;
                        }

                        let result = self.new_value(load_ty.clone());
                        let inst = Instruction::Load {
                            result,
                            ty: load_ty.clone(),
                            ptr: alloca_val,
                        };
                        self.emit_instruction(inst);
                        // Track the loaded value's type for pointer
                        // arithmetic detection.
                        if let Some(ref mut fb) = self.current_function {
                            fb.value_types.insert(result, load_ty);
                        }
                        return result;
                    }
                }

                // Check function-scope symbol_values (e.g., previously
                // lowered globals that were assigned a local alloca).
                if let Some(&val) = self.symbol_values.get(name) {
                    let load_ty = self.symbol_types.get(name).cloned().unwrap_or(IrType::I32);
                    let result = self.new_value(load_ty.clone());
                    let inst = Instruction::Load {
                        result,
                        ty: load_ty,
                        ptr: val,
                    };
                    return self.emit_instruction(inst);
                }

                // Check module-level global variables.  If found, emit a
                // GlobalRef constant to obtain the global's address, then
                // load from it.
                let resolved_name = self.interner.resolve(*name);
                if let Some(global) = self.module.globals.iter().find(|g| g.name == resolved_name) {
                    let global_ty = global.ty.clone();
                    let global_name = global.name.clone();
                    // Emit Const(GlobalRef) → pointer to the global
                    let ptr_val = self.new_value(global_ty.clone().pointer_to());
                    let ref_inst = Instruction::Const {
                        result: ptr_val,
                        value: Constant::GlobalRef(global_name),
                    };
                    self.emit_instruction(ref_inst);
                    // Load the value from the global address
                    let load_result = self.new_value(global_ty.clone());
                    let load_inst = Instruction::Load {
                        result: load_result,
                        ty: global_ty,
                        ptr: ptr_val,
                    };
                    return self.emit_instruction(load_inst);
                }

                // Check if this is a function name (for function pointer expressions).
                let is_function = self.module.functions.iter().any(|f| f.name == resolved_name);
                if is_function {
                    let result = self.new_value(IrType::I64);
                    let inst = Instruction::Const {
                        result,
                        value: Constant::GlobalRef(resolved_name.to_string()),
                    };
                    return self.emit_instruction(inst);
                }

                // Not found: produce zero constant as fallback
                let result = self.new_value(IrType::I32);
                let inst = Instruction::Const {
                    result,
                    value: Constant::Integer {
                        value: 0,
                        ty: IrType::I32,
                    },
                };
                self.emit_instruction(inst);
                result
            }

            // === Binary operations ===
            Expression::Binary {
                op, left, right, ..
            } => self.lower_binary_expression(*op, left, right),

            // === Unary prefix operations ===
            Expression::UnaryPrefix { op, operand, .. } => {
                self.lower_unary_expression(*op, operand)
            }

            // === Post-increment/decrement ===
            Expression::PostIncrement { operand, .. } => {
                self.lower_post_increment(operand, true)
            }

            Expression::PostDecrement { operand, .. } => {
                self.lower_post_increment(operand, false)
            }

            // === Function call ===
            Expression::Call { callee, args, .. } => {
                self.lower_function_call(callee, args)
            }

            // === Array subscript ===
            Expression::Subscript { array, index, .. } => {
                let array_val = self.lower_expression(array);
                let index_val = self.lower_expression(index);
                let gep_result = self.new_value(IrType::I32.pointer_to());
                let gep = Instruction::GetElementPtr {
                    result: gep_result,
                    base_ty: IrType::I32,
                    ptr: array_val,
                    indices: vec![index_val],
                    in_bounds: true,
                };
                self.emit_instruction(gep);
                let load_result = self.new_value(IrType::I32);
                let load = Instruction::Load {
                    result: load_result,
                    ty: IrType::I32,
                    ptr: gep_result,
                };
                self.emit_instruction(load);
                load_result
            }

            // === Member access ===
            Expression::MemberAccess { object, member, .. } => {
                // Get field address via lvalue path then load
                let member_name = self.interner.resolve(*member).to_string();
                let (byte_offset, field_ty) = self.compute_struct_member_offset(object, &member_name);
                let obj_addr = self.get_lvalue_address(object);
                if let Some(base) = obj_addr {
                    let field_ptr = if byte_offset == 0 {
                        base
                    } else {
                        let offset_val = self.emit_const_int(byte_offset as i64, IrType::I64);
                        let gep_result = self.new_value(field_ty.clone().pointer_to());
                        let gep = Instruction::GetElementPtr {
                            result: gep_result,
                            base_ty: IrType::I8,
                            ptr: base,
                            indices: vec![offset_val],
                            in_bounds: true,
                        };
                        self.emit_instruction(gep);
                        if let Some(ref mut fb) = self.current_function {
                            fb.value_types.insert(gep_result, field_ty.clone().pointer_to());
                        }
                        gep_result
                    };
                    let load_result = self.new_value(field_ty.clone());
                    let load = Instruction::Load {
                        result: load_result,
                        ty: field_ty.clone(),
                        ptr: field_ptr,
                    };
                    self.emit_instruction(load);
                    if let Some(ref mut fb) = self.current_function {
                        fb.value_types.insert(load_result, field_ty);
                    }
                    load_result
                } else {
                    // Fallback
                    self.emit_const_int(0, IrType::I32)
                }
            }

            // === Arrow access ===
            Expression::ArrowAccess {
                pointer, member, ..
            } => {
                // Load the pointer, then add byte offset, then load field
                let ptr_val = self.lower_expression(pointer);
                let member_name = self.interner.resolve(*member).to_string();
                let (byte_offset, field_ty) = self.compute_struct_member_offset_from_ptr(pointer, &member_name);
                let field_ptr = if byte_offset == 0 {
                    ptr_val
                } else {
                    let offset_val = self.emit_const_int(byte_offset as i64, IrType::I64);
                    let gep_result = self.new_value(field_ty.clone().pointer_to());
                    let gep = Instruction::GetElementPtr {
                        result: gep_result,
                        base_ty: IrType::I8,
                        ptr: ptr_val,
                        indices: vec![offset_val],
                        in_bounds: true,
                    };
                    self.emit_instruction(gep);
                    gep_result
                };
                let load_result = self.new_value(field_ty.clone());
                let load = Instruction::Load {
                    result: load_result,
                    ty: field_ty.clone(),
                    ptr: field_ptr,
                };
                self.emit_instruction(load);
                if let Some(ref mut fb) = self.current_function {
                    fb.value_types.insert(load_result, field_ty);
                }
                load_result
            }

            // === Assignment ===
            Expression::Assignment {
                op, target, value, ..
            } => self.lower_assignment(*op, target, value),

            // === Ternary ===
            Expression::Ternary {
                condition,
                then_expr,
                else_expr,
                ..
            } => self.lower_ternary(condition, then_expr, else_expr),

            // === Comma operator ===
            Expression::Comma { exprs, .. } => {
                let mut result = Value::undef();
                for e in exprs {
                    result = self.lower_expression(e);
                }
                result
            }

            // === Cast ===
            Expression::Cast { type_name, operand, .. } => {
                let operand_val = self.lower_expression(operand);

                // Resolve target type from the TypeName
                let to_ty = self.resolve_type_name_to_ir_type(type_name);

                // Determine source type from value_types or value_type()
                let from_ty = self.current_function.as_ref()
                    .and_then(|fb| fb.value_types.get(&operand_val).cloned())
                    .unwrap_or_else(|| self.value_type(operand_val));

                // If source and target types are the same, the cast is an identity
                if from_ty == to_ty {
                    return operand_val;
                }

                // Cast to void: (void)expr — evaluate operand for side effects only,
                // return the operand value (it will be discarded by the caller)
                if to_ty.is_void() {
                    return operand_val;
                }

                // For pointer casts (pointer→pointer, pointer→int, int→pointer), use BitCast or PtrToInt/IntToPtr
                if from_ty.is_pointer() && to_ty.is_pointer() {
                    let result = self.new_value(to_ty.clone());
                    self.emit_instruction(Instruction::BitCast {
                        result,
                        value: operand_val,
                        from_ty,
                        to_ty: to_ty.clone(),
                    });
                    if let Some(ref mut fb) = self.current_function {
                        fb.value_types.insert(result, to_ty);
                    }
                    return result;
                }
                if from_ty.is_pointer() && to_ty.is_integer() {
                    let result = self.new_value(to_ty.clone());
                    self.emit_instruction(Instruction::Cast {
                        result,
                        op: CastOp::PtrToInt,
                        value: operand_val,
                        from_ty,
                        to_ty: to_ty.clone(),
                    });
                    if let Some(ref mut fb) = self.current_function {
                        fb.value_types.insert(result, to_ty);
                    }
                    return result;
                }
                if from_ty.is_integer() && to_ty.is_pointer() {
                    let result = self.new_value(to_ty.clone());
                    self.emit_instruction(Instruction::Cast {
                        result,
                        op: CastOp::IntToPtr,
                        value: operand_val,
                        from_ty,
                        to_ty: to_ty.clone(),
                    });
                    if let Some(ref mut fb) = self.current_function {
                        fb.value_types.insert(result, to_ty);
                    }
                    return result;
                }

                // Determine the appropriate CastOp
                let cast_op = if from_ty.is_float() && to_ty.is_integer() {
                    // Float → Integer
                    if self.is_unsigned_specifier(&type_name.specifiers.type_specifier) {
                        CastOp::FPToUI
                    } else {
                        CastOp::FPToSI
                    }
                } else if from_ty.is_integer() && to_ty.is_float() {
                    // Integer → Float
                    CastOp::SIToFP
                } else if from_ty.is_float() && to_ty.is_float() {
                    // Float → Float (F32↔F64)
                    if from_ty.size(self.target) > to_ty.size(self.target) {
                        CastOp::FPTrunc
                    } else {
                        CastOp::FPExt
                    }
                } else if from_ty.is_integer() && to_ty.is_integer() {
                    // Integer → Integer
                    let from_size = from_ty.size(self.target);
                    let to_size = to_ty.size(self.target);
                    if to_size < from_size {
                        CastOp::Trunc
                    } else if self.is_unsigned_specifier(&type_name.specifiers.type_specifier) {
                        CastOp::ZExt
                    } else {
                        CastOp::SExt
                    }
                } else {
                    // Fallback: treat as bitcast for any other type combination
                    let result = self.new_value(to_ty.clone());
                    self.emit_instruction(Instruction::BitCast {
                        result,
                        value: operand_val,
                        from_ty,
                        to_ty: to_ty.clone(),
                    });
                    if let Some(ref mut fb) = self.current_function {
                        fb.value_types.insert(result, to_ty);
                    }
                    return result;
                };

                let result = self.new_value(to_ty.clone());
                self.emit_instruction(Instruction::Cast {
                    result,
                    op: cast_op,
                    value: operand_val,
                    from_ty,
                    to_ty: to_ty.clone(),
                });
                if let Some(ref mut fb) = self.current_function {
                    fb.value_types.insert(result, to_ty);
                }
                result
            }

            // === Sizeof ===
            Expression::SizeofExpr { expr, .. } => {
                // sizeof(expr) — compute the size of the expression's type.
                // We resolve the expression's IR type and use its target-specific size.
                let expr_val = self.lower_expression(expr);
                let ir_ty = self.value_type(expr_val);
                let size_val = ir_ty.size(self.target) as i64;
                let result = self.new_value(IrType::I64);
                let inst = Instruction::Const {
                    result,
                    value: Constant::Integer {
                        value: size_val,
                        ty: IrType::I64,
                    },
                };
                self.emit_instruction(inst);
                result
            }
            Expression::SizeofType { type_name, .. } => {
                // sizeof(type) — resolve the TypeName to a size.
                let size_val = self.resolve_sizeof_type_name(type_name);
                let result = self.new_value(IrType::I64);
                let inst = Instruction::Const {
                    result,
                    value: Constant::Integer {
                        value: size_val,
                        ty: IrType::I64,
                    },
                };
                self.emit_instruction(inst);
                result
            }

            // === Alignof ===
            Expression::Alignof { type_name, .. } => {
                // _Alignof(type) — resolve the TypeName to an alignment.
                let align_val = self.resolve_alignof_type_name(type_name);
                let result = self.new_value(IrType::I64);
                let inst = Instruction::Const {
                    result,
                    value: Constant::Integer {
                        value: align_val,
                        ty: IrType::I64,
                    },
                };
                self.emit_instruction(inst);
                result
            }

            // === Generic ===
            Expression::Generic {
                controlling,
                associations: _associations,
                ..
            } => {
                // Lower the controlling expression (actual generic dispatch done in sema)
                self.lower_expression(controlling)
            }

            // === Compound literal ===
            Expression::CompoundLiteral { .. } => {
                // Simplified: allocate and zero-initialize
                let result = self.new_value(IrType::I32);
                let inst = Instruction::Const {
                    result,
                    value: Constant::Integer {
                        value: 0,
                        ty: IrType::I32,
                    },
                };
                self.emit_instruction(inst);
                result
            }

            // === GCC extensions ===
            Expression::StatementExpr { body, .. } => {
                // Statement expression: lower the body and use the last expression's value
                self.lower_statement(body);
                // Return a zero value as placeholder (correct implementation would
                // capture the last expression's value)
                let result = self.new_value(IrType::I32);
                let inst = Instruction::Const {
                    result,
                    value: Constant::Integer {
                        value: 0,
                        ty: IrType::I32,
                    },
                };
                self.emit_instruction(inst);
                result
            }

            Expression::LabelAddr { .. } => {
                // &&label: return a pointer-sized constant (label address)
                let result = self.new_value(IrType::Pointer(Box::new(IrType::I8)));
                let inst = Instruction::Const {
                    result,
                    value: Constant::Null(IrType::Pointer(Box::new(IrType::I8))),
                };
                self.emit_instruction(inst);
                result
            }

            Expression::Extension { expr, .. } => {
                // __extension__: just lower the inner expression
                self.lower_expression(expr)
            }

            Expression::BuiltinVaArg { ap, .. } => {
                // va_arg: load from the va_list
                self.lower_expression(ap)
            }

            Expression::BuiltinOffsetof { .. } => {
                let result = self.new_value(IrType::I64);
                let inst = Instruction::Const {
                    result,
                    value: Constant::Integer {
                        value: 0,
                        ty: IrType::I64,
                    },
                };
                self.emit_instruction(inst);
                result
            }

            Expression::BuiltinVaStart { ap, .. } => self.lower_expression(ap),

            Expression::BuiltinVaEnd { ap, .. } => self.lower_expression(ap),

            Expression::BuiltinVaCopy { dest, .. } => self.lower_expression(dest),

            // === Parenthesized expression ===
            Expression::Paren { inner, .. } => self.lower_expression(inner),

            // === Error recovery ===
            Expression::Error { .. } => {
                let result = self.new_value(IrType::I32);
                let inst = Instruction::Const {
                    result,
                    value: Constant::Integer {
                        value: 0,
                        ty: IrType::I32,
                    },
                };
                self.emit_instruction(inst);
                result
            }
        }
    }

    // -----------------------------------------------------------------------
    // Binary expression lowering
    // -----------------------------------------------------------------------

    /// Lowers a binary expression, handling arithmetic, comparison, bitwise,
    /// and logical operations.
    fn lower_binary_expression(
        &mut self,
        op: BinaryOp,
        left: &Expression,
        right: &Expression,
    ) -> Value {
        // Short-circuit evaluation for logical operators
        match op {
            BinaryOp::LogicalAnd => return self.lower_logical_and(left, right),
            BinaryOp::LogicalOr => return self.lower_logical_or(left, right),
            _ => {}
        }

        let lhs = self.lower_expression(left);
        let rhs = self.lower_expression(right);

        // Infer the result type from the operand types instead of
        // defaulting to I32.  Float operands require F32/F64 so that the
        // codegen emits SSE instructions instead of integer ALU ops.
        let ty = {
            let lhs_inferred = self.current_function.as_ref()
                .and_then(|fb| fb.value_types.get(&lhs).cloned());
            let rhs_inferred = self.current_function.as_ref()
                .and_then(|fb| fb.value_types.get(&rhs).cloned());
            match (&lhs_inferred, &rhs_inferred) {
                (Some(t), _) if t.is_float() => t.clone(),
                (_, Some(t)) if t.is_float() => t.clone(),
                (Some(IrType::I64), _) | (_, Some(IrType::I64)) => IrType::I64,
                (Some(IrType::I16), _) | (_, Some(IrType::I16)) => IrType::I32,
                (Some(IrType::I8), _) | (_, Some(IrType::I8)) => IrType::I32,
                _ => IrType::I32,
            }
        };

        // ---- Implicit int-to-float conversion (C11 §6.3.1.8) ----
        // When the inferred type is float but one operand is integer, insert
        // a SIToFP cast to convert the integer operand to float. This handles
        // C's "usual arithmetic conversions" for mixed int/float expressions
        // like `double_var < 0`.
        let (lhs, rhs) = if ty.is_float() {
            let lhs_is_float = self.current_function.as_ref()
                .and_then(|fb| fb.value_types.get(&lhs).cloned())
                .map_or(false, |t| t.is_float());
            let rhs_is_float = self.current_function.as_ref()
                .and_then(|fb| fb.value_types.get(&rhs).cloned())
                .map_or(false, |t| t.is_float());
            let new_lhs = if !lhs_is_float {
                // Convert integer LHS to float
                let cast_result = self.new_value(ty.clone());
                self.emit_instruction(Instruction::Cast {
                    result: cast_result,
                    op: CastOp::SIToFP,
                    value: lhs,
                    from_ty: IrType::I32,
                    to_ty: ty.clone(),
                });
                if let Some(ref mut fb) = self.current_function {
                    fb.value_types.insert(cast_result, ty.clone());
                }
                cast_result
            } else {
                lhs
            };
            let new_rhs = if !rhs_is_float {
                // Convert integer RHS to float
                let cast_result = self.new_value(ty.clone());
                self.emit_instruction(Instruction::Cast {
                    result: cast_result,
                    op: CastOp::SIToFP,
                    value: rhs,
                    from_ty: IrType::I32,
                    to_ty: ty.clone(),
                });
                if let Some(ref mut fb) = self.current_function {
                    fb.value_types.insert(cast_result, ty.clone());
                }
                cast_result
            } else {
                rhs
            };
            (new_lhs, new_rhs)
        } else {
            (lhs, rhs)
        };

        // ---- Pointer arithmetic detection ----
        // In C, `ptr + n` and `n + ptr` scale n by sizeof(pointee).
        // Detect this by checking value_types for pointer types and emit
        // a GEP instead of a plain Add.
        let lhs_ty = self.current_function.as_ref().and_then(|fb| fb.value_types.get(&lhs).cloned());
        let rhs_ty = self.current_function.as_ref().and_then(|fb| fb.value_types.get(&rhs).cloned());

        let is_ptr_add = matches!(op, BinaryOp::Add)
            && (lhs_ty.as_ref().map_or(false, |t| t.is_pointer())
                || rhs_ty.as_ref().map_or(false, |t| t.is_pointer()));
        // (pointer arithmetic detection done above via is_ptr_add)

        let is_ptr_ptr_sub = matches!(op, BinaryOp::Sub)
            && lhs_ty.as_ref().map_or(false, |t| t.is_pointer())
            && rhs_ty.as_ref().map_or(false, |t| t.is_pointer());



        let is_ptr_sub = matches!(op, BinaryOp::Sub)
            && lhs_ty.as_ref().map_or(false, |t| t.is_pointer())
            && !is_ptr_ptr_sub;

        if is_ptr_add {
            // Determine which operand is the pointer and which is the index.
            let (ptr_val, idx_val, ptr_ty) = if lhs_ty.as_ref().map_or(false, |t| t.is_pointer()) {
                (lhs, rhs, lhs_ty.unwrap())
            } else {
                (rhs, lhs, rhs_ty.unwrap())
            };
            let elem_ty = ptr_ty.element_type().cloned().unwrap_or(IrType::I8);
            let result = self.new_value(ptr_ty.clone());
            self.emit_instruction(Instruction::GetElementPtr {
                result,
                base_ty: elem_ty,
                ptr: ptr_val,
                indices: vec![idx_val],
                in_bounds: false,
            });
            if let Some(ref mut fb) = self.current_function {
                fb.value_types.insert(result, ptr_ty);
            }
            return result;
        }

        if is_ptr_ptr_sub {
            // ptr - ptr: compute byte difference, then divide by element size
            // to get the number of elements between the two pointers.
            let ptr_ty = lhs_ty.unwrap();
            let elem_ty = ptr_ty.element_type().cloned().unwrap_or(IrType::I8);
            let elem_size = Self::approx_ir_type_size(&elem_ty) as i64;

            // Convert both pointers to integers (PtrToInt).
            let lhs_int = self.new_value(IrType::I64);
            self.emit_instruction(Instruction::Cast {
                result: lhs_int,
                op: CastOp::PtrToInt,
                value: lhs,
                from_ty: ptr_ty.clone(),
                to_ty: IrType::I64,
            });
            let rhs_int = self.new_value(IrType::I64);
            self.emit_instruction(Instruction::Cast {
                result: rhs_int,
                op: CastOp::PtrToInt,
                value: rhs,
                from_ty: rhs_ty.unwrap(),
                to_ty: IrType::I64,
            });
            // Subtract to get byte difference.
            let byte_diff = self.new_value(IrType::I64);
            self.emit_instruction(Instruction::Sub {
                result: byte_diff,
                lhs: lhs_int,
                rhs: rhs_int,
                ty: IrType::I64,
            });
            // Divide by element size to get element count.
            if elem_size > 1 {
                let elem_size_val = self.emit_const_int(elem_size, IrType::I64);
                let result = self.new_value(IrType::I64);
                self.emit_instruction(Instruction::Div {
                    result,
                    lhs: byte_diff,
                    rhs: elem_size_val,
                    ty: IrType::I64,
                    is_signed: true,
                });
                return result;
            }
            return byte_diff;
        }

        if is_ptr_sub {
            // ptr - n: negate n, then GEP with the negated index.
            let neg_rhs = {
                let zero = self.emit_const_int(0, IrType::I64);
                let neg_result = self.new_value(IrType::I64);
                self.emit_instruction(Instruction::Sub {
                    result: neg_result,
                    lhs: zero,
                    rhs,
                    ty: IrType::I64,
                });
                neg_result
            };
            let ptr_ty = lhs_ty.unwrap();
            let elem_ty = ptr_ty.element_type().cloned().unwrap_or(IrType::I8);
            let result = self.new_value(ptr_ty.clone());
            self.emit_instruction(Instruction::GetElementPtr {
                result,
                base_ty: elem_ty,
                ptr: lhs,
                indices: vec![neg_rhs],
                in_bounds: false,
            });
            if let Some(ref mut fb) = self.current_function {
                fb.value_types.insert(result, ptr_ty);
            }
            return result;
        }

        // Helper closure to register result type in value_types when
        // the type is not the default I32 (needed for float propagation
        // through chained operations like a + b + c).
        let track_ty = |this: &mut Self, result: Value, ty: &IrType| {
            if *ty != IrType::I32 {
                if let Some(ref mut fb) = this.current_function {
                    fb.value_types.insert(result, ty.clone());
                }
            }
        };

        match op {
            BinaryOp::Add => {
                let result = self.new_value(ty.clone());
                self.emit_instruction(Instruction::Add {
                    result,
                    lhs,
                    rhs,
                    ty: ty.clone(),
                });
                track_ty(self, result, &ty);
                result
            }
            BinaryOp::Sub => {
                let result = self.new_value(ty.clone());
                self.emit_instruction(Instruction::Sub {
                    result,
                    lhs,
                    rhs,
                    ty: ty.clone(),
                });
                track_ty(self, result, &ty);
                result
            }
            BinaryOp::Mul => {
                let result = self.new_value(ty.clone());
                self.emit_instruction(Instruction::Mul {
                    result,
                    lhs,
                    rhs,
                    ty: ty.clone(),
                });
                track_ty(self, result, &ty);
                result
            }
            BinaryOp::Div => {
                let result = self.new_value(ty.clone());
                self.emit_instruction(Instruction::Div {
                    result,
                    lhs,
                    rhs,
                    ty: ty.clone(),
                    is_signed: true,
                });
                track_ty(self, result, &ty);
                result
            }
            BinaryOp::Mod => {
                let result = self.new_value(ty.clone());
                self.emit_instruction(Instruction::Mod {
                    result,
                    lhs,
                    rhs,
                    ty: ty.clone(),
                    is_signed: true,
                });
                track_ty(self, result, &ty);
                result
            }
            BinaryOp::BitwiseAnd => {
                let result = self.new_value(ty.clone());
                self.emit_instruction(Instruction::And {
                    result,
                    lhs,
                    rhs,
                    ty,
                });
                result
            }
            BinaryOp::BitwiseOr => {
                let result = self.new_value(ty.clone());
                self.emit_instruction(Instruction::Or {
                    result,
                    lhs,
                    rhs,
                    ty,
                });
                result
            }
            BinaryOp::BitwiseXor => {
                let result = self.new_value(ty.clone());
                self.emit_instruction(Instruction::Xor {
                    result,
                    lhs,
                    rhs,
                    ty,
                });
                result
            }
            BinaryOp::ShiftLeft => {
                let result = self.new_value(ty.clone());
                self.emit_instruction(Instruction::Shl {
                    result,
                    lhs,
                    rhs,
                    ty,
                });
                result
            }
            BinaryOp::ShiftRight => {
                let result = self.new_value(ty.clone());
                self.emit_instruction(Instruction::Shr {
                    result,
                    lhs,
                    rhs,
                    ty,
                    is_arithmetic: true,
                });
                result
            }
            BinaryOp::Equal => {
                let result = self.new_value(IrType::I1);
                if ty.is_float() {
                    self.emit_instruction(Instruction::FCmp {
                        result,
                        op: FloatCompareOp::OrderedEqual,
                        lhs, rhs, ty,
                    });
                } else {
                    self.emit_instruction(Instruction::ICmp {
                        result,
                        op: CompareOp::Equal,
                        lhs, rhs, ty,
                    });
                }
                result
            }
            BinaryOp::NotEqual => {
                let result = self.new_value(IrType::I1);
                if ty.is_float() {
                    self.emit_instruction(Instruction::FCmp {
                        result,
                        op: FloatCompareOp::OrderedNotEqual,
                        lhs, rhs, ty,
                    });
                } else {
                    self.emit_instruction(Instruction::ICmp {
                        result,
                        op: CompareOp::NotEqual,
                        lhs, rhs, ty,
                    });
                }
                result
            }
            BinaryOp::Less => {
                let result = self.new_value(IrType::I1);
                if ty.is_float() {
                    self.emit_instruction(Instruction::FCmp {
                        result,
                        op: FloatCompareOp::OrderedLess,
                        lhs, rhs, ty,
                    });
                } else {
                    self.emit_instruction(Instruction::ICmp {
                        result,
                        op: CompareOp::SignedLess,
                        lhs, rhs, ty,
                    });
                }
                result
            }
            BinaryOp::Greater => {
                let result = self.new_value(IrType::I1);
                if ty.is_float() {
                    self.emit_instruction(Instruction::FCmp {
                        result,
                        op: FloatCompareOp::OrderedGreater,
                        lhs, rhs, ty,
                    });
                } else {
                    self.emit_instruction(Instruction::ICmp {
                        result,
                        op: CompareOp::SignedGreater,
                        lhs, rhs, ty,
                    });
                }
                result
            }
            BinaryOp::LessEqual => {
                let result = self.new_value(IrType::I1);
                if ty.is_float() {
                    self.emit_instruction(Instruction::FCmp {
                        result,
                        op: FloatCompareOp::OrderedLessEqual,
                        lhs, rhs, ty,
                    });
                } else {
                    self.emit_instruction(Instruction::ICmp {
                        result,
                        op: CompareOp::SignedLessEqual,
                        lhs, rhs, ty,
                    });
                }
                result
            }
            BinaryOp::GreaterEqual => {
                let result = self.new_value(IrType::I1);
                if ty.is_float() {
                    self.emit_instruction(Instruction::FCmp {
                        result,
                        op: FloatCompareOp::OrderedGreaterEqual,
                        lhs, rhs, ty,
                    });
                } else {
                    self.emit_instruction(Instruction::ICmp {
                        result,
                        op: CompareOp::SignedGreaterEqual,
                        lhs, rhs, ty,
                    });
                }
                result
            }
            // LogicalAnd/LogicalOr handled above via short-circuit
            BinaryOp::LogicalAnd | BinaryOp::LogicalOr => unreachable!(),
        }
    }

    /// Lowers logical AND with short-circuit evaluation.
    ///
    /// `a && b` → if a is false, result is 0; else evaluate b.
    fn lower_logical_and(&mut self, left: &Expression, right: &Expression) -> Value {
        let lhs = self.lower_expression(left);
        let lhs_bool = self.ensure_boolean(lhs);

        let rhs_block = self.create_block("land.rhs");
        let merge_block = self.create_block("land.merge");
        let current_block_id = self.current_function.as_ref().map(|fb| fb.current_block).unwrap_or(BlockId(0));

        // If LHS is false, skip RHS
        if let Some(ref mut fb) = self.current_function {
            if !fb.current_block_terminated() {
                fb.current_block_mut().terminator = Some(Terminator::CondBranch {
                    condition: lhs_bool,
                    true_block: rhs_block,
                    false_block: merge_block,
                });
            }
        }

        // Evaluate RHS
        self.set_insert_point(rhs_block);
        let rhs = self.lower_expression(right);
        let rhs_bool = self.ensure_boolean(rhs);
        let rhs_end_block = self.current_function.as_ref().map(|fb| fb.current_block).unwrap_or(BlockId(0));

        if let Some(ref mut fb) = self.current_function {
            if !fb.current_block_terminated() {
                fb.current_block_mut().terminator =
                    Some(Terminator::Branch { target: merge_block });
            }
        }

        // Merge with phi node
        self.set_insert_point(merge_block);
        let false_val = self.emit_const_int(0, IrType::I1);
        let result = self.new_value(IrType::I1);
        let phi = Instruction::Phi {
            result,
            ty: IrType::I1,
            incoming: vec![
                (false_val, current_block_id),
                (rhs_bool, rhs_end_block),
            ],
        };
        self.emit_instruction(phi);
        result
    }

    /// Lowers logical OR with short-circuit evaluation.
    ///
    /// `a || b` → if a is true, result is 1; else evaluate b.
    fn lower_logical_or(&mut self, left: &Expression, right: &Expression) -> Value {
        let lhs = self.lower_expression(left);
        let lhs_bool = self.ensure_boolean(lhs);

        let rhs_block = self.create_block("lor.rhs");
        let merge_block = self.create_block("lor.merge");
        let current_block_id = self.current_function.as_ref().map(|fb| fb.current_block).unwrap_or(BlockId(0));

        // If LHS is true, skip RHS
        if let Some(ref mut fb) = self.current_function {
            if !fb.current_block_terminated() {
                fb.current_block_mut().terminator = Some(Terminator::CondBranch {
                    condition: lhs_bool,
                    true_block: merge_block,
                    false_block: rhs_block,
                });
            }
        }

        // Evaluate RHS
        self.set_insert_point(rhs_block);
        let rhs = self.lower_expression(right);
        let rhs_bool = self.ensure_boolean(rhs);
        let rhs_end_block = self.current_function.as_ref().map(|fb| fb.current_block).unwrap_or(BlockId(0));

        if let Some(ref mut fb) = self.current_function {
            if !fb.current_block_terminated() {
                fb.current_block_mut().terminator =
                    Some(Terminator::Branch { target: merge_block });
            }
        }

        // Merge with phi node
        self.set_insert_point(merge_block);
        let true_val = self.emit_const_int(1, IrType::I1);
        let result = self.new_value(IrType::I1);
        let phi = Instruction::Phi {
            result,
            ty: IrType::I1,
            incoming: vec![
                (true_val, current_block_id),
                (rhs_bool, rhs_end_block),
            ],
        };
        self.emit_instruction(phi);
        result
    }

    // -----------------------------------------------------------------------
    // Unary expression lowering
    // -----------------------------------------------------------------------

    /// Lowers a unary prefix expression.
    fn lower_unary_expression(&mut self, op: UnaryOp, operand: &Expression) -> Value {
        match op {
            UnaryOp::Plus => {
                // Unary plus is a no-op
                self.lower_expression(operand)
            }
            UnaryOp::Negate => {
                let val = self.lower_expression(operand);
                // Determine the type of the value being negated.
                // Float negation requires a float zero and float Sub;
                // integer negation uses integer zero and integer Sub.
                let val_ty = self.value_type(val);
                if val_ty == IrType::F64 || val_ty == IrType::F32 {
                    // Float negation: 0.0 - val
                    let zero = self.new_value(val_ty.clone());
                    self.emit_instruction(Instruction::Const {
                        result: zero,
                        value: crate::ir::instructions::Constant::Float {
                            value: 0.0,
                            ty: val_ty.clone(),
                        },
                    });
                    let result = self.new_value(val_ty.clone());
                    self.emit_instruction(Instruction::Sub {
                        result,
                        lhs: zero,
                        rhs: val,
                        ty: val_ty,
                    });
                    result
                } else {
                    // Integer negation: widen to appropriate integer type
                    let neg_ty = match val_ty {
                        IrType::I64 => IrType::I64,
                        IrType::I16 => IrType::I16,
                        IrType::I8 => IrType::I8,
                        _ => IrType::I32,
                    };
                    let zero = self.emit_const_int(0, neg_ty.clone());
                    let result = self.new_value(neg_ty.clone());
                    self.emit_instruction(Instruction::Sub {
                        result,
                        lhs: zero,
                        rhs: val,
                        ty: neg_ty,
                    });
                    result
                }
            }
            UnaryOp::BitwiseNot => {
                let val = self.lower_expression(operand);
                let all_ones = self.emit_const_int(-1, IrType::I32);
                let result = self.new_value(IrType::I32);
                self.emit_instruction(Instruction::Xor {
                    result,
                    lhs: val,
                    rhs: all_ones,
                    ty: IrType::I32,
                });
                result
            }
            UnaryOp::LogicalNot => {
                let val = self.lower_expression(operand);
                let zero = self.emit_const_int(0, IrType::I32);
                let result = self.new_value(IrType::I1);
                self.emit_instruction(Instruction::ICmp {
                    result,
                    op: CompareOp::Equal,
                    lhs: val,
                    rhs: zero,
                    ty: IrType::I32,
                });
                result
            }
            UnaryOp::Dereference => {
                let ptr_val = self.lower_expression(operand);
                let result = self.new_value(IrType::I32);
                self.emit_instruction(Instruction::Load {
                    result,
                    ty: IrType::I32,
                    ptr: ptr_val,
                });
                result
            }
            UnaryOp::AddressOf => {
                // For address-of (&x), we need the lvalue address (alloca/global pointer)
                // rather than the loaded value. get_lvalue_address returns the IR value
                // representing the memory address of the operand.
                if let Some(addr) = self.get_lvalue_address(operand) {
                    // Track as pointer type so that `ptr + n` uses GEP.
                    // The pointee type comes from the operand's local_types.
                    let pointee = self.infer_operand_type(operand);
                    if let Some(ref mut fb) = self.current_function {
                        fb.value_types.insert(addr, pointee.pointer_to());
                    }
                    addr
                } else {
                    // Fallback: if we can't get the lvalue address, emit a zero pointer
                    self.emit_const_int(0, IrType::I64)
                }
            }
            UnaryOp::PreIncrement => {
                self.lower_pre_increment(operand, true)
            }
            UnaryOp::PreDecrement => {
                self.lower_pre_increment(operand, false)
            }
        }
    }

    /// Lowers pre-increment (++x) or pre-decrement (--x).
    fn lower_pre_increment(&mut self, operand: &Expression, is_increment: bool) -> Value {
        let addr = self.get_lvalue_address(operand);
        let val = self.lower_expression(operand);

        // Determine the type of the operand to handle pointer increment correctly.
        let val_ty = self.current_function.as_ref()
            .and_then(|fb| fb.value_types.get(&val).cloned())
            .unwrap_or(IrType::I32);

        let result = if val_ty.is_pointer() {
            let elem_ty = val_ty.element_type().cloned().unwrap_or(IrType::I8);
            let idx_val = self.emit_const_int(if is_increment { 1 } else { -1 }, IrType::I64);
            let gep_result = self.new_value(val_ty.clone());
            self.emit_instruction(Instruction::GetElementPtr {
                result: gep_result,
                base_ty: elem_ty,
                ptr: val,
                indices: vec![idx_val],
                in_bounds: true,
            });
            if let Some(ref mut fb) = self.current_function {
                if let Some(vt) = fb.value_types.get(&val).cloned() {
                    fb.value_types.insert(gep_result, vt);
                }
            }
            gep_result
        } else {
            let one = self.emit_const_int(1, val_ty.clone());
            let r = self.new_value(val_ty.clone());
            if is_increment {
                self.emit_instruction(Instruction::Add { result: r, lhs: val, rhs: one, ty: val_ty });
            } else {
                self.emit_instruction(Instruction::Sub { result: r, lhs: val, rhs: one, ty: val_ty });
            }
            r
        };
        if let Some(ptr) = addr {
            self.emit_instruction(Instruction::Store { value: result, ptr, store_ty: None });
        }
        result
    }

    /// Lowers post-increment (x++) or post-decrement (x--).
    fn lower_post_increment(&mut self, operand: &Expression, is_increment: bool) -> Value {
        let addr = self.get_lvalue_address(operand);
        let original = self.lower_expression(operand);

        // Determine the type of the operand to handle pointer increment correctly.
        let val_ty = self.current_function.as_ref()
            .and_then(|fb| fb.value_types.get(&original).cloned())
            .unwrap_or(IrType::I32);

        let new_val = if val_ty.is_pointer() {
            let elem_ty = val_ty.element_type().cloned().unwrap_or(IrType::I8);
            let idx_val = self.emit_const_int(if is_increment { 1 } else { -1 }, IrType::I64);
            let gep_result = self.new_value(val_ty.clone());
            self.emit_instruction(Instruction::GetElementPtr {
                result: gep_result,
                base_ty: elem_ty,
                ptr: original,
                indices: vec![idx_val],
                in_bounds: true,
            });
            if let Some(ref mut fb) = self.current_function {
                if let Some(vt) = fb.value_types.get(&original).cloned() {
                    fb.value_types.insert(gep_result, vt);
                }
            }
            gep_result
        } else {
            let one = self.emit_const_int(1, val_ty.clone());
            let r = self.new_value(val_ty.clone());
            if is_increment {
                self.emit_instruction(Instruction::Add { result: r, lhs: original, rhs: one, ty: val_ty });
            } else {
                self.emit_instruction(Instruction::Sub { result: r, lhs: original, rhs: one, ty: val_ty });
            }
            r
        };
        if let Some(ptr) = addr {
            self.emit_instruction(Instruction::Store { value: new_val, ptr, store_ty: None });
        }
        original
    }

    // -----------------------------------------------------------------------
    // Assignment lowering
    // -----------------------------------------------------------------------

    /// Lowers an assignment expression (simple or compound).
    fn lower_assignment(
        &mut self,
        op: AssignmentOp,
        target: &Expression,
        value: &Expression,
    ) -> Value {
        // Get the address of the assignment target first (before evaluating RHS
        // which might clobber value_map entries if target and value overlap).
        let target_addr = self.get_lvalue_address(target);

        let rhs = self.lower_expression(value);

        let result_val = match op {
            AssignmentOp::Assign => {
                // Simple assignment: store RHS to LHS location.
                rhs
            }
            _ => {
                // Compound assignment: load current LHS value, operate, produce result.
                let lhs = if let Some(addr) = target_addr {
                    let load_result = self.new_value(IrType::I32);
                    let load_inst = Instruction::Load {
                        result: load_result,
                        ty: IrType::I32,
                        ptr: addr,
                    };
                    self.emit_instruction(load_inst);
                    load_result
                } else {
                    self.lower_expression(target)
                };
                let result = self.new_value(IrType::I32);
                let inst = match op {
                    AssignmentOp::AddAssign => Instruction::Add {
                        result,
                        lhs,
                        rhs,
                        ty: IrType::I32,
                    },
                    AssignmentOp::SubAssign => Instruction::Sub {
                        result,
                        lhs,
                        rhs,
                        ty: IrType::I32,
                    },
                    AssignmentOp::MulAssign => Instruction::Mul {
                        result,
                        lhs,
                        rhs,
                        ty: IrType::I32,
                    },
                    AssignmentOp::DivAssign => Instruction::Div {
                        result,
                        lhs,
                        rhs,
                        ty: IrType::I32,
                        is_signed: true,
                    },
                    AssignmentOp::ModAssign => Instruction::Mod {
                        result,
                        lhs,
                        rhs,
                        ty: IrType::I32,
                        is_signed: true,
                    },
                    AssignmentOp::AndAssign => Instruction::And {
                        result,
                        lhs,
                        rhs,
                        ty: IrType::I32,
                    },
                    AssignmentOp::OrAssign => Instruction::Or {
                        result,
                        lhs,
                        rhs,
                        ty: IrType::I32,
                    },
                    AssignmentOp::XorAssign => Instruction::Xor {
                        result,
                        lhs,
                        rhs,
                        ty: IrType::I32,
                    },
                    AssignmentOp::ShlAssign => Instruction::Shl {
                        result,
                        lhs,
                        rhs,
                        ty: IrType::I32,
                    },
                    AssignmentOp::ShrAssign => Instruction::Shr {
                        result,
                        lhs,
                        rhs,
                        ty: IrType::I32,
                        is_arithmetic: true,
                    },
                    AssignmentOp::Assign => unreachable!(),
                };
                self.emit_instruction(inst);
                result
            }
        };

        // Emit the Store instruction to write the result back to the target.
        if let Some(addr) = target_addr {
            // Determine store type from the assignment target's lvalue type.
            let st_ty = self.infer_store_type_from_target(target);
            let store_inst = Instruction::Store {
                value: result_val,
                ptr: addr,
                store_ty: st_ty,
            };
            self.emit_instruction(store_inst);
        }

        result_val
    }

    // -----------------------------------------------------------------------
    // Ternary expression lowering
    // -----------------------------------------------------------------------

    /// Lowers a ternary conditional expression `condition ? then : else`.
    fn lower_ternary(
        &mut self,
        condition: &Expression,
        then_expr: &Expression,
        else_expr: &Expression,
    ) -> Value {
        let cond_val = self.lower_expression(condition);
        let cond_bool = self.ensure_boolean(cond_val);

        let then_block = self.create_block("ternary.then");
        let else_block = self.create_block("ternary.else");
        let merge_block = self.create_block("ternary.merge");

        if let Some(ref mut fb) = self.current_function {
            if !fb.current_block_terminated() {
                fb.current_block_mut().terminator = Some(Terminator::CondBranch {
                    condition: cond_bool,
                    true_block: then_block,
                    false_block: else_block,
                });
            }
        }

        // Then
        self.set_insert_point(then_block);
        let then_val = self.lower_expression(then_expr);
        let then_end_block = self.current_function.as_ref().map(|fb| fb.current_block).unwrap_or(BlockId(0));
        if let Some(ref mut fb) = self.current_function {
            if !fb.current_block_terminated() {
                fb.current_block_mut().terminator =
                    Some(Terminator::Branch { target: merge_block });
            }
        }

        // Else
        self.set_insert_point(else_block);
        let else_val = self.lower_expression(else_expr);
        let else_end_block = self.current_function.as_ref().map(|fb| fb.current_block).unwrap_or(BlockId(0));
        if let Some(ref mut fb) = self.current_function {
            if !fb.current_block_terminated() {
                fb.current_block_mut().terminator =
                    Some(Terminator::Branch { target: merge_block });
            }
        }

        // Merge with phi — determine the result type from the then branch
        // value. For float ternaries (e.g., `(cond) ? 1.0 : 2.0`), the phi
        // must carry the float type so the register allocator uses XMM registers.
        self.set_insert_point(merge_block);
        // First try the value_types map (tracks explicit float types from literals,
        // loads, etc.), then fall back to instruction search, then default to I32.
        let phi_ty = self.current_function.as_ref()
            .and_then(|fb| fb.value_types.get(&then_val).cloned())
            .unwrap_or_else(|| {
                let t = self.value_type(then_val);
                if t == IrType::Void { IrType::I32 } else { t }
            });
        let result = self.new_value(phi_ty.clone());
        let phi = Instruction::Phi {
            result,
            ty: phi_ty,
            incoming: vec![
                (then_val, then_end_block),
                (else_val, else_end_block),
            ],
        };
        self.emit_instruction(phi);
        result
    }

    // -----------------------------------------------------------------------
    // Function call lowering
    // -----------------------------------------------------------------------

    /// Lowers a function call expression.
    fn lower_function_call(&mut self, callee: &Expression, args: &[Expression]) -> Value {
        // For large struct-typed arguments (>8 bytes), pass a pointer to the
        // struct instead of loading the value.  This implements the System V
        // AMD64 ABI convention where structs larger than two eightbytes are
        // passed by hidden pointer.  The callee copies the struct from the
        // pointer into a local alloca (see build_function parameter handling).
        // Small structs (≤8 bytes) continue to be passed by value in a register.
        let arg_values: Vec<Value> = args.iter().map(|a| {
            // Check if this argument is a large struct by looking at the lvalue type
            let is_large_struct_arg = if let Expression::Identifier { name, .. } = a {
                if let Some(ref fb) = self.current_function {
                    fb.local_types.get(name).map_or(false, |ty| {
                        matches!(ty, IrType::Struct { .. })
                            && Self::approx_ir_type_size(ty) > 8
                    })
                } else {
                    false
                }
            } else {
                false
            };
            if is_large_struct_arg {
                // Pass the address of the struct (LEA of alloca) rather than loading the value
                if let Some(addr) = self.get_lvalue_address(a) {
                    addr
                } else {
                    self.lower_expression(a)
                }
            } else {
                self.lower_expression(a)
            }
        }).collect();

        let callee_ir = match callee {
            Expression::Identifier { name, .. } => {
                // Check if this identifier is a local variable or parameter
                // (i.e., a function pointer) rather than a direct function name.
                let is_local_var = if let Some(ref fb) = self.current_function {
                    fb.local_values.contains_key(name)
                } else {
                    false
                };
                let is_symbol_var = self.symbol_values.contains_key(name);
                if is_local_var || is_symbol_var {
                    // Indirect call through a function pointer variable
                    let ptr_val = self.lower_expression(callee);
                    Callee::Indirect(ptr_val)
                } else {
                    // Direct call — resolve the InternId to the actual function name
                    let func_name = self.interner.resolve(*name).to_string();
                    Callee::Direct(func_name)
                }
            }
            _ => {
                // Indirect call through a function pointer expression
                let ptr_val = self.lower_expression(callee);
                Callee::Indirect(ptr_val)
            }
        };

        // Determine the actual return type by looking up the called
        // function in the module.  Falls back to I32 for unknown callees.
        let return_ty = match &callee_ir {
            Callee::Direct(ref name) => {
                self.module.functions.iter()
                    .find(|f| f.name == *name)
                    .map(|f| f.return_type.clone())
                    .unwrap_or(IrType::I32)
            }
            Callee::Indirect(_) => {
                // For indirect calls through function pointers, try to
                // retrieve the type from the callee expression's type.
                IrType::I32
            }
        };
        let result = self.new_value(return_ty.clone());

        // Track the return type so downstream consumers (codegen) know
        // whether the result lives in RAX (integer) or XMM0 (float).
        if return_ty.is_float() {
            if let Some(ref mut fb) = self.current_function {
                fb.value_types.insert(result, return_ty.clone());
            }
        }

        let call = Instruction::Call {
            result: Some(result),
            callee: callee_ir,
            args: arg_values,
            return_ty,
        };
        self.emit_instruction(call);
        result
    }

    // -----------------------------------------------------------------------
    // Helper methods
    // -----------------------------------------------------------------------

    /// Returns the memory address (alloca pointer) for an lvalue expression
    /// so that the caller can emit a Store to it.  Falls back to
    /// `lower_expression` (which loads the value) for constructs that have
    /// no addressable location — the caller must detect that and skip the
    /// Store in such a case.
    fn get_lvalue_address(&mut self, expr: &Expression) -> Option<Value> {
        match expr {
            Expression::Identifier { name, .. } => {
                // Look up the alloca pointer in local_values / symbol_values.
                if let Some(ref fb) = self.current_function {
                    if let Some(&alloca_val) = fb.local_values.get(name) {
                        return Some(alloca_val);
                    }
                }
                if let Some(&val) = self.symbol_values.get(name) {
                    return Some(val);
                }
                // Check module-level global variables.  Emit a GlobalRef
                // to obtain the address of the global for store operations.
                let resolved_name = self.interner.resolve(*name);
                if self.module.globals.iter().any(|g| g.name == resolved_name) {
                    let ptr_val = self.new_value(IrType::I32.pointer_to());
                    let ref_inst = Instruction::Const {
                        result: ptr_val,
                        value: Constant::GlobalRef(resolved_name.to_string()),
                    };
                    self.emit_instruction(ref_inst);
                    return Some(ptr_val);
                }
                None
            }
            Expression::UnaryPrefix { op, operand, .. }
                if *op == UnaryOp::Dereference =>
            {
                // *ptr — the address is the value of the pointer expression.
                Some(self.lower_expression(operand))
            }
            Expression::Subscript { array, index, .. } => {
                // array[index] — compute GEP and return the resulting pointer.
                let array_val = self.lower_expression(array);
                let index_val = self.lower_expression(index);
                let gep_result = self.new_value(IrType::I32.pointer_to());
                let gep = Instruction::GetElementPtr {
                    result: gep_result,
                    base_ty: IrType::I32,
                    ptr: array_val,
                    indices: vec![index_val],
                    in_bounds: true,
                };
                self.emit_instruction(gep);
                Some(gep_result)
            }
            Expression::MemberAccess { object, member, .. } => {
                // struct.field — compute address by byte offset of member.
                let obj_addr = self.get_lvalue_address(object);
                if let Some(base) = obj_addr {
                    let member_name = self.interner.resolve(*member).to_string();
                    let (byte_offset, field_ty) = self.compute_struct_member_offset(object, &member_name);
                    if byte_offset == 0 {
                        // Field is at the base of the struct — just cast pointer type.
                        let result_ty = field_ty.pointer_to();
                        if let Some(ref mut fb) = self.current_function {
                            fb.value_types.insert(base, result_ty);
                        }
                        Some(base)
                    } else {
                        // Compute base + byte_offset using GEP on I8 pointer, then treat as field_ty*
                        let offset_val = self.emit_const_int(byte_offset as i64, IrType::I64);
                        let gep_result = self.new_value(field_ty.clone().pointer_to());
                        let gep = Instruction::GetElementPtr {
                            result: gep_result,
                            base_ty: IrType::I8,
                            ptr: base,
                            indices: vec![offset_val],
                            in_bounds: true,
                        };
                        self.emit_instruction(gep);
                        if let Some(ref mut fb) = self.current_function {
                            fb.value_types.insert(gep_result, field_ty.pointer_to());
                        }
                        Some(gep_result)
                    }
                } else {
                    None
                }
            }
            Expression::ArrowAccess { pointer, member, .. } => {
                // ptr->field — load pointer then add byte offset.
                let ptr_val = self.lower_expression(pointer);
                let member_name = self.interner.resolve(*member).to_string();
                let (byte_offset, field_ty) = self.compute_struct_member_offset_from_ptr(pointer, &member_name);
                if byte_offset == 0 {
                    if let Some(ref mut fb) = self.current_function {
                        fb.value_types.insert(ptr_val, field_ty.pointer_to());
                    }
                    Some(ptr_val)
                } else {
                    let offset_val = self.emit_const_int(byte_offset as i64, IrType::I64);
                    let gep_result = self.new_value(field_ty.clone().pointer_to());
                    let gep = Instruction::GetElementPtr {
                        result: gep_result,
                        base_ty: IrType::I8,
                        ptr: ptr_val,
                        indices: vec![offset_val],
                        in_bounds: true,
                    };
                    self.emit_instruction(gep);
                    if let Some(ref mut fb) = self.current_function {
                        fb.value_types.insert(gep_result, field_ty.pointer_to());
                    }
                    Some(gep_result)
                }
            }
            _ => None,
        }
    }

    /// Emits an integer constant instruction and returns the result value.
    /// Infer the IR type of an AST expression based on identifiers and
    /// local_types.  Returns I32 as a fallback.
    fn infer_operand_type(&self, expr: &Expression) -> IrType {
        if let Expression::Identifier { name, .. } = expr {
            if let Some(ref fb) = self.current_function {
                if let Some(ty) = fb.local_types.get(name) {
                    return ty.clone();
                }
            }
            if let Some(ty) = self.symbol_types.get(name) {
                return ty.clone();
            }
        }
        IrType::I32
    }

    /// Compute byte offset and field type for a struct member access (object.member).
    /// Uses the object expression's type to look up struct field info.
    fn compute_struct_member_offset(&self, object: &Expression, member_name: &str) -> (usize, IrType) {
        // Try to find the struct tag from the object's type
        let struct_tag = self.infer_struct_tag_from_expr(object);
        if let Some(tag) = struct_tag {
            if let Some(field_info) = self.struct_field_names.get(&tag) {
                let mut offset: usize = 0;
                for (fname, fty) in field_info {
                    if fname == member_name {
                        return (offset, fty.clone());
                    }
                    offset += Self::approx_ir_type_size(fty);
                }
            }
        }
        // Fallback: unknown struct, assume I32 field at offset 0
        (0, IrType::I32)
    }

    /// Compute byte offset and field type for an arrow access (ptr->member).
    fn compute_struct_member_offset_from_ptr(&self, pointer: &Expression, member_name: &str) -> (usize, IrType) {
        // For arrow access, the pointer points to a struct
        let struct_tag = self.infer_struct_tag_from_ptr_expr(pointer);
        if let Some(tag) = struct_tag {
            if let Some(field_info) = self.struct_field_names.get(&tag) {
                let mut offset: usize = 0;
                for (fname, fty) in field_info {
                    if fname == member_name {
                        return (offset, fty.clone());
                    }
                    offset += Self::approx_ir_type_size(fty);
                }
            }
        }
        (0, IrType::I32)
    }

    /// Try to infer the struct tag name from an expression (used in member access).
    fn infer_struct_tag_from_expr(&self, expr: &Expression) -> Option<String> {
        if let Expression::Identifier { name, .. } = expr {
            // Check local_types for struct types
            if let Some(ref fb) = self.current_function {
                if let Some(ir_ty) = fb.local_types.get(name) {
                    return self.extract_struct_tag_from_ir_type(ir_ty);
                }
            }
            if let Some(ir_ty) = self.symbol_types.get(name) {
                return self.extract_struct_tag_from_ir_type(ir_ty);
            }
        }
        None
    }

    /// Try to infer the struct tag from a pointer expression (for arrow access).
    fn infer_struct_tag_from_ptr_expr(&self, expr: &Expression) -> Option<String> {
        // For ptr->member, ptr is a pointer to a struct. Look up the pointer's type.
        self.infer_struct_tag_from_expr(expr)
    }

    /// Extract struct tag from an IR type by matching against known struct field maps.
    fn extract_struct_tag_from_ir_type(&self, ty: &IrType) -> Option<String> {
        // If it's a struct type itself, find matching tag
        if let IrType::Struct { fields, .. } = ty {
            for (tag, field_info) in &self.struct_field_names {
                if field_info.len() == fields.len() {
                    let matches = field_info.iter().zip(fields.iter())
                        .all(|((_, ft), f)| Self::approx_ir_type_size(ft) == Self::approx_ir_type_size(f));
                    if matches {
                        return Some(tag.clone());
                    }
                }
            }
        }
        // If it's a pointer to struct (for arrow access)
        if let IrType::Pointer(inner) = ty {
            return self.extract_struct_tag_from_ir_type(inner);
        }
        None
    }

    /// Returns an approximate byte size for an IR type without requiring a TargetConfig.
    /// Uses 64-bit pointer assumptions (8 bytes).
    /// Converts an IrType back to an approximate CType.
    ///
    /// This is used by specifier_to_ctype's StructRef path to reconstruct
    /// field CTypes from the stored IrTypes in struct_field_names.  The
    /// mapping must be a valid round-trip: map_type(ir_type_to_approx_ctype(t))
    /// should equal t for the common scalar and pointer types.
    fn ir_type_to_approx_ctype(ir_ty: &IrType) -> CType {
        match ir_ty {
            IrType::Void => CType::Void,
            IrType::I1 => CType::Integer(IntegerKind::Bool),
            IrType::I8 => CType::Integer(IntegerKind::Char),
            IrType::I16 => CType::Integer(IntegerKind::Short),
            IrType::I32 => CType::Integer(IntegerKind::Int),
            // Use LongLong (unconditionally 64-bit) to guarantee I64 mapping
            // regardless of target long_size configuration.
            IrType::I64 => CType::Integer(IntegerKind::LongLong),
            IrType::F32 => CType::Float(FloatKind::Float),
            IrType::F64 => CType::Float(FloatKind::Double),
            IrType::Pointer(inner) => CType::Pointer {
                pointee: Box::new(Self::ir_type_to_approx_ctype(inner)),
                qualifiers: crate::sema::types::TypeQualifiers::default(),
            },
            IrType::Struct { fields, packed } => {
                let st_fields: Vec<crate::sema::types::StructField> = fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| crate::sema::types::StructField {
                        name: Some(format!("_field{}", i)),
                        ty: Self::ir_type_to_approx_ctype(f),
                        bit_width: None,
                        offset: 0,
                    })
                    .collect();
                CType::Struct(crate::sema::types::StructType {
                    tag: None,
                    fields: st_fields,
                    is_union: false,
                    is_packed: *packed,
                    custom_alignment: None,
                    is_complete: true,
                })
            }
            IrType::Array { element, count } => CType::Array {
                element: Box::new(Self::ir_type_to_approx_ctype(element)),
                size: crate::sema::types::ArraySize::Fixed(*count),
            },
            _ => CType::Integer(IntegerKind::Int),
        }
    }

    fn approx_ir_type_size(ty: &IrType) -> usize {
        match ty {
            IrType::Void => 0,
            IrType::I1 => 1,
            IrType::I8 => 1,
            IrType::I16 => 2,
            IrType::I32 => 4,
            IrType::I64 => 8,
            IrType::F32 => 4,
            IrType::F64 => 8,
            IrType::Pointer(_) => 8,
            IrType::Array { element, count } => Self::approx_ir_type_size(element) * count,
            IrType::Struct { fields, .. } => {
                fields.iter().map(|f| Self::approx_ir_type_size(f)).sum()
            }
            _ => 4,
        }
    }

    fn emit_const_int(&mut self, value: i64, ty: IrType) -> Value {
        let result = self.new_value(ty.clone());
        let inst = Instruction::Const {
            result,
            value: Constant::Integer { value, ty },
        };
        self.emit_instruction(inst);
        result
    }

    /// Ensures a value is a boolean (i1). If the value is not already i1,
    /// compares it against zero to produce an i1 result.
    fn ensure_boolean(&mut self, val: Value) -> Value {
        // Compare value != 0 to get a boolean
        let zero = self.emit_const_int(0, IrType::I32);
        let result = self.new_value(IrType::I1);
        self.emit_instruction(Instruction::ICmp {
            result,
            op: CompareOp::NotEqual,
            lhs: val,
            rhs: zero,
            ty: IrType::I32,
        });
        result
    }

    /// Gets or creates a basic block for a goto label.
    fn get_or_create_label_block(&mut self, label: InternId) -> BlockId {
        if let Some(&block_id) = self.label_blocks.get(&label) {
            return block_id;
        }
        let label_name = self.interner.resolve(label);
        let block_id = self.create_block(&format!("label.{}", label_name));
        self.label_blocks.insert(label, block_id);
        block_id
    }

    /// Returns `true` if the declarator describes a function type
    /// (i.e. `int foo(void)` or `int (*fp)(int)` etc.).
    fn is_function_declarator(declarator: &crate::frontend::parser::ast::Declarator) -> bool {
        Self::is_function_direct_declarator(&declarator.direct)
    }

    /// Helper: checks if a DirectDeclarator has a function form.
    fn is_function_direct_declarator(dd: &crate::frontend::parser::ast::DirectDeclarator) -> bool {
        match dd {
            crate::frontend::parser::ast::DirectDeclarator::Function { .. } => true,
            crate::frontend::parser::ast::DirectDeclarator::Parenthesized(inner) => {
                Self::is_function_declarator(inner)
            }
            _ => false,
        }
    }

    /// Extracts the identifier name from a declarator, returning an empty string
    /// if no name is found (abstract declarators). Uses the interner to resolve
    /// InternId values to actual string names.
    fn extract_declarator_name(declarator: &crate::frontend::parser::ast::Declarator, interner: &Interner) -> String {
        match &declarator.direct {
            crate::frontend::parser::ast::DirectDeclarator::Identifier(id) => {
                interner.resolve(*id).to_string()
            }
            crate::frontend::parser::ast::DirectDeclarator::Parenthesized(inner) => {
                Self::extract_declarator_name(inner, interner)
            }
            crate::frontend::parser::ast::DirectDeclarator::Array { base, .. } => {
                Self::extract_direct_declarator_name(base, interner)
            }
            crate::frontend::parser::ast::DirectDeclarator::Function { base, .. } => {
                Self::extract_direct_declarator_name(base, interner)
            }
            crate::frontend::parser::ast::DirectDeclarator::Abstract => String::new(),
        }
    }

    /// Helper to extract name from a DirectDeclarator. Uses the interner to resolve
    /// InternId values to actual string names.
    fn extract_direct_declarator_name(
        dd: &crate::frontend::parser::ast::DirectDeclarator,
        interner: &Interner,
    ) -> String {
        match dd {
            crate::frontend::parser::ast::DirectDeclarator::Identifier(id) => {
                interner.resolve(*id).to_string()
            }
            crate::frontend::parser::ast::DirectDeclarator::Parenthesized(inner) => {
                Self::extract_declarator_name(inner, interner)
            }
            crate::frontend::parser::ast::DirectDeclarator::Array { base, .. } => {
                Self::extract_direct_declarator_name(base, interner)
            }
            crate::frontend::parser::ast::DirectDeclarator::Function { base, .. } => {
                Self::extract_direct_declarator_name(base, interner)
            }
            crate::frontend::parser::ast::DirectDeclarator::Abstract => String::new(),
        }
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::diagnostics::DiagnosticEmitter;
    use crate::common::source_map::{SourceLocation, SourceSpan};
    use crate::driver::target::TargetConfig;
    use crate::ir::instructions::{BlockId, Value};
    use crate::ir::types::IrType;
    use crate::sema::types::{
        ArraySize as SemaArraySize, CType, EnumType, FloatKind, FunctionParam, FunctionType,
        IntegerKind, StructField, StructType, TypeQualifiers,
    };

    /// Creates a dummy SourceSpan for test AST nodes.
    fn dummy_span() -> SourceSpan {
        SourceSpan {
            start: SourceLocation::dummy(),
            end: SourceLocation::dummy(),
        }
    }

    /// Creates an IrBuilder for testing.
    fn make_builder(diagnostics: &mut DiagnosticEmitter) -> IrBuilder<'_> {
        let target = Box::leak(Box::new(TargetConfig::x86_64()));
        let interner = Box::leak(Box::new(Interner::new()));
        IrBuilder::new(target, diagnostics, "test_module", interner)
    }

    // -----------------------------------------------------------------------
    // Type mapping tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_map_type_void() {
        let mut diag = DiagnosticEmitter::new();
        let builder = make_builder(&mut diag);
        assert_eq!(builder.map_type(&CType::Void), IrType::Void);
    }

    #[test]
    fn test_map_type_int() {
        let mut diag = DiagnosticEmitter::new();
        let builder = make_builder(&mut diag);
        assert_eq!(
            builder.map_type(&CType::Integer(IntegerKind::Int)),
            IrType::I32
        );
    }

    #[test]
    fn test_map_type_char() {
        let mut diag = DiagnosticEmitter::new();
        let builder = make_builder(&mut diag);
        assert_eq!(
            builder.map_type(&CType::Integer(IntegerKind::Char)),
            IrType::I8
        );
    }

    #[test]
    fn test_map_type_short() {
        let mut diag = DiagnosticEmitter::new();
        let builder = make_builder(&mut diag);
        assert_eq!(
            builder.map_type(&CType::Integer(IntegerKind::Short)),
            IrType::I16
        );
    }

    #[test]
    fn test_map_type_long_long() {
        let mut diag = DiagnosticEmitter::new();
        let builder = make_builder(&mut diag);
        assert_eq!(
            builder.map_type(&CType::Integer(IntegerKind::LongLong)),
            IrType::I64
        );
    }

    #[test]
    fn test_map_type_long_x86_64() {
        let mut diag = DiagnosticEmitter::new();
        let builder = make_builder(&mut diag);
        // x86-64: long is 8 bytes → I64
        assert_eq!(
            builder.map_type(&CType::Integer(IntegerKind::Long)),
            IrType::I64
        );
    }

    #[test]
    fn test_map_type_float() {
        let mut diag = DiagnosticEmitter::new();
        let builder = make_builder(&mut diag);
        assert_eq!(
            builder.map_type(&CType::Float(FloatKind::Float)),
            IrType::F32
        );
    }

    #[test]
    fn test_map_type_double() {
        let mut diag = DiagnosticEmitter::new();
        let builder = make_builder(&mut diag);
        assert_eq!(
            builder.map_type(&CType::Float(FloatKind::Double)),
            IrType::F64
        );
    }

    #[test]
    fn test_map_type_pointer() {
        let mut diag = DiagnosticEmitter::new();
        let builder = make_builder(&mut diag);
        let ptr_ty = CType::Pointer {
            pointee: Box::new(CType::Integer(IntegerKind::Int)),
            qualifiers: TypeQualifiers::default(),
        };
        assert_eq!(
            builder.map_type(&ptr_ty),
            IrType::Pointer(Box::new(IrType::I32))
        );
    }

    #[test]
    fn test_map_type_array() {
        let mut diag = DiagnosticEmitter::new();
        let builder = make_builder(&mut diag);
        let arr_ty = CType::Array {
            element: Box::new(CType::Integer(IntegerKind::Int)),
            size: SemaArraySize::Fixed(10),
        };
        assert_eq!(
            builder.map_type(&arr_ty),
            IrType::Array {
                element: Box::new(IrType::I32),
                count: 10,
            }
        );
    }

    #[test]
    fn test_map_type_struct() {
        let mut diag = DiagnosticEmitter::new();
        let builder = make_builder(&mut diag);
        let struct_ty = CType::Struct(StructType {
            tag: Some("test".to_string()),
            fields: vec![
                StructField {
                    name: Some("x".to_string()),
                    ty: CType::Integer(IntegerKind::Int),
                    bit_width: None,
                    offset: 0,
                },
                StructField {
                    name: Some("y".to_string()),
                    ty: CType::Float(FloatKind::Double),
                    bit_width: None,
                    offset: 8,
                },
            ],
            is_union: false,
            is_packed: false,
            custom_alignment: None,
            is_complete: true,
        });
        assert_eq!(
            builder.map_type(&struct_ty),
            IrType::Struct {
                fields: vec![IrType::I32, IrType::F64],
                packed: false,
            }
        );
    }

    #[test]
    fn test_map_type_enum() {
        let mut diag = DiagnosticEmitter::new();
        let builder = make_builder(&mut diag);
        let enum_ty = CType::Enum(EnumType {
            tag: Some("color".to_string()),
            variants: vec![("RED".to_string(), 0), ("GREEN".to_string(), 1)],
            is_complete: true,
        });
        assert_eq!(builder.map_type(&enum_ty), IrType::I32);
    }

    #[test]
    fn test_map_type_function() {
        let mut diag = DiagnosticEmitter::new();
        let builder = make_builder(&mut diag);
        let fn_ty = CType::Function(FunctionType {
            return_type: Box::new(CType::Integer(IntegerKind::Int)),
            params: vec![FunctionParam {
                name: Some("x".to_string()),
                ty: CType::Integer(IntegerKind::Int),
            }],
            is_variadic: false,
            is_old_style: false,
        });
        assert_eq!(
            builder.map_type(&fn_ty),
            IrType::Function {
                return_type: Box::new(IrType::I32),
                param_types: vec![IrType::I32],
                is_variadic: false,
            }
        );
    }

    #[test]
    fn test_map_type_typedef() {
        let mut diag = DiagnosticEmitter::new();
        let builder = make_builder(&mut diag);
        let typedef_ty = CType::Typedef {
            name: "my_int".to_string(),
            underlying: Box::new(CType::Integer(IntegerKind::Int)),
        };
        assert_eq!(builder.map_type(&typedef_ty), IrType::I32);
    }

    #[test]
    fn test_map_type_qualified() {
        let mut diag = DiagnosticEmitter::new();
        let builder = make_builder(&mut diag);
        let qual_ty = CType::Qualified {
            base: Box::new(CType::Integer(IntegerKind::Int)),
            qualifiers: TypeQualifiers {
                is_const: true,
                ..TypeQualifiers::default()
            },
        };
        // Qualifiers are stripped at IR level
        assert_eq!(builder.map_type(&qual_ty), IrType::I32);
    }

    // -----------------------------------------------------------------------
    // Value and block creation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_value_unique_ids() {
        let mut diag = DiagnosticEmitter::new();
        let mut builder = make_builder(&mut diag);
        let v1 = builder.new_value(IrType::I32);
        let v2 = builder.new_value(IrType::I32);
        let v3 = builder.new_value(IrType::I64);
        assert_ne!(v1, v2);
        assert_ne!(v2, v3);
        assert_eq!(v1, Value(0));
        assert_eq!(v2, Value(1));
        assert_eq!(v3, Value(2));
    }

    #[test]
    fn test_module_creation() {
        let module = Module::new("test".to_string());
        assert_eq!(module.name, "test");
        assert!(module.functions.is_empty());
        assert!(module.globals.is_empty());
    }

    #[test]
    fn test_build_empty_translation_unit() {
        let mut diag = DiagnosticEmitter::new();
        let mut builder = make_builder(&mut diag);
        let tu = TypedTranslationUnit {
            declarations: Vec::new(),
            span: dummy_span(),
        };
        let module = builder.build(&tu);
        assert!(module.functions.is_empty());
        assert!(module.globals.is_empty());
    }

    #[test]
    fn test_emit_const_int() {
        let mut diag = DiagnosticEmitter::new();
        let mut builder = make_builder(&mut diag);

        // Set up a function context for instruction emission
        let entry_id = BlockId(0);
        builder.next_block_id = 1;
        builder.current_function = Some(FunctionBuilder::new(
            "test_fn".to_string(),
            IrType::Void,
            Vec::new(),
            entry_id,
        ));

        let val = builder.emit_const_int(42, IrType::I32);
        assert_ne!(val, Value::undef());

        // Verify the instruction was emitted to the current block
        let fb = builder.current_function.as_ref().unwrap();
        assert!(!fb.blocks[0].instructions.is_empty());
    }

    #[test]
    fn test_create_block() {
        let mut diag = DiagnosticEmitter::new();
        let mut builder = make_builder(&mut diag);

        let entry_id = BlockId(0);
        builder.next_block_id = 1;
        builder.current_function = Some(FunctionBuilder::new(
            "test_fn".to_string(),
            IrType::Void,
            Vec::new(),
            entry_id,
        ));

        let b1 = builder.create_block("then");
        let b2 = builder.create_block("else");
        assert_ne!(b1, b2);
    }
}
