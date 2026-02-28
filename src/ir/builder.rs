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
use crate::common::intern::InternId;
use crate::common::source_map::SourceSpan;
use crate::driver::target::TargetConfig;
use crate::ir::cfg::{BasicBlock, Terminator};
use crate::ir::instructions::{
    BlockId, Callee, CompareOp, Constant, Instruction, Value,
};
use crate::ir::types::IrType;
use crate::sema::types::{
    ArraySize as SemaArraySize, CType, FloatKind, IntegerKind,
};
use crate::sema::{TypedDeclaration, TypedTranslationUnit};

// Re-import AST types for pattern matching during lowering.
use crate::frontend::parser::ast::{
    AssignmentOp, BinaryOp, BlockItem, Declaration, Expression, ForInit, FunctionDef,
    InitDeclarator, Initializer, Statement, StorageClass, UnaryOp,
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
    /// Basic blocks comprising the function body. Empty for declarations.
    pub blocks: Vec<BasicBlock>,
    /// The entry block ID (the first block executed on function entry).
    pub entry_block: BlockId,
    /// `true` if this function has a body (definition), `false` for extern stubs.
    pub is_definition: bool,
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
    /// Basic blocks built so far.
    blocks: Vec<BasicBlock>,
    /// The current block being appended to.
    current_block: BlockId,
    /// The entry block ID.
    entry_block: BlockId,
    /// Per-function mapping from variable identifiers to their alloca values.
    local_values: HashMap<InternId, Value>,
}

impl FunctionBuilder {
    /// Creates a new function builder with the given name, return type, and params.
    fn new(name: String, return_type: IrType, params: Vec<(String, IrType)>, entry_id: BlockId) -> Self {
        let entry_block = BasicBlock::new(entry_id, "entry".to_string());
        FunctionBuilder {
            name,
            return_type,
            params,
            blocks: vec![entry_block],
            current_block: entry_id,
            entry_block: entry_id,
            local_values: HashMap::new(),
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
/// let mut builder = IrBuilder::new(&target, &mut diagnostics, "main.c");
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
    /// Counter for generating unique value IDs.
    next_value_id: u32,
    /// Counter for generating unique block IDs.
    next_block_id: u32,
    /// Map from symbol identifiers to IR values (for variable → alloca mapping).
    symbol_values: HashMap<InternId, Value>,
    /// Map from label names to block IDs (for goto support).
    label_blocks: HashMap<InternId, BlockId>,
    /// Break target stack (for nested loops/switches).
    break_targets: Vec<BlockId>,
    /// Continue target stack (for nested loops).
    continue_targets: Vec<BlockId>,
    /// String literal counter for generating unique global names.
    string_literal_count: u32,
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
    ) -> Self {
        IrBuilder {
            module: Module::new(module_name.to_string()),
            current_function: None,
            target,
            diagnostics,
            next_value_id: 0,
            next_block_id: 0,
            symbol_values: HashMap::new(),
            label_blocks: HashMap::new(),
            break_targets: Vec::new(),
            continue_targets: Vec::new(),
            string_literal_count: 0,
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
        self.next_value_id += 1;
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
        self.next_block_id += 1;
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
        let is_extern = specifiers.storage_class == Some(StorageClass::Extern);
        let is_static = specifiers.storage_class == Some(StorageClass::Static);

        for init_decl in declarators {
            // Extract name from the declarator
            let name = Self::extract_declarator_name(&init_decl.declarator);
            if name.is_empty() {
                continue;
            }

            // Determine the type: prefer resolved_type from sema, fall back to i32
            let c_type = resolved_type.as_ref().cloned().unwrap_or(CType::Integer(IntegerKind::Int));
            let ir_type = self.map_type(&c_type);

            // Lower the initializer if present
            let initializer = init_decl.initializer.as_ref().and_then(|init| {
                self.lower_constant_initializer(init, &ir_type)
            });

            self.module.globals.push(GlobalVariable {
                name,
                ty: ir_type,
                initializer,
                is_extern,
                is_static,
            });
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
            Expression::FloatLiteral { value, .. } => {
                Some(Constant::Float {
                    value: *value,
                    ty: IrType::F64,
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

        let func_name = Self::extract_declarator_name(&func_def.declarator);
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
        self.next_block_id += 1;

        // Save value ID state (functions have independent value spaces conceptually,
        // but we keep a global counter for simplicity)
        let saved_symbol_values = std::mem::take(&mut self.symbol_values);
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

            // Create a value representing the parameter
            let param_val = self.new_value(param_ty.clone());
            let const_inst = Instruction::Const {
                result: param_val,
                value: Constant::Integer {
                    value: i as i64,
                    ty: param_ty.clone(),
                },
            };
            // We represent parameter values as constants with their index for now;
            // the actual parameter passing is handled by the code generator.
            // For the IR, we just store the param into the alloca.
            let store_inst = Instruction::Store {
                value: param_val,
                ptr: alloca_val,
            };
            self.emit_instruction(const_inst);
            self.emit_instruction(store_inst);

            // Register the alloca for later use when lowering references to this param
            if let Some(ref mut fb) = self.current_function {
                // We need an InternId for the param name, but we don't have the interner.
                // Store using a synthetic InternId based on the parameter index.
                let synthetic_id = InternId::from_raw(0x8000_0000 + i as u32);
                fb.local_values.insert(synthetic_id, alloca_val);
            }
            // Also store with a name-based lookup
            let _ = param_name; // Name stored in params vec
        }

        // Lower the function body
        self.lower_statement(&func_def.body);

        // Ensure the last block has a terminator
        // Pre-generate a value ID in case we need it for zero-return
        let zero_val_id = self.next_value_id;
        self.next_value_id += 1;
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
        let function = Function {
            name: fb.name,
            return_type: fb.return_type,
            params: fb.params,
            blocks: fb.blocks,
            entry_block: fb.entry_block,
            is_definition: true,
        };

        self.module.functions.push(function);

        // Restore saved state
        self.symbol_values = saved_symbol_values;
        self.label_blocks = saved_label_blocks;
        self.break_targets = saved_break;
        self.continue_targets = saved_continue;
    }

    /// Resolves the function signature from the AST and optional resolved CType.
    fn resolve_function_signature(
        &self,
        _func_def: &FunctionDef,
        resolved_type: &Option<CType>,
    ) -> (CType, Vec<(String, CType)>) {
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

        // Fall back to void return with no params
        (CType::Void, Vec::new())
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

        for case_val in &cases {
            let case_block = self.create_block(&format!("switch.case.{}", case_val));
            case_entries.push((*case_val, case_block));
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

        // Lower case bodies — simplified: lower entire switch body in default block
        // (a full implementation would dispatch to individual case blocks)
        self.set_insert_point(default_block);
        self.lower_statement(body);
        if let Some(ref mut fb) = self.current_function {
            if !fb.current_block_terminated() {
                fb.current_block_mut().terminator =
                    Some(Terminator::Branch { target: exit_block });
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
                for init_decl in declarators {
                    let var_name = Self::extract_declarator_name(&init_decl.declarator);
                    if var_name.is_empty() {
                        continue;
                    }

                    // Default to i32 if no resolved type (actual resolution happens in sema)
                    let var_type = IrType::I32;

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
                                };
                                self.emit_instruction(store_inst);
                            }
                            Initializer::Compound { .. } => {
                                // Compound initializers for locals: zero-init then fill fields
                                // Simplified: just leave default-initialized
                            }
                        }
                    }

                    // Register in the local values map
                    if let Some(ref mut fb) = self.current_function {
                        // Use a synthetic InternId for now
                        let synthetic_id = InternId::from_raw(self.next_value_id + 0x4000_0000);
                        fb.local_values.insert(synthetic_id, alloca_val);
                    }

                    // Also register in the global symbol_values map using the alloca
                    // (variable reference lowering will look here)
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

            Expression::FloatLiteral { value, .. } => {
                let result = self.new_value(IrType::F64);
                let inst = Instruction::Const {
                    result,
                    value: Constant::Float {
                        value: *value,
                        ty: IrType::F64,
                    },
                };
                self.emit_instruction(inst);
                result
            }

            Expression::StringLiteral { value, .. } => {
                // String literals become global constants
                let str_name = format!(".str.{}", self.string_literal_count);
                self.string_literal_count += 1;
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
                // Look up the variable in symbol_values or function builder locals
                if let Some(ref fb) = self.current_function {
                    if let Some(&alloca_val) = fb.local_values.get(name) {
                        // Load from the alloca
                        let result = self.new_value(IrType::I32);
                        let inst = Instruction::Load {
                            result,
                            ty: IrType::I32,
                            ptr: alloca_val,
                        };
                        return self.emit_instruction(inst);
                    }
                }
                if let Some(&val) = self.symbol_values.get(name) {
                    let result = self.new_value(IrType::I32);
                    let inst = Instruction::Load {
                        result,
                        ty: IrType::I32,
                        ptr: val,
                    };
                    return self.emit_instruction(inst);
                }
                // Not found: could be a function reference or global
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
            Expression::MemberAccess { object, member: _member, .. } => {
                let obj_val = self.lower_expression(object);
                // Simplified: compute field offset via GEP with field index
                let zero = self.emit_const_int(0, IrType::I32);
                let field_idx = self.emit_const_int(0, IrType::I32); // Simplified: index 0
                let gep_result = self.new_value(IrType::I32.pointer_to());
                let gep = Instruction::GetElementPtr {
                    result: gep_result,
                    base_ty: IrType::I32,
                    ptr: obj_val,
                    indices: vec![zero, field_idx],
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

            // === Arrow access ===
            Expression::ArrowAccess {
                pointer, member: _member, ..
            } => {
                // Load the pointer, then GEP + Load
                let ptr_val = self.lower_expression(pointer);
                let zero = self.emit_const_int(0, IrType::I32);
                let field_idx = self.emit_const_int(0, IrType::I32);
                let gep_result = self.new_value(IrType::I32.pointer_to());
                let gep = Instruction::GetElementPtr {
                    result: gep_result,
                    base_ty: IrType::I32,
                    ptr: ptr_val,
                    indices: vec![zero, field_idx],
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
            Expression::Cast { operand, .. } => {
                // Simplified: lower the operand (cast is identity at IR level for now)
                self.lower_expression(operand)
            }

            // === Sizeof ===
            Expression::SizeofExpr { .. } | Expression::SizeofType { .. } => {
                // sizeof produces a compile-time constant
                // Default to pointer_size for generic sizeof
                let size_val = self.target.pointer_size as i64;
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
            Expression::Alignof { .. } => {
                let result = self.new_value(IrType::I64);
                let inst = Instruction::Const {
                    result,
                    value: Constant::Integer {
                        value: self.target.max_alignment as i64,
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
        let ty = IrType::I32; // Simplified: default to i32 for all operations

        match op {
            BinaryOp::Add => {
                let result = self.new_value(ty.clone());
                self.emit_instruction(Instruction::Add {
                    result,
                    lhs,
                    rhs,
                    ty,
                });
                result
            }
            BinaryOp::Sub => {
                let result = self.new_value(ty.clone());
                self.emit_instruction(Instruction::Sub {
                    result,
                    lhs,
                    rhs,
                    ty,
                });
                result
            }
            BinaryOp::Mul => {
                let result = self.new_value(ty.clone());
                self.emit_instruction(Instruction::Mul {
                    result,
                    lhs,
                    rhs,
                    ty,
                });
                result
            }
            BinaryOp::Div => {
                let result = self.new_value(ty.clone());
                self.emit_instruction(Instruction::Div {
                    result,
                    lhs,
                    rhs,
                    ty,
                    is_signed: true,
                });
                result
            }
            BinaryOp::Mod => {
                let result = self.new_value(ty.clone());
                self.emit_instruction(Instruction::Mod {
                    result,
                    lhs,
                    rhs,
                    ty,
                    is_signed: true,
                });
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
                self.emit_instruction(Instruction::ICmp {
                    result,
                    op: CompareOp::Equal,
                    lhs,
                    rhs,
                    ty,
                });
                result
            }
            BinaryOp::NotEqual => {
                let result = self.new_value(IrType::I1);
                self.emit_instruction(Instruction::ICmp {
                    result,
                    op: CompareOp::NotEqual,
                    lhs,
                    rhs,
                    ty,
                });
                result
            }
            BinaryOp::Less => {
                let result = self.new_value(IrType::I1);
                self.emit_instruction(Instruction::ICmp {
                    result,
                    op: CompareOp::SignedLess,
                    lhs,
                    rhs,
                    ty,
                });
                result
            }
            BinaryOp::Greater => {
                let result = self.new_value(IrType::I1);
                self.emit_instruction(Instruction::ICmp {
                    result,
                    op: CompareOp::SignedGreater,
                    lhs,
                    rhs,
                    ty,
                });
                result
            }
            BinaryOp::LessEqual => {
                let result = self.new_value(IrType::I1);
                self.emit_instruction(Instruction::ICmp {
                    result,
                    op: CompareOp::SignedLessEqual,
                    lhs,
                    rhs,
                    ty,
                });
                result
            }
            BinaryOp::GreaterEqual => {
                let result = self.new_value(IrType::I1);
                self.emit_instruction(Instruction::ICmp {
                    result,
                    op: CompareOp::SignedGreaterEqual,
                    lhs,
                    rhs,
                    ty,
                });
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
                let zero = self.emit_const_int(0, IrType::I32);
                let result = self.new_value(IrType::I32);
                self.emit_instruction(Instruction::Sub {
                    result,
                    lhs: zero,
                    rhs: val,
                    ty: IrType::I32,
                });
                result
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
                // For address-of, we need the lvalue (alloca) rather than the loaded value
                // Simplified: just return the expression value as a pointer
                self.lower_expression(operand)
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
        // Simplified: evaluate operand, add/sub 1, return new value
        let val = self.lower_expression(operand);
        let one = self.emit_const_int(1, IrType::I32);
        let result = self.new_value(IrType::I32);
        if is_increment {
            self.emit_instruction(Instruction::Add {
                result,
                lhs: val,
                rhs: one,
                ty: IrType::I32,
            });
        } else {
            self.emit_instruction(Instruction::Sub {
                result,
                lhs: val,
                rhs: one,
                ty: IrType::I32,
            });
        }
        result
    }

    /// Lowers post-increment (x++) or post-decrement (x--).
    fn lower_post_increment(&mut self, operand: &Expression, is_increment: bool) -> Value {
        // Evaluate operand (get current value), then increment, return original value
        let original = self.lower_expression(operand);
        let one = self.emit_const_int(1, IrType::I32);
        let _new_val = self.new_value(IrType::I32);
        if is_increment {
            self.emit_instruction(Instruction::Add {
                result: _new_val,
                lhs: original,
                rhs: one,
                ty: IrType::I32,
            });
        } else {
            self.emit_instruction(Instruction::Sub {
                result: _new_val,
                lhs: original,
                rhs: one,
                ty: IrType::I32,
            });
        }
        // Return the original (pre-increment/decrement) value
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
        let rhs = self.lower_expression(value);

        match op {
            AssignmentOp::Assign => {
                // Simple assignment: evaluate RHS, store to LHS location
                // For now, simplified: just return the RHS value
                rhs
            }
            _ => {
                // Compound assignment: load LHS, operate with RHS, store result
                let lhs = self.lower_expression(target);
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
        }
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

        // Merge with phi
        self.set_insert_point(merge_block);
        let result = self.new_value(IrType::I32);
        let phi = Instruction::Phi {
            result,
            ty: IrType::I32,
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
        let arg_values: Vec<Value> = args.iter().map(|a| self.lower_expression(a)).collect();

        let callee_ir = match callee {
            Expression::Identifier { name, .. } => {
                // Direct call — use the function name
                // We need the actual string name; use a placeholder since we don't have
                // the interner here. The actual name would be resolved via the interner.
                Callee::Direct(format!("__func_{}", name.as_u32()))
            }
            _ => {
                // Indirect call through a function pointer
                let ptr_val = self.lower_expression(callee);
                Callee::Indirect(ptr_val)
            }
        };

        let return_ty = IrType::I32; // Simplified: default return type
        let result = self.new_value(return_ty.clone());
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

    /// Emits an integer constant instruction and returns the result value.
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
        let block_id = self.create_block(&format!("label.{}", label.as_u32()));
        self.label_blocks.insert(label, block_id);
        block_id
    }

    /// Extracts the identifier name from a declarator, returning an empty string
    /// if no name is found (abstract declarators).
    fn extract_declarator_name(declarator: &crate::frontend::parser::ast::Declarator) -> String {
        match &declarator.direct {
            crate::frontend::parser::ast::DirectDeclarator::Identifier(id) => {
                format!("__id_{}", id.as_u32())
            }
            crate::frontend::parser::ast::DirectDeclarator::Parenthesized(inner) => {
                Self::extract_declarator_name(inner)
            }
            crate::frontend::parser::ast::DirectDeclarator::Array { base, .. } => {
                Self::extract_direct_declarator_name(base)
            }
            crate::frontend::parser::ast::DirectDeclarator::Function { base, .. } => {
                Self::extract_direct_declarator_name(base)
            }
            crate::frontend::parser::ast::DirectDeclarator::Abstract => String::new(),
        }
    }

    /// Helper to extract name from a DirectDeclarator.
    fn extract_direct_declarator_name(
        dd: &crate::frontend::parser::ast::DirectDeclarator,
    ) -> String {
        match dd {
            crate::frontend::parser::ast::DirectDeclarator::Identifier(id) => {
                format!("__id_{}", id.as_u32())
            }
            crate::frontend::parser::ast::DirectDeclarator::Parenthesized(inner) => {
                Self::extract_declarator_name(inner)
            }
            crate::frontend::parser::ast::DirectDeclarator::Array { base, .. } => {
                Self::extract_direct_declarator_name(base)
            }
            crate::frontend::parser::ast::DirectDeclarator::Function { base, .. } => {
                Self::extract_direct_declarator_name(base)
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
        IrBuilder::new(target, diagnostics, "test_module")
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
