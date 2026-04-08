//! Semantic analysis module for the `bcc` C compiler.
//!
//! This is the primary entry point for semantic analysis — the third stage of the
//! compilation pipeline following preprocessing and parsing. It receives an untyped
//! AST ([`TranslationUnit`]) from the parser and produces a [`TypedTranslationUnit`]
//! with:
//!
//! - **Resolved types** — every expression has an assigned [`CType`]
//! - **Symbol table** — all identifiers bound to their declarations
//! - **Implicit conversions** — insertion points for integer promotions, arithmetic
//!   conversions, array/function decay, etc.
//! - **Validated semantics** — type errors, undeclared identifiers, scope violations,
//!   and storage class conflicts have been diagnosed
//!
//! # Architecture
//!
//! The module delegates to six submodules:
//!
//! | Submodule          | Responsibility                                       |
//! |--------------------|------------------------------------------------------|
//! | `types`            | C type representation with target-parametric sizes   |
//! | `type_check`       | Type compatibility checking for all expression kinds  |
//! | `type_conversion`  | Implicit conversion rules (promotions, decay, casts) |
//! | `scope`            | Lexical scope stack management (C11 §6.2.1)          |
//! | `symbol_table`     | Scope-aware symbol storage with three namespaces     |
//! | `storage`          | Storage class validation and linkage resolution       |
//!
//! # Four-Architecture Support
//!
//! All type size computations are parameterized by [`TargetConfig`], enabling
//! correct analysis across x86-64, i686, AArch64, and RISC-V 64 targets.
//! For example, `sizeof(long)` yields 8 on x86-64 but 4 on i686.
//!
//! # GCC-Compatible Diagnostics
//!
//! All errors and warnings are reported via [`DiagnosticEmitter`] in
//! `file:line:col: error: message` format on stderr.
//!
//! # Zero External Dependencies
//!
//! Only imports from `std` and internal crate modules. No external crates.
//!
//! # Safety
//!
//! This module contains zero `unsafe` blocks. All operations use safe Rust.

// ===========================================================================
// Submodule declarations
// ===========================================================================

pub mod scope;
pub mod storage;
pub mod symbol_table;
pub mod type_check;
pub mod type_conversion;
pub mod types;

// ===========================================================================
// Public re-exports — downstream consumers use `crate::sema::CType`, etc.
// ===========================================================================

// From types.rs — the complete C type system
pub use self::types::{
    ArraySize, CType, EnumType, FloatKind, FunctionParam, FunctionType, IntegerKind, StructField,
    StructType, TypeQualifiers,
};

// From scope.rs — lexical scope management
pub use self::scope::{Scope, ScopeKind, ScopeStack};

// From symbol_table.rs — scope-aware symbol storage
pub use self::symbol_table::{Symbol, SymbolKind, SymbolTable};

// From storage.rs — storage class validation and linkage
pub use self::storage::{Linkage, StorageDuration};

// From type_conversion.rs — implicit conversion classification
pub use self::type_conversion::{ConversionResult, ImplicitCastKind};

// ===========================================================================
// Imports for SemanticAnalyzer implementation
// ===========================================================================

use crate::common::diagnostics::DiagnosticEmitter;
use crate::common::intern::{InternId, Interner};
use crate::common::source_map::SourceSpan;
use crate::driver::target::TargetConfig;
use crate::frontend::parser::ast::{
    AssignmentOp, BlockItem, DeclSpecifiers, Declaration, Declarator, DirectDeclarator, Expression,
    ForInit, FunctionDef, InitDeclarator, Initializer, Statement, StorageClass, TranslationUnit,
    TypeName, TypeSpecifier,
};

// ===========================================================================
// TypedTranslationUnit — output of semantic analysis
// ===========================================================================

/// The output of semantic analysis: a typed translation unit containing
/// declarations annotated with resolved types, symbol references, and
/// implicit conversion information.
///
/// This struct wraps the typed declarations that the IR builder will consume.
/// Each declaration has been validated for type correctness, scope resolution,
/// and storage class compliance.
#[derive(Debug, Clone)]
pub struct TypedTranslationUnit {
    /// Top-level typed declarations (variables, functions, typedefs, etc.).
    /// Each declaration has been semantically validated and carries resolved
    /// type information.
    pub declarations: Vec<TypedDeclaration>,
    /// Source span covering the entire translation unit.
    pub span: SourceSpan,
}

/// A typed top-level declaration produced by semantic analysis.
///
/// Wraps the original AST declaration with its resolved semantic information,
/// including the C type after analysis.
#[derive(Debug, Clone)]
pub struct TypedDeclaration {
    /// The original AST declaration node.
    pub decl: Declaration,
    /// The resolved type(s) of this declaration. For variable declarations,
    /// this is the variable's type. For function definitions, this is the
    /// function type. For typedefs, this is the aliased type.
    pub resolved_type: Option<CType>,
}

// ===========================================================================
// SemanticAnalyzer — the main analysis engine
// ===========================================================================

/// The central semantic analysis engine for the `bcc` compiler.
///
/// `SemanticAnalyzer` coordinates type checking, scope management, symbol table
/// population, and implicit type conversion insertion across an entire translation
/// unit. It is parameterized by the target architecture (via [`TargetConfig`])
/// to ensure correct type sizes for multi-architecture support.
///
/// # Usage
///
/// ```ignore
/// let result = SemanticAnalyzer::analyze(&ast, &target, &interner, &mut diagnostics);
/// match result {
///     Ok(typed_tu) => { /* proceed to IR generation */ }
///     Err(()) => { /* errors were emitted via diagnostics */ }
/// }
/// ```
///
/// # Lifetime `'a`
///
/// The analyzer borrows the target config, string interner, and diagnostic
/// emitter for the duration of analysis. These references are stored in the
/// struct fields and used throughout declaration, statement, and expression
/// analysis.
pub struct SemanticAnalyzer<'a> {
    /// Symbol table for tracking declarations across scopes.
    /// Public so the IR builder can access resolved symbols after analysis.
    pub symbol_table: SymbolTable,

    /// Scope management for lexical scoping (file, function, block, prototype).
    scope_stack: ScopeStack,

    /// Target configuration for architecture-specific type sizes.
    /// Used for sizeof evaluation, type compatibility, and integer promotions.
    target: &'a TargetConfig,

    /// Diagnostic emitter for GCC-compatible error/warning reporting on stderr.
    diagnostics: &'a mut DiagnosticEmitter,

    /// String interner for resolving InternId handles to string slices
    /// (needed for diagnostic messages and symbol name display).
    interner: &'a Interner,

    /// Current function return type, set when entering a function definition.
    /// Used by `analyze_statement` to validate `return` expressions.
    /// `None` when not inside a function body.
    current_function_return_type: Option<CType>,

    /// Whether we are currently inside a loop body (for/while/do-while).
    /// Used to validate `break` and `continue` statements.
    in_loop: bool,

    /// Whether we are currently inside a switch statement body.
    /// Used to validate `break`, `case`, and `default` statements.
    in_switch: bool,

    /// Accumulated typed declarations for the output TypedTranslationUnit.
    typed_declarations: Vec<TypedDeclaration>,
}

impl<'a> SemanticAnalyzer<'a> {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Creates a new `SemanticAnalyzer` with the given target configuration,
    /// string interner, and diagnostic emitter.
    ///
    /// The analyzer starts with an empty symbol table, file-scope scope stack,
    /// and no function context.
    pub fn new(
        target: &'a TargetConfig,
        interner: &'a Interner,
        diagnostics: &'a mut DiagnosticEmitter,
    ) -> Self {
        SemanticAnalyzer {
            symbol_table: SymbolTable::new(),
            scope_stack: ScopeStack::new(),
            target,
            diagnostics,
            interner,
            current_function_return_type: None,
            in_loop: false,
            in_switch: false,
            typed_declarations: Vec::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Built-in type registration
    // -----------------------------------------------------------------------

    /// Registers compiler built-in types in the symbol table.
    ///
    /// Called once during analyzer initialization, before processing any user
    /// declarations. This makes built-in types like `__builtin_va_list` available
    /// as typedef names throughout the translation unit.
    fn register_builtin_types(&mut self) {
        // __builtin_va_list — required by <stdarg.h>.
        // Represented as a pointer to void for type-checking purposes.
        // The actual ABI-specific layout (struct on x86-64/AArch64, char* on
        // i686/RISC-V) is handled by the code generation backends.
        if let Some(va_list_id) = self.interner.get("__builtin_va_list") {
            let va_list_type = CType::Pointer {
                pointee: Box::new(CType::Void),
                qualifiers: TypeQualifiers::default(),
            };
            let sym = Symbol {
                name: va_list_id,
                ty: va_list_type,
                storage_class: None,
                linkage: Linkage::None,
                is_defined: true,
                is_tentative: false,
                kind: SymbolKind::Typedef,
                location: SourceSpan::dummy(),
                scope_depth: 0,
            };
            let _ = self.symbol_table.insert(va_list_id, sym, self.diagnostics);
        }
    }

    // -----------------------------------------------------------------------
    // Public entry point: analyze()
    // -----------------------------------------------------------------------

    /// Performs semantic analysis on a complete translation unit.
    ///
    /// This is the primary public API entry point. It creates a `SemanticAnalyzer`,
    /// enters file scope, iterates over all top-level declarations, performs type
    /// checking, scope resolution, and storage class validation, then returns
    /// either a [`TypedTranslationUnit`] on success or `Err(())` if any errors
    /// were emitted via the diagnostic emitter.
    ///
    /// # Arguments
    ///
    /// * `ast` — The untyped AST from the parser.
    /// * `target` — Target architecture configuration for type sizes.
    /// * `interner` — String interner for identifier resolution.
    /// * `diagnostics` — Mutable reference to the diagnostic emitter.
    ///
    /// # Returns
    ///
    /// `Ok(TypedTranslationUnit)` if analysis completes without errors, or
    /// `Err(())` if errors were emitted (errors are accumulated in `diagnostics`).
    pub fn analyze(
        ast: &TranslationUnit,
        target: &'a TargetConfig,
        interner: &'a Interner,
        diagnostics: &'a mut DiagnosticEmitter,
    ) -> Result<TypedTranslationUnit, ()> {
        let mut analyzer = SemanticAnalyzer::new(target, interner, diagnostics);

        // Register built-in types in the symbol table before processing any
        // user declarations. __builtin_va_list is required by <stdarg.h> and
        // is represented as a pointer to void (the actual ABI-specific layout
        // is handled by the code generation backends).
        analyzer.register_builtin_types();

        // File scope is already established by SymbolTable::new() and ScopeStack::new().
        // Iterate over all top-level declarations.
        for decl in &ast.declarations {
            analyzer.analyze_declaration(decl);
        }

        // Check for any errors that were accumulated during analysis.
        if analyzer.diagnostics.has_errors() {
            return Err(());
        }

        Ok(TypedTranslationUnit {
            declarations: analyzer.typed_declarations,
            span: ast.span,
        })
    }

    // -----------------------------------------------------------------------
    // Declaration analysis
    // -----------------------------------------------------------------------

    /// Analyzes a declaration that appears inside a function body (local scope).
    ///
    /// Performs the same type checking and symbol table registration as
    /// `analyze_declaration`, but does NOT add the declaration to the
    /// `typed_declarations` list (which feeds into the IR builder's top-level
    /// declaration processing). Local declarations are handled by the IR
    /// builder's `lower_local_declaration` when it processes the function body.
    pub fn analyze_local_declaration(&mut self, decl: &Declaration) {
        match decl {
            Declaration::Variable {
                specifiers,
                declarators,
                span,
            } => {
                self.analyze_variable_declaration(specifiers, declarators, *span);
                // Do NOT push to typed_declarations — local vars are not top-level.
            }
            Declaration::Typedef {
                specifiers,
                declarators,
                span,
            } => {
                self.analyze_typedef(specifiers, declarators, *span);
                // Typedefs don't produce IR, but we still don't push them at local scope.
            }
            Declaration::StaticAssert { expr, .. } => {
                let _ = self.analyze_expression(expr);
                // Static asserts don't produce IR.
            }
            Declaration::Function(_) => {
                // Nested function definitions are not valid in C11 at local scope,
                // but GCC allows nested functions as an extension. We'll just
                // analyze it but not add to top-level declarations.
                // For now, skip — nested functions are out of scope.
            }
            Declaration::Empty { .. } => {}
            Declaration::TopLevelAsm { .. } => {
                // Top-level asm is a passthrough — no semantic analysis needed.
            }
        }
    }

    /// Analyzes a single declaration, dispatching to the appropriate handler
    /// based on the declaration kind.
    ///
    /// Handles variable declarations, function definitions, typedefs,
    /// static assertions, and empty declarations.
    /// NOTE: This method adds declarations to `typed_declarations` for top-level
    /// (file scope) processing by the IR builder. For declarations inside function
    /// bodies, use `analyze_local_declaration` instead.
    pub fn analyze_declaration(&mut self, decl: &Declaration) {
        match decl {
            Declaration::Variable {
                specifiers,
                declarators,
                span,
            } => {
                self.analyze_variable_declaration(specifiers, declarators, *span);
                self.typed_declarations.push(TypedDeclaration {
                    decl: decl.clone(),
                    resolved_type: None,
                });
            }

            Declaration::Function(func_def) => {
                self.analyze_function_definition(func_def);
                self.typed_declarations.push(TypedDeclaration {
                    decl: decl.clone(),
                    resolved_type: None,
                });
            }

            Declaration::Typedef {
                specifiers,
                declarators,
                span,
            } => {
                self.analyze_typedef(specifiers, declarators, *span);
                self.typed_declarations.push(TypedDeclaration {
                    decl: decl.clone(),
                    resolved_type: None,
                });
            }

            Declaration::StaticAssert {
                expr,
                message,
                span,
            } => {
                self.analyze_static_assert(expr, message, *span);
                self.typed_declarations.push(TypedDeclaration {
                    decl: decl.clone(),
                    resolved_type: None,
                });
            }

            Declaration::Empty { .. } => {
                // Empty declarations are syntactically valid and require no analysis.
                self.typed_declarations.push(TypedDeclaration {
                    decl: decl.clone(),
                    resolved_type: None,
                });
            }
            Declaration::TopLevelAsm { .. } => {
                // Top-level asm declarations are passed through to the IR
                // builder without semantic analysis (they contain opaque
                // assembly text).
                self.typed_declarations.push(TypedDeclaration {
                    decl: decl.clone(),
                    resolved_type: None,
                });
            }
        }
    }

    /// Analyzes a variable declaration with specifiers and declarators.
    ///
    /// For each declarator in the declaration list:
    /// 1. Resolves the base type from the type specifier
    /// 2. Applies pointer/array/function modifiers from the declarator
    /// 3. Validates storage class specifiers for the current scope
    /// 4. Checks initializer type compatibility (if present)
    /// 5. Registers the symbol in the symbol table
    fn analyze_variable_declaration(
        &mut self,
        specifiers: &DeclSpecifiers,
        declarators: &[InitDeclarator],
        span: SourceSpan,
    ) {
        let base_type = self.resolve_type_specifier(&specifiers.type_specifier);
        let storage_class = specifiers.storage_class;

        // Validate storage class for this scope context.
        let scope_kind = self.scope_stack.current().kind;
        let linkage_result = storage::validate_storage_class(
            storage_class,
            scope_kind,
            false, // not a function declaration
            self.diagnostics,
            span,
        );

        let (linkage, _storage_duration) = match linkage_result {
            Ok(result) => result,
            Err(()) => (Linkage::None, StorageDuration::Automatic),
        };

        // Check for conflicting storage class specifiers.
        if let Some(sc) = storage_class {
            storage::check_conflicting_specifiers(&[sc], self.diagnostics, span);
        }

        for init_decl in declarators {
            let decl_type = self.apply_declarator_to_type(&base_type, &init_decl.declarator);
            let name = self.extract_declarator_name(&init_decl.declarator);

            if let Some(name_id) = name {
                // Determine if this is a definition (has initializer or is not extern).
                // Function declarations without a body (handled here, not in
                // analyze_function_definition) are NEVER definitions — they are
                // merely forward declarations. Only variable declarations at
                // file scope without `extern` are tentative definitions.
                let is_function_type = matches!(decl_type, CType::Function(_));
                let is_defined = if is_function_type {
                    // Function declarations in this path never have a body,
                    // so they are never definitions.
                    false
                } else {
                    init_decl.initializer.is_some()
                        || (storage_class != Some(StorageClass::Extern)
                            && self.scope_stack.is_file_scope())
                };
                let is_tentative = !is_defined
                    && !is_function_type
                    && storage_class != Some(StorageClass::Extern)
                    && self.scope_stack.is_file_scope();

                let symbol = Symbol {
                    name: name_id,
                    ty: decl_type.clone(),
                    storage_class,
                    linkage,
                    is_defined,
                    is_tentative,
                    kind: if is_function_type {
                        SymbolKind::Function
                    } else {
                        SymbolKind::Variable
                    },
                    location: init_decl.span,
                    scope_depth: self.scope_stack.depth(),
                };

                let _ = self.symbol_table.insert(name_id, symbol, self.diagnostics);

                // Validate initializer type compatibility.
                if let Some(ref init) = init_decl.initializer {
                    self.analyze_initializer(init, &decl_type, init_decl.span);
                }
            }
        }
    }

    /// Analyzes a function definition: validates the return type, parameters,
    /// and body.
    ///
    /// The analysis proceeds as follows:
    /// 1. Resolve the return type from specifiers
    /// 2. Build the function type from parameters
    /// 3. Validate storage class for functions
    /// 4. Register the function symbol in the current scope
    /// 5. Enter function scope and block scope
    /// 6. Register parameters as local variables
    /// 7. Analyze the function body
    /// 8. Exit block scope and function scope
    fn analyze_function_definition(&mut self, func_def: &FunctionDef) {
        let return_type = self.resolve_type_specifier(&func_def.specifiers.type_specifier);
        let storage_class = func_def.specifiers.storage_class;
        let span = func_def.span;

        // Validate storage class for function declarations.
        let scope_kind = self.scope_stack.current().kind;
        let linkage_result = storage::validate_storage_class(
            storage_class,
            scope_kind,
            true, // is a function declaration
            self.diagnostics,
            span,
        );

        let (linkage, _storage_duration) = match linkage_result {
            Ok(result) => result,
            Err(()) => (Linkage::External, StorageDuration::Static),
        };

        // Build the function type from the declarator.
        let (func_name, func_type) = self.build_function_type(&func_def.declarator, &return_type);

        if let Some(name_id) = func_name {
            // Register function in the symbol table.
            let symbol = Symbol {
                name: name_id,
                ty: CType::Function(func_type.clone()),
                storage_class,
                linkage,
                is_defined: true,
                is_tentative: false,
                kind: SymbolKind::Function,
                location: span,
                scope_depth: self.scope_stack.depth(),
            };

            let _ = self.symbol_table.insert(name_id, symbol, self.diagnostics);
        }

        // Save the previous function context and enter the new one.
        let prev_return_type = self.current_function_return_type.take();
        let prev_in_loop = self.in_loop;
        let prev_in_switch = self.in_switch;
        // Use the actual return type from the function type (which includes
        // pointer declarator modifiers), not just the base specifier type.
        // For `int *foo()`, func_type.return_type is Pointer(Int), not just Int.
        self.current_function_return_type = Some(*func_type.return_type.clone());
        self.in_loop = false;
        self.in_switch = false;

        // Enter function scope for labels.
        self.scope_stack.push(ScopeKind::Function);
        self.symbol_table.push_scope(ScopeKind::Function);

        // Enter block scope for the function body.
        self.scope_stack.push(ScopeKind::Block);
        self.symbol_table.push_scope(ScopeKind::Block);

        // Register parameters as local variables.
        for param in &func_type.params {
            if let Some(ref name) = param.name {
                let param_name_id = self.interner.get(name).unwrap_or_else(|| {
                    // If the parameter name isn't interned yet, we use a dummy.
                    // In practice, the parser should have already interned it.
                    InternId::from_raw(0)
                });
                if param_name_id.as_u32() != 0 || self.interner.resolve(param_name_id) == name {
                    let param_symbol = Symbol {
                        name: param_name_id,
                        ty: param.ty.clone(),
                        storage_class: None,
                        linkage: Linkage::None,
                        is_defined: true,
                        is_tentative: false,
                        kind: SymbolKind::Variable,
                        location: span,
                        scope_depth: self.scope_stack.depth(),
                    };
                    let _ = self
                        .symbol_table
                        .insert(param_name_id, param_symbol, self.diagnostics);
                }
            }
        }

        // Analyze the function body.
        self.analyze_statement(&func_def.body);

        // Clear labels (check for undefined label references).
        self.symbol_table.clear_labels();

        // Exit block scope and function scope.
        self.symbol_table.pop_scope();
        self.scope_stack.pop();
        self.symbol_table.pop_scope();
        self.scope_stack.pop();

        // Restore previous function context.
        self.current_function_return_type = prev_return_type;
        self.in_loop = prev_in_loop;
        self.in_switch = prev_in_switch;
    }

    /// Analyzes a typedef declaration.
    ///
    /// Resolves the base type and registers each typedef name in the symbol table.
    fn analyze_typedef(
        &mut self,
        specifiers: &DeclSpecifiers,
        declarators: &[Declarator],
        span: SourceSpan,
    ) {
        let base_type = self.resolve_type_specifier(&specifiers.type_specifier);

        for declarator in declarators {
            let typedef_type = self.apply_declarator_to_type(&base_type, declarator);
            let name = self.extract_declarator_name(declarator);

            if let Some(name_id) = name {
                let name_str = self.interner.resolve(name_id).to_string();
                let aliased = CType::Typedef {
                    name: name_str,
                    underlying: Box::new(typedef_type),
                };

                let symbol = Symbol {
                    name: name_id,
                    ty: aliased,
                    storage_class: Some(StorageClass::Static), // typedefs have no linkage
                    linkage: Linkage::None,
                    is_defined: true,
                    is_tentative: false,
                    kind: SymbolKind::Typedef,
                    location: span,
                    scope_depth: self.scope_stack.depth(),
                };

                let _ = self.symbol_table.insert(name_id, symbol, self.diagnostics);
            }
        }
    }

    /// Analyzes a `_Static_assert` declaration.
    ///
    /// Evaluates the constant expression. If it evaluates to zero, emits an
    /// error with the provided message string.
    fn analyze_static_assert(&mut self, expr: &Expression, message: &str, span: SourceSpan) {
        let expr_type = self.analyze_expression(expr);
        if !expr_type.is_integer() && !expr_type.is_error() {
            self.diagnostics.error(
                span.start,
                "static assertion expression is not an integer constant expression",
            );
        }
        // Full compile-time evaluation would determine the value.
        // For now, we only verify the type is valid — runtime assertion
        // checking is deferred to constant expression evaluation in the
        // IR builder. The message is preserved for the error diagnostic.
        let _ = message;
    }

    // -----------------------------------------------------------------------
    // Statement analysis
    // -----------------------------------------------------------------------

    /// Analyzes a single statement, dispatching to the appropriate handler
    /// based on the statement kind.
    ///
    /// Validates control flow constraints (break/continue only in loops/switches),
    /// type constraints (conditions must be scalar), and scope management
    /// (compound statements push/pop block scope).
    pub fn analyze_statement(&mut self, stmt: &Statement) {
        match stmt {
            Statement::Compound { items, .. } => {
                self.scope_stack.push(ScopeKind::Block);
                self.symbol_table.push_scope(ScopeKind::Block);

                for item in items {
                    match item {
                        BlockItem::Declaration(decl) => {
                            self.analyze_local_declaration(decl);
                        }
                        BlockItem::Statement(sub_stmt) => {
                            self.analyze_statement(sub_stmt);
                        }
                    }
                }

                self.symbol_table.pop_scope();
                self.scope_stack.pop();
            }

            Statement::If {
                condition,
                then_branch,
                else_branch,
                span,
            } => {
                let cond_type = self.analyze_expression(condition);
                if !cond_type.is_scalar() && !cond_type.is_error() {
                    self.diagnostics.error(
                        span.start,
                        format!("controlling expression type '{}' is not scalar", cond_type),
                    );
                }
                self.analyze_statement(then_branch);
                if let Some(ref else_body) = else_branch {
                    self.analyze_statement(else_body);
                }
            }

            Statement::For {
                init,
                condition,
                increment,
                body,
                span,
            } => {
                // For-loop has its own scope for C99 declarations in the init clause.
                self.scope_stack.push(ScopeKind::Block);
                self.symbol_table.push_scope(ScopeKind::Block);

                if let Some(ref for_init) = init {
                    match for_init.as_ref() {
                        ForInit::Declaration(decl) => {
                            self.analyze_local_declaration(decl);
                        }
                        ForInit::Expression(expr) => {
                            self.analyze_expression(expr);
                        }
                    }
                }

                if let Some(ref cond) = condition {
                    let cond_type = self.analyze_expression(cond);
                    if !cond_type.is_scalar() && !cond_type.is_error() {
                        self.diagnostics.error(
                            span.start,
                            format!("controlling expression type '{}' is not scalar", cond_type),
                        );
                    }
                }

                if let Some(ref incr) = increment {
                    self.analyze_expression(incr);
                }

                let prev_in_loop = self.in_loop;
                self.in_loop = true;
                self.analyze_statement(body);
                self.in_loop = prev_in_loop;

                self.symbol_table.pop_scope();
                self.scope_stack.pop();
            }

            Statement::While {
                condition,
                body,
                span,
            } => {
                let cond_type = self.analyze_expression(condition);
                if !cond_type.is_scalar() && !cond_type.is_error() {
                    self.diagnostics.error(
                        span.start,
                        format!("controlling expression type '{}' is not scalar", cond_type),
                    );
                }

                let prev_in_loop = self.in_loop;
                self.in_loop = true;
                self.analyze_statement(body);
                self.in_loop = prev_in_loop;
            }

            Statement::DoWhile {
                body,
                condition,
                span,
            } => {
                let prev_in_loop = self.in_loop;
                self.in_loop = true;
                self.analyze_statement(body);
                self.in_loop = prev_in_loop;

                let cond_type = self.analyze_expression(condition);
                if !cond_type.is_scalar() && !cond_type.is_error() {
                    self.diagnostics.error(
                        span.start,
                        format!("controlling expression type '{}' is not scalar", cond_type),
                    );
                }
            }

            Statement::Switch { expr, body, span } => {
                let expr_type = self.analyze_expression(expr);
                // C11 §6.8.4.2: The controlling expression of a switch
                // statement shall have integer type. Enum types are integer
                // types in C (underlying type is int).
                if !expr_type.is_integer() && !expr_type.is_enum() && !expr_type.is_error() {
                    self.diagnostics.error(
                        span.start,
                        format!(
                            "switch controlling expression type '{}' is not an integer",
                            expr_type
                        ),
                    );
                }

                let prev_in_switch = self.in_switch;
                self.in_switch = true;
                self.analyze_statement(body);
                self.in_switch = prev_in_switch;
            }

            Statement::Case {
                value,
                range_end,
                body,
                span,
            } => {
                if !self.in_switch {
                    self.diagnostics
                        .error(span.start, "'case' label not within a switch statement");
                }
                let val_type = self.analyze_expression(value);
                if !val_type.is_integer() && !val_type.is_error() {
                    self.diagnostics.error(
                        span.start,
                        "case expression is not an integer constant expression",
                    );
                }
                if let Some(ref range_expr) = range_end {
                    let range_type = self.analyze_expression(range_expr);
                    if !range_type.is_integer() && !range_type.is_error() {
                        self.diagnostics.error(
                            span.start,
                            "case range expression is not an integer constant expression",
                        );
                    }
                }
                self.analyze_statement(body);
            }

            Statement::Default { body, span } => {
                if !self.in_switch {
                    self.diagnostics
                        .error(span.start, "'default' label not within a switch statement");
                }
                self.analyze_statement(body);
            }

            Statement::Break { span } => {
                if !self.in_loop && !self.in_switch {
                    self.diagnostics.error(
                        span.start,
                        "'break' statement not in loop or switch statement",
                    );
                }
            }

            Statement::Continue { span } => {
                if !self.in_loop {
                    self.diagnostics
                        .error(span.start, "'continue' statement not in loop statement");
                }
            }

            Statement::Return { value, span } => {
                let return_type = value.as_ref().map(|expr| self.analyze_expression(expr));

                if let Some(ref expected) = self.current_function_return_type {
                    type_check::check_return(&return_type, expected, self.diagnostics, *span);
                } else {
                    self.diagnostics
                        .error(span.start, "'return' statement outside of function body");
                }
            }

            Statement::Goto { label, span } => {
                if !self.scope_stack.is_in_function() {
                    self.diagnostics
                        .error(span.start, "'goto' statement outside of function body");
                }
                // Register label as a forward reference (not defined).
                let label_symbol = Symbol {
                    name: *label,
                    ty: CType::Void,
                    storage_class: None,
                    linkage: Linkage::None,
                    is_defined: false,
                    is_tentative: false,
                    kind: SymbolKind::Label,
                    location: *span,
                    scope_depth: self.scope_stack.depth(),
                };
                let _ = self
                    .symbol_table
                    .insert_label(*label, label_symbol, self.diagnostics);
            }

            Statement::Labeled {
                label, body, span, ..
            } => {
                // Register label as defined.
                let label_symbol = Symbol {
                    name: *label,
                    ty: CType::Void,
                    storage_class: None,
                    linkage: Linkage::None,
                    is_defined: true,
                    is_tentative: false,
                    kind: SymbolKind::Label,
                    location: *span,
                    scope_depth: self.scope_stack.depth(),
                };
                let _ = self
                    .symbol_table
                    .insert_label(*label, label_symbol, self.diagnostics);

                self.analyze_statement(body);
            }

            Statement::Expression { expr, .. } => {
                self.analyze_expression(expr);
            }

            Statement::Null { .. } => {
                // Null statement — no analysis needed.
            }

            Statement::Declaration(decl) => {
                self.analyze_local_declaration(decl);
            }

            Statement::Asm(_) => {
                // Inline assembly — type checking is minimal; accept as-is.
                // The assembler validates the constraints during code generation.
            }

            Statement::ComputedGoto { target, span } => {
                let target_type = self.analyze_expression(target);
                if !target_type.is_pointer() && !target_type.is_error() {
                    self.diagnostics
                        .error(span.start, "argument to computed goto must be a pointer");
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Expression analysis
    // -----------------------------------------------------------------------

    /// Analyzes an expression and returns its resolved C type.
    ///
    /// Dispatches to type checking functions for operator validation,
    /// type conversion functions for implicit conversions, and the symbol
    /// table for identifier resolution.
    pub fn analyze_expression(&mut self, expr: &Expression) -> CType {
        match expr {
            // --- Literals ---
            Expression::IntegerLiteral { suffix, .. } => self.resolve_integer_literal_type(suffix),

            Expression::FloatLiteral { suffix, .. } => {
                use crate::frontend::parser::ast::FloatSuffix;
                match suffix {
                    FloatSuffix::None => CType::Float(FloatKind::Double),
                    FloatSuffix::Float => CType::Float(FloatKind::Float),
                    FloatSuffix::Long => CType::Float(FloatKind::LongDouble),
                }
            }

            Expression::StringLiteral { .. } => {
                // String literals have type `char[N]` which decays to `char *`.
                CType::Pointer {
                    pointee: Box::new(CType::Integer(IntegerKind::Char)),
                    qualifiers: TypeQualifiers {
                        is_const: true,
                        ..TypeQualifiers::default()
                    },
                }
            }

            Expression::CharLiteral { .. } => {
                // Character literals have type `int` in C (C11 §6.4.4.4).
                CType::Integer(IntegerKind::Int)
            }

            // --- Identifier ---
            Expression::Identifier { name, span } => {
                if let Some(sym) = self.symbol_table.lookup(*name) {
                    sym.ty.clone()
                } else {
                    let name_str = self.interner.resolve(*name);
                    // GCC builtins: implicitly declare __builtin_* functions
                    // with a generic signature returning int and accepting
                    // variadic arguments.  This enables kernel code that
                    // calls builtins like __builtin_memcmp, __builtin_memcpy
                    // etc. without explicit declarations.
                    if name_str.starts_with("__builtin_")
                        || name_str.starts_with("__sync_")
                        || name_str.starts_with("__atomic_")
                        || Self::is_implicit_libc_function(name_str)
                    {
                        self.diagnostics.warning(
                            span.start,
                            format!("implicit declaration of builtin function '{}'", name_str),
                        );
                        // Return a function type so that call expressions
                        // type-check.  The exact return type varies per
                        // builtin but `int` is a safe default; the backend
                        // lowers these to the appropriate IR.
                        let builtin_ret = Self::builtin_return_type(name_str);
                        CType::Function(crate::sema::types::FunctionType {
                            return_type: Box::new(builtin_ret),
                            params: Vec::new(),
                            is_variadic: true,
                            is_old_style: false,
                        })
                    } else {
                        self.diagnostics.error(
                            span.start,
                            format!("use of undeclared identifier '{}'", name_str),
                        );
                        CType::Error
                    }
                }
            }

            // --- Binary operations ---
            Expression::Binary {
                op,
                left,
                right,
                span,
            } => {
                let left_type = self.analyze_expression(left);
                let right_type = self.analyze_expression(right);
                type_check::check_binary_op(
                    *op,
                    &left_type,
                    &right_type,
                    self.target,
                    self.diagnostics,
                    *span,
                )
            }

            // --- Unary prefix operations ---
            Expression::UnaryPrefix { op, operand, span } => {
                let operand_type = self.analyze_expression(operand);
                type_check::check_unary_op(*op, &operand_type, self.target, self.diagnostics, *span)
            }

            // --- Postfix increment/decrement ---
            Expression::PostIncrement { operand, span } => {
                let operand_type = self.analyze_expression(operand);
                type_check::check_postfix_op(&operand_type, self.diagnostics, *span);
                operand_type
            }

            Expression::PostDecrement { operand, span } => {
                let operand_type = self.analyze_expression(operand);
                type_check::check_postfix_op(&operand_type, self.diagnostics, *span);
                operand_type
            }

            // --- Function call ---
            Expression::Call { callee, args, span } => {
                let callee_type = self.analyze_expression(callee);
                let arg_types: Vec<CType> = args
                    .iter()
                    .map(|arg| self.analyze_expression(arg))
                    .collect();
                type_check::check_function_call(
                    &callee_type,
                    &arg_types,
                    self.target,
                    self.diagnostics,
                    *span,
                )
            }

            // --- Array subscript ---
            Expression::Subscript { array, index, span } => {
                let array_type = self.analyze_expression(array);
                let index_type = self.analyze_expression(index);
                type_check::check_subscript(&array_type, &index_type, self.diagnostics, *span)
            }

            // --- Member access ---
            Expression::MemberAccess {
                object,
                member,
                span,
            } => {
                let object_type = self.analyze_expression(object);
                // Resolve incomplete struct types to their complete definitions.
                let resolved_type = self.resolve_complete_type(&object_type);
                let member_str = self.interner.resolve(*member);
                type_check::check_member_access(
                    &resolved_type,
                    *member,
                    member_str,
                    false, // direct access (not arrow)
                    self.diagnostics,
                    *span,
                )
            }

            Expression::ArrowAccess {
                pointer,
                member,
                span,
            } => {
                let pointer_type = self.analyze_expression(pointer);
                // Resolve incomplete struct types to their complete definitions.
                // This is critical for self-referential structs like:
                //   struct Node { int val; struct Node *next; };
                // where `next` was typed as pointer-to-incomplete-Node at parse
                // time, but Node is now complete.
                let resolved_type = self.resolve_complete_type(&pointer_type);
                let member_str = self.interner.resolve(*member);
                type_check::check_member_access(
                    &resolved_type,
                    *member,
                    member_str,
                    true, // arrow access (dereference first)
                    self.diagnostics,
                    *span,
                )
            }

            // --- Assignment ---
            Expression::Assignment {
                op,
                target,
                value,
                span,
            } => {
                let target_type = self.analyze_expression(target);
                let value_type = self.analyze_expression(value);

                // Validate lvalue.
                type_check::check_lvalue(target, &self.symbol_table, self.diagnostics, *span);

                match op {
                    AssignmentOp::Assign => {
                        type_check::check_assignment(
                            &target_type,
                            &value_type,
                            self.diagnostics,
                            *span,
                        );
                    }
                    _ => {
                        // Compound assignment (+=, -=, etc.) — the target type
                        // must be compatible with the binary operation result.
                        // The result type is the target type after the operation.
                    }
                }

                target_type
            }

            // --- Ternary conditional ---
            Expression::Ternary {
                condition,
                then_expr,
                else_expr,
                span,
            } => {
                let cond_type = self.analyze_expression(condition);
                if !cond_type.is_scalar() && !cond_type.is_error() {
                    self.diagnostics.error(
                        span.start,
                        format!("controlling expression type '{}' is not scalar", cond_type),
                    );
                }

                let then_type = self.analyze_expression(then_expr);
                let else_type = self.analyze_expression(else_expr);

                type_check::check_conditional(
                    &cond_type,
                    &then_type,
                    &else_type,
                    self.target,
                    self.diagnostics,
                    *span,
                )
            }

            // --- Comma ---
            Expression::Comma { exprs, .. } => {
                let mut last_type = CType::Void;
                for sub_expr in exprs {
                    last_type = self.analyze_expression(sub_expr);
                }
                last_type
            }

            // --- Cast ---
            Expression::Cast {
                type_name,
                operand,
                span,
            } => {
                let target_type = self.resolve_type_name(type_name);
                let operand_type = self.analyze_expression(operand);
                type_check::check_cast(&target_type, &operand_type, self.diagnostics, *span);
                target_type
            }

            // --- Sizeof ---
            Expression::SizeofExpr { expr, .. } => {
                // sizeof yields size_t, which is unsigned long on 64-bit
                // and unsigned int on 32-bit.
                let _expr_type = self.analyze_expression(expr);
                self.size_t_type()
            }

            Expression::SizeofType { type_name, .. } => {
                let _resolved = self.resolve_type_name(type_name);
                self.size_t_type()
            }

            // --- Alignof ---
            Expression::Alignof { type_name, .. } => {
                let _resolved = self.resolve_type_name(type_name);
                self.size_t_type()
            }

            // --- C11 _Generic ---
            Expression::Generic {
                controlling,
                associations,
                span,
            } => {
                use crate::frontend::parser::ast::GenericAssociation;
                let ctrl_type = self.analyze_expression(controlling);
                // Evaluate associations — find the matching type or default.
                let mut result_type = CType::Error;
                let mut found_match = false;
                for assoc in associations {
                    match assoc {
                        GenericAssociation::Type {
                            type_name, expr, ..
                        } => {
                            let assoc_type = self.resolve_type_name(type_name);
                            if ctrl_type.is_compatible(&assoc_type) && !found_match {
                                result_type = self.analyze_expression(expr);
                                found_match = true;
                            }
                        }
                        GenericAssociation::Default { expr, .. } => {
                            // default association
                            if !found_match {
                                result_type = self.analyze_expression(expr);
                                found_match = true;
                            }
                        }
                    }
                }
                if !found_match {
                    self.diagnostics.error(
                        span.start,
                        "no matching generic association for controlling expression",
                    );
                }
                result_type
            }

            // --- Compound literal ---
            Expression::CompoundLiteral {
                type_name,
                initializer,
                span,
            } => {
                let resolved = self.resolve_type_name(type_name);
                self.analyze_initializer(initializer, &resolved, *span);
                resolved
            }

            // --- GCC Extensions ---
            Expression::StatementExpr { body, .. } => {
                // GCC statement expression: `({ stmts; expr; })`.
                // The type is the type of the last expression statement in the
                // compound body.  If the body is empty or ends with a
                // non-expression item, the result type is void.
                //
                // We must capture the type DURING analysis (while variables
                // declared inside the body are still in scope), not after.
                if let Statement::Compound { items, .. } = body.as_ref() {
                    self.scope_stack.push(ScopeKind::Block);
                    self.symbol_table.push_scope(ScopeKind::Block);
                    let mut last_expr_type = CType::Void;
                    for item in items.iter() {
                        match item {
                            BlockItem::Declaration(decl) => {
                                self.analyze_local_declaration(decl);
                                last_expr_type = CType::Void;
                            }
                            BlockItem::Statement(Statement::Expression { expr, .. }) => {
                                last_expr_type = self.analyze_expression(expr);
                            }
                            BlockItem::Statement(stmt) => {
                                self.analyze_statement(stmt);
                                last_expr_type = CType::Void;
                            }
                        }
                    }
                    self.symbol_table.pop_scope();
                    self.scope_stack.pop();
                    last_expr_type
                } else {
                    self.analyze_statement(body);
                    CType::Void
                }
            }

            Expression::LabelAddr { label, span } => {
                // GCC &&label yields void *.
                if self.symbol_table.lookup_label(*label).is_none() {
                    // The label might be forward-referenced — register it.
                    let label_sym = Symbol {
                        name: *label,
                        ty: CType::Void,
                        storage_class: None,
                        linkage: Linkage::None,
                        is_defined: false,
                        is_tentative: false,
                        kind: SymbolKind::Label,
                        location: *span,
                        scope_depth: self.scope_stack.depth(),
                    };
                    let _ = self
                        .symbol_table
                        .insert_label(*label, label_sym, self.diagnostics);
                }
                CType::Pointer {
                    pointee: Box::new(CType::Void),
                    qualifiers: TypeQualifiers::default(),
                }
            }

            Expression::Extension { expr, .. } => {
                // __extension__ just suppresses warnings; analyze the inner expr.
                self.analyze_expression(expr)
            }

            Expression::BuiltinVaArg { ap, type_name, .. } => {
                let _ap_type = self.analyze_expression(ap);
                self.resolve_type_name(type_name)
            }

            Expression::BuiltinOffsetof { type_name, .. } => {
                let _resolved = self.resolve_type_name(type_name);
                self.size_t_type()
            }

            Expression::BuiltinVaStart { ap, param, .. } => {
                let _ap_type = self.analyze_expression(ap);
                let _param_type = self.analyze_expression(param);
                CType::Void
            }

            Expression::BuiltinVaEnd { ap, .. } => {
                let _ap_type = self.analyze_expression(ap);
                CType::Void
            }

            Expression::BuiltinVaCopy { dest, src, .. } => {
                let _dest_type = self.analyze_expression(dest);
                let _src_type = self.analyze_expression(src);
                CType::Void
            }

            // --- Parenthesized expression ---
            Expression::Paren { inner, .. } => self.analyze_expression(inner),

            // --- Error recovery ---
            Expression::Error { .. } => CType::Error,
        }
    }

    // -----------------------------------------------------------------------
    // Type resolution helpers
    // -----------------------------------------------------------------------

    /// Resolves a [`TypeSpecifier`] from the AST into a [`CType`].
    ///
    /// Maps parser-level type specifiers to the semantic analysis type system,
    /// handling basic types, signed/unsigned modifiers, composite types, typedef
    /// references, and GCC extensions.
    /// Resolve an incomplete struct/union type to its complete definition if one
    /// exists in the current scope. When a `struct Node *next;` is declared
    /// inside the struct body, the pointee is captured as an incomplete
    /// `CType::Struct(StructType { is_complete: false, .. })`. By the time
    /// we perform member access (`a.next->value`), the struct definition has
    /// been completed, so we look up the tag in the symbol table and return
    /// the complete type.
    fn resolve_complete_type(&self, ty: &CType) -> CType {
        match ty.canonical() {
            CType::Struct(st) if !st.is_complete => {
                if let Some(ref tag) = st.tag {
                    // Use `get` (non-mutating lookup) instead of `intern`.
                    if let Some(tag_intern) = self.interner.get(tag) {
                        if let Some(sym) = self.symbol_table.lookup_tag(tag_intern) {
                            if let CType::Struct(complete_st) = sym.ty.canonical() {
                                if complete_st.is_complete {
                                    return CType::Struct(complete_st.clone());
                                }
                            }
                        }
                    }
                }
                ty.clone()
            }
            CType::Pointer {
                pointee,
                qualifiers,
            } => {
                let resolved_pointee = self.resolve_complete_type(pointee);
                CType::Pointer {
                    pointee: Box::new(resolved_pointee),
                    qualifiers: *qualifiers,
                }
            }
            _ => ty.clone(),
        }
    }

    /// Returns `true` if the given name is a standard C library function
    /// that should be implicitly declared when encountered without a prior
    /// declaration (common in freestanding kernel code where builtins
    /// expand to bare library names).
    fn is_implicit_libc_function(name: &str) -> bool {
        matches!(
            name,
            "memcpy"
                | "memmove"
                | "memset"
                | "memcmp"
                | "memchr"
                | "strlen"
                | "strcmp"
                | "strncmp"
                | "strcpy"
                | "strncpy"
                | "strcat"
                | "strncat"
                | "strchr"
                | "strrchr"
                | "strstr"
                | "strtol"
                | "strtoul"
                | "strtoll"
                | "strtoull"
                | "printf"
                | "fprintf"
                | "sprintf"
                | "snprintf"
                | "vprintf"
                | "vfprintf"
                | "vsprintf"
                | "vsnprintf"
                | "puts"
                | "putchar"
                | "getchar"
                | "malloc"
                | "calloc"
                | "realloc"
                | "free"
                | "abort"
                | "exit"
                | "_exit"
                | "abs"
                | "labs"
                | "llabs"
        )
    }

    /// Returns the appropriate return type for a known GCC builtin
    /// function.  Builtins that return pointers (memcpy, memset, etc.)
    /// get `void *`; builtins that return sizes get `unsigned long`;
    /// everything else defaults to `int`.
    fn builtin_return_type(name: &str) -> CType {
        match name {
            // Memory builtins returning void*
            "__builtin_memcpy"
            | "__builtin_memmove"
            | "__builtin_memset"
            | "__builtin_memcpy_inline"
            | "__builtin_alloca"
            | "__builtin_return_address"
            | "__builtin_frame_address"
            | "__builtin___memcpy_chk"
            | "__builtin___memmove_chk"
            | "__builtin___memset_chk" => CType::Pointer {
                pointee: Box::new(CType::Void),
                qualifiers: TypeQualifiers::default(),
            },
            // Size/length builtins returning unsigned long
            "__builtin_strlen"
            | "__builtin_object_size"
            | "__builtin_offsetof"
            | "__builtin_sizeof" => CType::Integer(IntegerKind::UnsignedLong),
            // Boolean-like builtins returning int
            "__builtin_memcmp"
            | "__builtin_strcmp"
            | "__builtin_strncmp"
            | "__builtin_constant_p"
            | "__builtin_expect"
            | "__builtin_expect_with_probability"
            | "__builtin_types_compatible_p"
            | "__builtin_clz"
            | "__builtin_clzl"
            | "__builtin_clzll"
            | "__builtin_ctz"
            | "__builtin_ctzl"
            | "__builtin_ctzll"
            | "__builtin_ffs"
            | "__builtin_ffsl"
            | "__builtin_ffsll"
            | "__builtin_popcount"
            | "__builtin_popcountl"
            | "__builtin_popcountll"
            | "__builtin_parity"
            | "__builtin_parityl"
            | "__builtin_parityll"
            | "__builtin_add_overflow"
            | "__builtin_sub_overflow"
            | "__builtin_mul_overflow"
            | "__builtin_unreachable"
            | "__builtin_trap"
            | "__builtin_prefetch"
            | "__builtin_ia32_pause"
            | "__builtin_bswap16"
            | "__builtin_bswap32"
            | "__builtin_bswap64"
            | "__builtin_assume_aligned"
            | "__builtin_choose_expr"
            | "__builtin_va_start"
            | "__builtin_va_end"
            | "__builtin_va_copy"
            | "__builtin_va_arg" => CType::Integer(IntegerKind::Int),
            // Default: int
            _ => CType::Integer(IntegerKind::Int),
        }
    }

    fn resolve_type_specifier(&mut self, spec: &TypeSpecifier) -> CType {
        match spec {
            TypeSpecifier::Void => CType::Void,
            TypeSpecifier::Char => CType::Integer(IntegerKind::Char),
            TypeSpecifier::Short => CType::Integer(IntegerKind::Short),
            TypeSpecifier::Int => CType::Integer(IntegerKind::Int),
            TypeSpecifier::Long => CType::Integer(IntegerKind::Long),
            TypeSpecifier::LongLong => CType::Integer(IntegerKind::LongLong),
            TypeSpecifier::Float => CType::Float(FloatKind::Float),
            TypeSpecifier::Double => CType::Float(FloatKind::Double),
            TypeSpecifier::LongDouble => CType::Float(FloatKind::LongDouble),
            TypeSpecifier::Bool => CType::Integer(IntegerKind::Bool),

            TypeSpecifier::Signed(inner) => {
                match inner.as_ref() {
                    TypeSpecifier::Char => CType::Integer(IntegerKind::SignedChar),
                    TypeSpecifier::Short => CType::Integer(IntegerKind::Short),
                    TypeSpecifier::Int => CType::Integer(IntegerKind::Int),
                    TypeSpecifier::Long => CType::Integer(IntegerKind::Long),
                    TypeSpecifier::LongLong => CType::Integer(IntegerKind::LongLong),
                    _ => {
                        // `signed` alone is `signed int`.
                        CType::Integer(IntegerKind::Int)
                    }
                }
            }

            TypeSpecifier::Unsigned(inner) => {
                match inner.as_ref() {
                    TypeSpecifier::Char => CType::Integer(IntegerKind::UnsignedChar),
                    TypeSpecifier::Short => CType::Integer(IntegerKind::UnsignedShort),
                    TypeSpecifier::Int => CType::Integer(IntegerKind::UnsignedInt),
                    TypeSpecifier::Long => CType::Integer(IntegerKind::UnsignedLong),
                    TypeSpecifier::LongLong => CType::Integer(IntegerKind::UnsignedLongLong),
                    _ => {
                        // `unsigned` alone is `unsigned int`.
                        CType::Integer(IntegerKind::UnsignedInt)
                    }
                }
            }

            TypeSpecifier::Complex(_inner) => {
                // _Complex types — simplified as double for now.
                CType::Float(FloatKind::Double)
            }

            TypeSpecifier::Atomic(inner) => {
                let base = self.resolve_type_specifier(inner);
                CType::Qualified {
                    base: Box::new(base),
                    qualifiers: TypeQualifiers {
                        is_atomic: true,
                        ..TypeQualifiers::default()
                    },
                }
            }

            TypeSpecifier::Struct(struct_def) => self.analyze_struct_def(struct_def, false),

            TypeSpecifier::Union(union_def) => self.analyze_union_def(union_def),

            TypeSpecifier::Enum(enum_def) => self.analyze_enum_def(enum_def),

            TypeSpecifier::StructRef { tag, span } => {
                if let Some(sym) = self.symbol_table.lookup_tag(*tag) {
                    sym.ty.clone()
                } else {
                    // Forward reference — create incomplete struct.
                    let name = self.interner.resolve(*tag).to_string();
                    let st = StructType {
                        tag: Some(name),
                        fields: Vec::new(),
                        is_union: false,
                        is_packed: false,
                        custom_alignment: None,
                        is_complete: false,
                    };
                    let ty = CType::Struct(st);
                    let tag_sym = Symbol {
                        name: *tag,
                        ty: ty.clone(),
                        storage_class: None,
                        linkage: Linkage::None,
                        is_defined: false,
                        is_tentative: false,
                        kind: SymbolKind::StructTag,
                        location: *span,
                        scope_depth: self.scope_stack.depth(),
                    };
                    let _ = self
                        .symbol_table
                        .insert_tag(*tag, tag_sym, self.diagnostics);
                    ty
                }
            }

            TypeSpecifier::UnionRef { tag, span } => {
                if let Some(sym) = self.symbol_table.lookup_tag(*tag) {
                    sym.ty.clone()
                } else {
                    let name = self.interner.resolve(*tag).to_string();
                    let st = StructType {
                        tag: Some(name),
                        fields: Vec::new(),
                        is_union: true,
                        is_packed: false,
                        custom_alignment: None,
                        is_complete: false,
                    };
                    let ty = CType::Struct(st);
                    let tag_sym = Symbol {
                        name: *tag,
                        ty: ty.clone(),
                        storage_class: None,
                        linkage: Linkage::None,
                        is_defined: false,
                        is_tentative: false,
                        kind: SymbolKind::UnionTag,
                        location: *span,
                        scope_depth: self.scope_stack.depth(),
                    };
                    let _ = self
                        .symbol_table
                        .insert_tag(*tag, tag_sym, self.diagnostics);
                    ty
                }
            }

            TypeSpecifier::EnumRef { tag, span } => {
                if let Some(sym) = self.symbol_table.lookup_tag(*tag) {
                    sym.ty.clone()
                } else {
                    let name = self.interner.resolve(*tag).to_string();
                    let et = EnumType {
                        tag: Some(name),
                        variants: Vec::new(),
                        is_complete: false,
                    };
                    let ty = CType::Enum(et);
                    let tag_sym = Symbol {
                        name: *tag,
                        ty: ty.clone(),
                        storage_class: None,
                        linkage: Linkage::None,
                        is_defined: false,
                        is_tentative: false,
                        kind: SymbolKind::EnumTag,
                        location: *span,
                        scope_depth: self.scope_stack.depth(),
                    };
                    let _ = self
                        .symbol_table
                        .insert_tag(*tag, tag_sym, self.diagnostics);
                    ty
                }
            }

            TypeSpecifier::TypedefName { name, span } => {
                if let Some(sym) = self.symbol_table.lookup(*name) {
                    sym.ty.clone()
                } else {
                    let name_str = self.interner.resolve(*name);
                    self.diagnostics
                        .error(span.start, format!("unknown type name '{}'", name_str));
                    CType::Error
                }
            }

            TypeSpecifier::Typeof { expr, .. } => {
                let ty = self.analyze_expression(expr);
                CType::TypeOf(Box::new(ty))
            }

            TypeSpecifier::TypeofType { type_name, .. } => {
                // Resolve base type from the type specifier
                let mut ty = self.resolve_type_specifier(&type_name.specifiers.type_specifier);

                // Apply abstract declarator modifiers (pointers, arrays)
                // so that typeof(int *) correctly produces Pointer(Int)
                if let Some(ref ad) = type_name.abstract_declarator {
                    for ptr in &ad.pointer {
                        let quals = self.resolve_qualifiers(&ptr.qualifiers);
                        ty = CType::Pointer {
                            pointee: Box::new(ty),
                            qualifiers: quals,
                        };
                    }
                    // Note: array modifiers in abstract declarator of typeof
                    // are rare but could be handled similarly if needed
                }

                CType::TypeOf(Box::new(ty))
            }

            TypeSpecifier::Qualified { qualifiers, inner } => {
                let base = self.resolve_type_specifier(inner);
                let quals = self.resolve_qualifiers(qualifiers);
                if quals.is_empty() {
                    base
                } else {
                    CType::Qualified {
                        base: Box::new(base),
                        qualifiers: quals,
                    }
                }
            }

            TypeSpecifier::Error => CType::Error,
        }
    }

    /// Resolves a list of AST type qualifiers into a [`TypeQualifiers`] struct.
    /// Attempts to evaluate an expression as a compile-time constant integer.
    ///
    /// Used for array size evaluation in declarations like `int arr[1024]`.
    /// Supports integer literals and simple binary/unary operations on constants.
    fn try_eval_array_size(expr: &crate::frontend::parser::ast::Expression) -> usize {
        use crate::frontend::parser::ast::{BinaryOp, Expression, UnaryOp};
        match expr {
            Expression::IntegerLiteral { value, .. } => *value as usize,
            Expression::UnaryPrefix { op, operand, .. } => {
                let inner = Self::try_eval_array_size(operand);
                match op {
                    UnaryOp::Negate => {
                        // Negative array size doesn't make sense, but handle gracefully
                        (-(inner as i64)) as usize
                    }
                    UnaryOp::Plus => inner,
                    _ => 0,
                }
            }
            Expression::Binary {
                op, left, right, ..
            } => {
                let l = Self::try_eval_array_size(left);
                let r = Self::try_eval_array_size(right);
                match op {
                    BinaryOp::Add => l.wrapping_add(r),
                    BinaryOp::Sub => l.wrapping_sub(r),
                    BinaryOp::Mul => l.wrapping_mul(r),
                    BinaryOp::Div if r != 0 => l / r,
                    BinaryOp::Mod if r != 0 => l % r,
                    BinaryOp::ShiftLeft => l << (r & 63),
                    BinaryOp::ShiftRight => l >> (r & 63),
                    _ => 0,
                }
            }
            Expression::Cast { operand, .. } => Self::try_eval_array_size(operand),
            _ => 0,
        }
    }

    fn resolve_qualifiers(
        &self,
        qualifiers: &[crate::frontend::parser::ast::TypeQualifier],
    ) -> TypeQualifiers {
        use crate::frontend::parser::ast::TypeQualifier;
        let mut result = TypeQualifiers::default();
        for q in qualifiers {
            match q {
                TypeQualifier::Const => result.is_const = true,
                TypeQualifier::Volatile => result.is_volatile = true,
                TypeQualifier::Restrict => result.is_restrict = true,
                TypeQualifier::Atomic => result.is_atomic = true,
            }
        }
        result
    }

    /// Resolves a [`TypeName`] (used in casts, sizeof, etc.) into a [`CType`].
    fn resolve_type_name(&mut self, type_name: &TypeName) -> CType {
        let base = self.resolve_type_specifier(&type_name.specifiers.type_specifier);
        // Apply qualifiers from specifiers.
        let quals = self.resolve_qualifiers(&type_name.specifiers.type_qualifiers);
        let qualified = if quals.is_empty() {
            base
        } else {
            CType::Qualified {
                base: Box::new(base),
                qualifiers: quals,
            }
        };
        // Apply abstract declarator if present.
        if let Some(ref abs_decl) = type_name.abstract_declarator {
            self.apply_abstract_declarator(qualified, abs_decl)
        } else {
            qualified
        }
    }

    /// Applies an abstract declarator's pointer/array/function modifiers to a base type.
    ///
    /// C declarators are read "inside-out": the innermost declarator modifies
    /// the base type first, and outer modifiers wrap the result. For example,
    /// `void(*)(int)` = pointer to function(int) returning void:
    ///   1. Function modifier applied to base `void` → function(int) → void
    ///   2. Parenthesized pointer wraps the function → pointer to function
    fn apply_abstract_declarator(
        &mut self,
        base: CType,
        abs: &crate::frontend::parser::ast::AbstractDeclarator,
    ) -> CType {
        use crate::frontend::parser::ast::DirectAbstractDeclarator;

        // First, apply the direct abstract declarator (function/array suffixes).
        let result = if let Some(ref direct) = abs.direct {
            self.apply_direct_abstract_declarator(base, direct)
        } else {
            base
        };

        // Then apply pointer modifiers (outermost).
        let mut result = result;
        for ptr in &abs.pointer {
            let quals = self.resolve_qualifiers(&ptr.qualifiers);
            result = CType::Pointer {
                pointee: Box::new(result),
                qualifiers: quals,
            };
        }
        result
    }

    /// Applies a direct abstract declarator to a base type (inside-out reading).
    fn apply_direct_abstract_declarator(
        &mut self,
        base: CType,
        direct: &crate::frontend::parser::ast::DirectAbstractDeclarator,
    ) -> CType {
        use crate::frontend::parser::ast::{ArraySize, DirectAbstractDeclarator};

        match direct {
            DirectAbstractDeclarator::Parenthesized(inner_abs) => {
                // Parenthesized grouping: apply the inner abstract declarator.
                self.apply_abstract_declarator(base, inner_abs)
            }

            DirectAbstractDeclarator::Function {
                base: inner_base,
                params,
            } => {
                // Function suffix: create a function type with `base` as the
                // return type and `params` as the parameter types.
                let param_types: Vec<crate::sema::types::FunctionParam> = params
                    .params
                    .iter()
                    .map(|p| {
                        let ptype = self.resolve_type_specifier(&p.specifiers.type_specifier);
                        let quals = self.resolve_qualifiers(&p.specifiers.type_qualifiers);
                        let mut ptype = if quals.is_empty() {
                            ptype
                        } else {
                            CType::Qualified {
                                base: Box::new(ptype),
                                qualifiers: quals,
                            }
                        };
                        // Apply the parameter's declarator (pointer, array, etc.)
                        if let Some(ref decl) = p.declarator {
                            ptype = self.apply_declarator_to_type(&ptype, decl);
                        }
                        crate::sema::types::FunctionParam {
                            name: None,
                            ty: ptype,
                        }
                    })
                    .collect();

                let func_type = CType::Function(crate::sema::types::FunctionType {
                    return_type: Box::new(base),
                    params: param_types,
                    is_variadic: params.variadic,
                    is_old_style: false,
                });

                // If there's an inner base (e.g., `(*)(int)` has base = Parenthesized(*)),
                // apply the function type to the inner base.
                if let Some(ref inner) = inner_base {
                    self.apply_direct_abstract_declarator(func_type, inner)
                } else {
                    func_type
                }
            }

            DirectAbstractDeclarator::Array {
                base: inner_base,
                size,
                ..
            } => {
                // Array suffix: create an array type.
                let array_size = match size {
                    ArraySize::Fixed(expr) => {
                        let val = Self::try_eval_array_size(expr);
                        if val > 0 {
                            crate::sema::types::ArraySize::Fixed(val)
                        } else {
                            crate::sema::types::ArraySize::Incomplete
                        }
                    }
                    ArraySize::Unspecified => crate::sema::types::ArraySize::Incomplete,
                    ArraySize::VLA => crate::sema::types::ArraySize::Variable,
                    ArraySize::Static(_) => crate::sema::types::ArraySize::Incomplete,
                };

                let array_type = CType::Array {
                    element: Box::new(base),
                    size: array_size,
                };

                if let Some(ref inner) = inner_base {
                    self.apply_direct_abstract_declarator(array_type, inner)
                } else {
                    array_type
                }
            }
        }
    }

    /// Applies a declarator's pointer/array/function modifiers to a base type.
    fn apply_declarator_to_type(&mut self, base: &CType, declarator: &Declarator) -> CType {
        let mut result = base.clone();

        // Apply pointer modifiers.
        for ptr in &declarator.pointer {
            let quals = self.resolve_qualifiers(&ptr.qualifiers);
            result = CType::Pointer {
                pointee: Box::new(result),
                qualifiers: quals,
            };
        }

        // Apply direct declarator modifiers (array, function).
        result = self.apply_direct_declarator(&result, &declarator.direct);

        result
    }

    /// Applies direct declarator modifiers (array sizes, function parameters).
    fn apply_direct_declarator(&mut self, base: &CType, direct: &DirectDeclarator) -> CType {
        match direct {
            DirectDeclarator::Identifier(_) => base.clone(),
            DirectDeclarator::Abstract => base.clone(),

            DirectDeclarator::Parenthesized(inner) => self.apply_declarator_to_type(base, inner),

            DirectDeclarator::Array {
                base: inner_base,
                size,
                ..
            } => {
                let inner = self.apply_direct_declarator(base, inner_base);
                let array_size = match size {
                    crate::frontend::parser::ast::ArraySize::Fixed(expr) => {
                        // Evaluate constant expression for array size.
                        let _ty = self.analyze_expression(expr);
                        let size_val = Self::try_eval_array_size(expr);
                        types::ArraySize::Fixed(size_val)
                    }
                    crate::frontend::parser::ast::ArraySize::Unspecified => {
                        types::ArraySize::Incomplete
                    }
                    crate::frontend::parser::ast::ArraySize::VLA => types::ArraySize::Variable,
                    crate::frontend::parser::ast::ArraySize::Static(expr) => {
                        let _ty = self.analyze_expression(expr);
                        let size_val = Self::try_eval_array_size(expr);
                        types::ArraySize::Fixed(size_val)
                    }
                };
                CType::Array {
                    element: Box::new(inner),
                    size: array_size,
                }
            }

            DirectDeclarator::Function {
                base: inner_base,
                params,
            } => {
                // C declarators use an "inside-out" rule. For function pointer
                // declarators like `(*fp)(int, int)`, the parenthesized pointer
                // `*` wraps the FUNCTION type, not the return type.
                //
                // When the base is Parenthesized(Decl { pointer: [*], direct: Ident(fp) }),
                // we must:
                //  1. Compute the return type from the inner direct declarator only
                //     (skipping pointer modifiers inside the parens)
                //  2. Build the function type with that return type
                //  3. Apply the inner pointer modifiers AROUND the function type
                //
                // For `int (*fp)(int, int)`:
                //   return_type = int   (from base specifier + Ident(fp))
                //   func_type   = Function(int, [int, int])
                //   result      = Pointer { pointee: Function(int, [int, int]) }

                // Determine return type and any deferred pointer wrapping.
                let (return_type, deferred_pointers) = match inner_base.as_ref() {
                    DirectDeclarator::Parenthesized(inner_decl)
                        if !inner_decl.pointer.is_empty() =>
                    {
                        // Recurse only on the direct part (skipping pointer modifiers).
                        let rt = self.apply_direct_declarator(base, &inner_decl.direct);
                        (rt, inner_decl.pointer.as_slice())
                    }
                    _ => {
                        let rt = self.apply_direct_declarator(base, inner_base);
                        (rt, [].as_slice())
                    }
                };

                let mut func_params = Vec::new();
                for param in &params.params {
                    let param_type = self.resolve_type_specifier(&param.specifiers.type_specifier);
                    let adjusted = if let Some(ref decl) = param.declarator {
                        self.apply_declarator_to_type(&param_type, decl)
                    } else {
                        param_type
                    };
                    // Apply parameter type adjustment: arrays decay to pointers,
                    // functions decay to function pointers.
                    let adjusted = adjusted.decay();
                    let name = param
                        .declarator
                        .as_ref()
                        .and_then(|d| self.extract_declarator_name(d))
                        .map(|id| self.interner.resolve(id).to_string());
                    func_params.push(FunctionParam { name, ty: adjusted });
                }
                let mut result = CType::Function(FunctionType {
                    return_type: Box::new(return_type),
                    params: func_params,
                    is_variadic: params.variadic,
                    is_old_style: false,
                });

                // Apply deferred pointer modifiers from the parenthesized
                // declarator around the function type (inside-out rule).
                for ptr in deferred_pointers {
                    let quals = self.resolve_qualifiers(&ptr.qualifiers);
                    result = CType::Pointer {
                        pointee: Box::new(result),
                        qualifiers: quals,
                    };
                }

                result
            }
        }
    }

    /// Extracts the identifier name from a declarator, if present.
    fn extract_declarator_name(&self, declarator: &Declarator) -> Option<InternId> {
        self.extract_direct_declarator_name(&declarator.direct)
    }

    /// Extracts the identifier name from a direct declarator.
    fn extract_direct_declarator_name(&self, direct: &DirectDeclarator) -> Option<InternId> {
        match direct {
            DirectDeclarator::Identifier(id) => Some(*id),
            DirectDeclarator::Parenthesized(inner) => self.extract_declarator_name(inner),
            DirectDeclarator::Array { base, .. } => self.extract_direct_declarator_name(base),
            DirectDeclarator::Function { base, .. } => self.extract_direct_declarator_name(base),
            DirectDeclarator::Abstract => None,
        }
    }

    /// Builds a function type from a declarator and return type.
    /// Returns the function name (if any) and the FunctionType.
    fn build_function_type(
        &mut self,
        declarator: &Declarator,
        return_type: &CType,
    ) -> (Option<InternId>, FunctionType) {
        let full_type = self.apply_declarator_to_type(return_type, declarator);
        let name = self.extract_declarator_name(declarator);

        match full_type {
            CType::Function(ft) => (name, ft),
            _ => {
                // Fallback: if the declarator didn't produce a function type,
                // create a minimal function type with the resolved return type.
                (
                    name,
                    FunctionType {
                        return_type: Box::new(return_type.clone()),
                        params: Vec::new(),
                        is_variadic: false,
                        is_old_style: true,
                    },
                )
            }
        }
    }

    /// Analyzes an initializer for type compatibility with the target type.
    fn analyze_initializer(&mut self, init: &Initializer, target_type: &CType, span: SourceSpan) {
        match init {
            Initializer::Expression(expr) => {
                let init_type = self.analyze_expression(expr);
                // Use check_initialization instead of check_assignment
                // because initializing a const variable is valid in C.
                type_check::check_initialization(target_type, &init_type, self.diagnostics, span);
            }
            Initializer::Compound { items, .. } => {
                // Compound initializers: validate each item against the
                // corresponding member/element type.
                // For arrays: use the element type for each initializer item.
                // For structs/unions: use the corresponding field type.
                // For scalar types: the first element initializes the scalar.
                match target_type {
                    CType::Array { element, .. } => {
                        for item in items {
                            self.analyze_initializer(&item.initializer, element, span);
                        }
                    }
                    CType::Struct(st) if !st.is_union => {
                        for (i, item) in items.iter().enumerate() {
                            if i < st.fields.len() {
                                self.analyze_initializer(&item.initializer, &st.fields[i].ty, span);
                            }
                        }
                    }
                    CType::Struct(st) if st.is_union => {
                        // Union initialization: only the first member.
                        if let Some(item) = items.first() {
                            if let Some(field) = st.fields.first() {
                                self.analyze_initializer(&item.initializer, &field.ty, span);
                            }
                        }
                    }
                    _ => {
                        // Scalar with compound initializer: C allows `int x = {42};`
                        if let Some(item) = items.first() {
                            self.analyze_initializer(&item.initializer, target_type, span);
                        }
                    }
                }
            }
        }
    }

    /// Analyzes a struct definition and returns the corresponding CType.
    fn analyze_struct_def(
        &mut self,
        struct_def: &crate::frontend::parser::ast::StructDef,
        is_union: bool,
    ) -> CType {
        let tag_name = struct_def
            .tag
            .map(|id| self.interner.resolve(id).to_string());

        let mut fields = Vec::new();
        for member in &struct_def.members {
            match member {
                crate::frontend::parser::ast::StructMember::Field {
                    specifiers,
                    declarators,
                    ..
                } => {
                    let base_type = self.resolve_type_specifier(&specifiers.type_specifier);

                    // C11 §6.7.2.1p13: If the member declaration list
                    // has no declarators and the type is a struct/union,
                    // this is an anonymous struct/union member whose
                    // fields are promoted into the enclosing type.
                    if declarators.is_empty() {
                        if matches!(base_type.unqualified().canonical(), CType::Struct(_)) {
                            fields.push(StructField {
                                name: None,
                                ty: base_type.clone(),
                                bit_width: None,
                                offset: 0,
                            });
                        }
                        // Else: bare type specifier like `int;` — valid
                        // but has no effect.
                    }

                    for field_decl in declarators {
                        let field_type = if let Some(ref decl) = field_decl.declarator {
                            self.apply_declarator_to_type(&base_type, decl)
                        } else {
                            base_type.clone()
                        };
                        let field_name = field_decl
                            .declarator
                            .as_ref()
                            .and_then(|d| self.extract_declarator_name(d))
                            .map(|id| self.interner.resolve(id).to_string());
                        let bit_width = field_decl.bit_width.as_ref().map(|_| 0u32);
                        fields.push(StructField {
                            name: field_name,
                            ty: field_type,
                            bit_width,
                            offset: 0,
                        });
                    }
                }
                crate::frontend::parser::ast::StructMember::Anonymous { type_spec, .. } => {
                    let anon_type = self.resolve_type_specifier(type_spec);
                    fields.push(StructField {
                        name: None,
                        ty: anon_type,
                        bit_width: None,
                        offset: 0,
                    });
                }
                crate::frontend::parser::ast::StructMember::StaticAssert {
                    expr,
                    message,
                    span,
                } => {
                    self.analyze_static_assert(expr, message, *span);
                }
            }
        }

        let st = StructType {
            tag: tag_name,
            fields,
            is_union,
            is_packed: false,
            custom_alignment: None,
            is_complete: true,
        };

        let ty = CType::Struct(st);

        // Register the tag in the tag namespace.
        if let Some(tag_id) = struct_def.tag {
            let kind = if is_union {
                SymbolKind::UnionTag
            } else {
                SymbolKind::StructTag
            };
            let tag_sym = Symbol {
                name: tag_id,
                ty: ty.clone(),
                storage_class: None,
                linkage: Linkage::None,
                is_defined: true,
                is_tentative: false,
                kind,
                location: struct_def.span,
                scope_depth: self.scope_stack.depth(),
            };
            let _ = self
                .symbol_table
                .insert_tag(tag_id, tag_sym, self.diagnostics);
        }

        ty
    }

    /// Analyzes a union definition.
    fn analyze_union_def(&mut self, union_def: &crate::frontend::parser::ast::UnionDef) -> CType {
        // Unions reuse the struct analysis with is_union = true.
        // Convert UnionDef to the same struct analysis flow.
        let tag_name = union_def
            .tag
            .map(|id| self.interner.resolve(id).to_string());

        let mut fields = Vec::new();
        for member in &union_def.members {
            match member {
                crate::frontend::parser::ast::StructMember::Field {
                    specifiers,
                    declarators,
                    ..
                } => {
                    let base_type = self.resolve_type_specifier(&specifiers.type_specifier);

                    // C11 §6.7.2.1p13: If the member declaration list
                    // has no declarators and the type is a struct/union,
                    // this is an anonymous struct/union member whose
                    // fields are promoted into the enclosing type.
                    if declarators.is_empty() {
                        if matches!(base_type.unqualified().canonical(), CType::Struct(_)) {
                            fields.push(StructField {
                                name: None,
                                ty: base_type.clone(),
                                bit_width: None,
                                offset: 0,
                            });
                        }
                        // Else: bare type specifier like `int;` — valid
                        // but has no effect.
                    }

                    for field_decl in declarators {
                        let field_type = if let Some(ref decl) = field_decl.declarator {
                            self.apply_declarator_to_type(&base_type, decl)
                        } else {
                            base_type.clone()
                        };
                        let field_name = field_decl
                            .declarator
                            .as_ref()
                            .and_then(|d| self.extract_declarator_name(d))
                            .map(|id| self.interner.resolve(id).to_string());
                        let bit_width = field_decl.bit_width.as_ref().map(|_| 0u32);
                        fields.push(StructField {
                            name: field_name,
                            ty: field_type,
                            bit_width,
                            offset: 0,
                        });
                    }
                }
                crate::frontend::parser::ast::StructMember::Anonymous { type_spec, .. } => {
                    let anon_type = self.resolve_type_specifier(type_spec);
                    fields.push(StructField {
                        name: None,
                        ty: anon_type,
                        bit_width: None,
                        offset: 0,
                    });
                }
                crate::frontend::parser::ast::StructMember::StaticAssert {
                    expr,
                    message,
                    span,
                } => {
                    self.analyze_static_assert(expr, message, *span);
                }
            }
        }

        let st = StructType {
            tag: tag_name,
            fields,
            is_union: true,
            is_packed: false,
            custom_alignment: None,
            is_complete: true,
        };

        let ty = CType::Struct(st);

        if let Some(tag_id) = union_def.tag {
            let tag_sym = Symbol {
                name: tag_id,
                ty: ty.clone(),
                storage_class: None,
                linkage: Linkage::None,
                is_defined: true,
                is_tentative: false,
                kind: SymbolKind::UnionTag,
                location: union_def.span,
                scope_depth: self.scope_stack.depth(),
            };
            let _ = self
                .symbol_table
                .insert_tag(tag_id, tag_sym, self.diagnostics);
        }

        ty
    }

    /// Analyzes an enum definition and returns the corresponding CType.
    fn analyze_enum_def(&mut self, enum_def: &crate::frontend::parser::ast::EnumDef) -> CType {
        let tag_name = enum_def.tag.map(|id| self.interner.resolve(id).to_string());

        let mut variants = Vec::new();
        let mut next_value: i64 = 0;

        for variant in &enum_def.variants {
            let value = if let Some(ref value_expr) = variant.value {
                let _ty = self.analyze_expression(value_expr);
                // Simplified: constant evaluation would determine the value.
                // For now, use sequential values.
                next_value
            } else {
                next_value
            };

            variants.push((self.interner.resolve(variant.name).to_string(), value));

            // Register enum constant in ordinary namespace.
            let _ = self.symbol_table.insert_enum_constant(
                variant.name,
                value,
                CType::Integer(IntegerKind::Int),
                self.diagnostics,
                variant.span,
            );

            next_value = value + 1;
        }

        let et = EnumType {
            tag: tag_name,
            variants,
            is_complete: true,
        };

        let ty = CType::Enum(et);

        if let Some(tag_id) = enum_def.tag {
            let tag_sym = Symbol {
                name: tag_id,
                ty: ty.clone(),
                storage_class: None,
                linkage: Linkage::None,
                is_defined: true,
                is_tentative: false,
                kind: SymbolKind::EnumTag,
                location: enum_def.span,
                scope_depth: self.scope_stack.depth(),
            };
            let _ = self
                .symbol_table
                .insert_tag(tag_id, tag_sym, self.diagnostics);
        }

        ty
    }

    /// Returns the `size_t` type for the current target.
    fn size_t_type(&self) -> CType {
        if self.target.is_64bit() {
            CType::Integer(IntegerKind::UnsignedLong)
        } else {
            CType::Integer(IntegerKind::UnsignedInt)
        }
    }

    /// Resolves the type of an integer literal based on its suffix.
    fn resolve_integer_literal_type(
        &self,
        suffix: &crate::frontend::parser::ast::IntSuffix,
    ) -> CType {
        use crate::frontend::parser::ast::IntSuffix;
        match suffix {
            IntSuffix::None => CType::Integer(IntegerKind::Int),
            IntSuffix::Unsigned => CType::Integer(IntegerKind::UnsignedInt),
            IntSuffix::Long => CType::Integer(IntegerKind::Long),
            IntSuffix::ULong => CType::Integer(IntegerKind::UnsignedLong),
            IntSuffix::LongLong => CType::Integer(IntegerKind::LongLong),
            IntSuffix::ULongLong => CType::Integer(IntegerKind::UnsignedLongLong),
        }
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::intern::Interner;
    use crate::common::source_map::{SourceLocation, SourceSpan};
    use crate::driver::target::TargetConfig;
    use crate::frontend::parser::ast::*;

    /// Creates a dummy SourceSpan for test AST nodes.
    fn dummy_span() -> SourceSpan {
        SourceSpan::dummy()
    }

    /// Creates a simple DeclSpecifiers with just a type specifier.
    fn simple_specifiers(type_spec: TypeSpecifier) -> DeclSpecifiers {
        DeclSpecifiers {
            storage_class: None,
            type_qualifiers: Vec::new(),
            type_specifier: type_spec,
            function_specifiers: Vec::new(),
            attributes: Vec::new(),
            alignment: None,
            span: dummy_span(),
        }
    }

    /// Creates a simple identifier declarator.
    fn simple_declarator(name: InternId) -> Declarator {
        Declarator {
            pointer: Vec::new(),
            direct: DirectDeclarator::Identifier(name),
            attributes: Vec::new(),
            span: dummy_span(),
        }
    }

    // -----------------------------------------------------------------------
    // Test: Analyze a simple variable declaration
    // -----------------------------------------------------------------------

    #[test]
    fn test_analyze_simple_variable_declaration() {
        let mut interner = Interner::new();
        let x_id = interner.intern("x");
        let target = TargetConfig::x86_64();
        let mut diagnostics = DiagnosticEmitter::new();

        let ast = TranslationUnit {
            declarations: vec![Declaration::Variable {
                specifiers: simple_specifiers(TypeSpecifier::Int),
                declarators: vec![InitDeclarator {
                    declarator: simple_declarator(x_id),
                    initializer: None,
                    span: dummy_span(),
                }],
                span: dummy_span(),
            }],
            span: dummy_span(),
        };

        let result = SemanticAnalyzer::analyze(&ast, &target, &interner, &mut diagnostics);
        assert!(result.is_ok());
        assert!(!diagnostics.has_errors());
    }

    // -----------------------------------------------------------------------
    // Test: Analyze a function definition
    // -----------------------------------------------------------------------

    #[test]
    fn test_analyze_function_definition() {
        let mut interner = Interner::new();
        let main_id = interner.intern("main");
        let target = TargetConfig::x86_64();
        let mut diagnostics = DiagnosticEmitter::new();

        let return_stmt = Statement::Return {
            value: Some(Box::new(Expression::IntegerLiteral {
                value: 0,
                suffix: IntSuffix::None,
                base: NumericBase::Decimal,
                span: dummy_span(),
            })),
            span: dummy_span(),
        };

        let func_def = FunctionDef {
            specifiers: simple_specifiers(TypeSpecifier::Int),
            declarator: Declarator {
                pointer: Vec::new(),
                direct: DirectDeclarator::Function {
                    base: Box::new(DirectDeclarator::Identifier(main_id)),
                    params: ParamList {
                        params: Vec::new(),
                        variadic: false,
                        span: dummy_span(),
                    },
                },
                attributes: Vec::new(),
                span: dummy_span(),
            },
            body: Box::new(Statement::Compound {
                items: vec![BlockItem::Statement(return_stmt)],
                span: dummy_span(),
            }),
            attributes: Vec::new(),
            span: dummy_span(),
        };

        let ast = TranslationUnit {
            declarations: vec![Declaration::Function(Box::new(func_def))],
            span: dummy_span(),
        };

        let result = SemanticAnalyzer::analyze(&ast, &target, &interner, &mut diagnostics);
        assert!(result.is_ok());
        assert!(!diagnostics.has_errors());
    }

    // -----------------------------------------------------------------------
    // Test: Undeclared identifier detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_undeclared_identifier_detection() {
        let mut interner = Interner::new();
        let main_id = interner.intern("main");
        let x_id = interner.intern("x");
        let target = TargetConfig::x86_64();
        let mut diagnostics = DiagnosticEmitter::new();

        // Function that uses undeclared variable `x`.
        let func_def = FunctionDef {
            specifiers: simple_specifiers(TypeSpecifier::Int),
            declarator: Declarator {
                pointer: Vec::new(),
                direct: DirectDeclarator::Function {
                    base: Box::new(DirectDeclarator::Identifier(main_id)),
                    params: ParamList {
                        params: Vec::new(),
                        variadic: false,
                        span: dummy_span(),
                    },
                },
                attributes: Vec::new(),
                span: dummy_span(),
            },
            body: Box::new(Statement::Compound {
                items: vec![BlockItem::Statement(Statement::Return {
                    value: Some(Box::new(Expression::Identifier {
                        name: x_id,
                        span: dummy_span(),
                    })),
                    span: dummy_span(),
                })],
                span: dummy_span(),
            }),
            attributes: Vec::new(),
            span: dummy_span(),
        };

        let ast = TranslationUnit {
            declarations: vec![Declaration::Function(Box::new(func_def))],
            span: dummy_span(),
        };

        let result = SemanticAnalyzer::analyze(&ast, &target, &interner, &mut diagnostics);
        assert!(result.is_err());
        assert!(diagnostics.has_errors());
    }

    // -----------------------------------------------------------------------
    // Test: Break outside loop detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_break_outside_loop() {
        let mut interner = Interner::new();
        let main_id = interner.intern("main");
        let target = TargetConfig::x86_64();
        let mut diagnostics = DiagnosticEmitter::new();

        let func_def = FunctionDef {
            specifiers: simple_specifiers(TypeSpecifier::Int),
            declarator: Declarator {
                pointer: Vec::new(),
                direct: DirectDeclarator::Function {
                    base: Box::new(DirectDeclarator::Identifier(main_id)),
                    params: ParamList {
                        params: Vec::new(),
                        variadic: false,
                        span: dummy_span(),
                    },
                },
                attributes: Vec::new(),
                span: dummy_span(),
            },
            body: Box::new(Statement::Compound {
                items: vec![BlockItem::Statement(Statement::Break {
                    span: dummy_span(),
                })],
                span: dummy_span(),
            }),
            attributes: Vec::new(),
            span: dummy_span(),
        };

        let ast = TranslationUnit {
            declarations: vec![Declaration::Function(Box::new(func_def))],
            span: dummy_span(),
        };

        let result = SemanticAnalyzer::analyze(&ast, &target, &interner, &mut diagnostics);
        assert!(result.is_err());
        assert!(diagnostics.has_errors());
    }

    // -----------------------------------------------------------------------
    // Test: Continue outside loop detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_continue_outside_loop() {
        let mut interner = Interner::new();
        let main_id = interner.intern("main");
        let target = TargetConfig::x86_64();
        let mut diagnostics = DiagnosticEmitter::new();

        let func_def = FunctionDef {
            specifiers: simple_specifiers(TypeSpecifier::Int),
            declarator: Declarator {
                pointer: Vec::new(),
                direct: DirectDeclarator::Function {
                    base: Box::new(DirectDeclarator::Identifier(main_id)),
                    params: ParamList {
                        params: Vec::new(),
                        variadic: false,
                        span: dummy_span(),
                    },
                },
                attributes: Vec::new(),
                span: dummy_span(),
            },
            body: Box::new(Statement::Compound {
                items: vec![BlockItem::Statement(Statement::Continue {
                    span: dummy_span(),
                })],
                span: dummy_span(),
            }),
            attributes: Vec::new(),
            span: dummy_span(),
        };

        let ast = TranslationUnit {
            declarations: vec![Declaration::Function(Box::new(func_def))],
            span: dummy_span(),
        };

        let result = SemanticAnalyzer::analyze(&ast, &target, &interner, &mut diagnostics);
        assert!(result.is_err());
        assert!(diagnostics.has_errors());
    }

    // -----------------------------------------------------------------------
    // Test: Empty translation unit
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_translation_unit() {
        let interner = Interner::new();
        let target = TargetConfig::x86_64();
        let mut diagnostics = DiagnosticEmitter::new();

        let ast = TranslationUnit {
            declarations: Vec::new(),
            span: dummy_span(),
        };

        let result = SemanticAnalyzer::analyze(&ast, &target, &interner, &mut diagnostics);
        assert!(result.is_ok());
        let typed_tu = result.unwrap();
        assert!(typed_tu.declarations.is_empty());
    }

    // -----------------------------------------------------------------------
    // Test: Target-aware type sizing (i686 vs x86-64)
    // -----------------------------------------------------------------------

    #[test]
    fn test_target_aware_sizing() {
        let target_64 = TargetConfig::x86_64();
        let target_32 = TargetConfig::i686();

        // long is 8 bytes on x86-64, 4 bytes on i686
        let long_type = CType::Integer(IntegerKind::Long);
        assert_eq!(long_type.size(&target_64), Some(8));
        assert_eq!(long_type.size(&target_32), Some(4));

        // Pointer is 8 bytes on x86-64, 4 bytes on i686
        let ptr_type = CType::Pointer {
            pointee: Box::new(CType::Integer(IntegerKind::Int)),
            qualifiers: TypeQualifiers::default(),
        };
        assert_eq!(ptr_type.size(&target_64), Some(8));
        assert_eq!(ptr_type.size(&target_32), Some(4));
    }
}
