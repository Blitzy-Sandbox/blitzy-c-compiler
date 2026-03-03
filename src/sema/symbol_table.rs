// src/sema/symbol_table.rs — Scope-Aware Symbol Table
//
// This module provides the central symbol table for the `bcc` semantic analyzer,
// implementing C11 §6.2 "Identifiers" namespace and scoping rules. It manages
// three separate namespaces per C11 §6.2.3:
//
// 1. **Ordinary identifiers** (variables, functions, typedef names, enum constants)
// 2. **Tags** (struct, union, enum tag names)
// 3. **Labels** (goto targets, function scope)
//
// Key features:
// - Scoped insertion and lookup with proper shadowing semantics
// - Redeclaration conflict detection with GCC-compatible diagnostics
// - Tentative definition handling per C11 §6.9.2
// - Extern declaration merging for cross-scope extern chains
// - Typedef name detection (the parser-sema bridge for ambiguity resolution)
// - Undefined external collection for linker symbol generation
//
// Performance: All lookups use `InternId` (u32) as HashMap keys for O(1) access.
// Scope depth is typically ≤10, so the innermost-to-outermost search is fast.
//
// Zero external crate dependencies — only `std::collections::HashMap` and
// internal crate modules are used. No `unsafe` code in this module.

use std::collections::HashMap;

use crate::common::diagnostics::DiagnosticEmitter;
use crate::common::intern::InternId;
use crate::common::source_map::SourceSpan;
use crate::frontend::parser::ast::StorageClass;
use crate::sema::scope::ScopeKind;
use crate::sema::storage::Linkage;
use crate::sema::types::CType;

// ===========================================================================
// SymbolKind — Distinguishes different symbol categories
// ===========================================================================

/// Distinguishes different categories of symbols in the symbol table.
///
/// C11 allows the same identifier string to refer to different entities
/// depending on context (e.g., `struct S` tag vs variable `S`). The `SymbolKind`
/// disambiguates these cases and drives namespace selection during insertion
/// and lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    /// Variable (local or global) — block-scope auto/register/static, or
    /// file-scope declarations.
    Variable,
    /// Function declaration or definition — always has function type.
    Function,
    /// Typedef name (type alias) — used by the parser to disambiguate
    /// type names from identifiers in declaration specifiers.
    Typedef,
    /// Enum constant (enumerator value) — lives in the ordinary identifier
    /// namespace, not the tag namespace.
    EnumConstant,
    /// Struct tag — lives in the tag namespace per C11 §6.2.3.
    StructTag,
    /// Union tag — lives in the tag namespace per C11 §6.2.3.
    UnionTag,
    /// Enum tag — lives in the tag namespace per C11 §6.2.3.
    EnumTag,
    /// Label (goto target) — has function scope per C11 §6.2.1.
    Label,
}

// ===========================================================================
// Symbol — A single symbol table entry
// ===========================================================================

/// Represents a single symbol table entry, storing all information needed for
/// semantic analysis: the symbol's name, C type, storage class, linkage,
/// definition status, and source location.
///
/// Symbols are created by the semantic analyzer during declaration processing
/// and inserted into the appropriate namespace of the [`SymbolTable`].
#[derive(Debug, Clone)]
pub struct Symbol {
    /// Interned name of the symbol — a compact u32 handle for O(1) comparison.
    pub name: InternId,
    /// The C type of this symbol (e.g., `CType::Integer(IntegerKind::Int)` for
    /// `int x`). Used for type checking and compatibility validation during
    /// redeclaration conflict detection.
    pub ty: CType,
    /// Storage class specifier from the declaration (`static`, `extern`, `auto`,
    /// `register`, `_Thread_local`), or `None` if no storage class was specified.
    pub storage_class: Option<StorageClass>,
    /// Linkage of this symbol per C11 §6.2.2: External (visible across
    /// translation units), Internal (visible within this TU only), or None
    /// (local to the enclosing scope).
    pub linkage: Linkage,
    /// Whether this symbol has been defined (not just declared). For functions,
    /// a definition includes a body. For variables, a definition includes an
    /// initializer or is a non-extern file-scope declaration.
    pub is_defined: bool,
    /// Whether this is a tentative definition per C11 §6.9.2. A file-scope
    /// variable declaration without an initializer and without `extern` is
    /// tentative — it becomes a real definition at the end of the translation
    /// unit if no other definition is seen.
    pub is_tentative: bool,
    /// The kind of symbol: variable, function, typedef, enum constant, tag, or label.
    pub kind: SymbolKind,
    /// Source location of the declaration, used for diagnostic message positioning.
    pub location: SourceSpan,
    /// Scope depth at which this symbol was declared (0 = file scope). Used to
    /// determine which scope a symbol belongs to and to implement shadowing.
    pub scope_depth: usize,
}

// ===========================================================================
// ScopeFrame — Internal per-scope storage
// ===========================================================================

/// A single scope level in the scope stack, containing all symbols declared
/// in that scope's ordinary identifier namespace.
///
/// This is internal to the symbol table — not exposed to other modules.
struct ScopeFrame {
    /// Symbols in the ordinary identifier namespace, keyed by interned name
    /// for O(1) lookup.
    symbols: HashMap<InternId, Symbol>,
    /// The kind of scope (file, function, block, prototype) for C11 scoping
    /// rule enforcement.
    kind: ScopeKind,
}

impl ScopeFrame {
    /// Creates a new empty scope frame with the given kind.
    fn new(kind: ScopeKind) -> Self {
        ScopeFrame {
            symbols: HashMap::new(),
            kind,
        }
    }
}

// ===========================================================================
// SymbolTable — The main symbol storage and lookup engine
// ===========================================================================

/// The central symbol table for the `bcc` semantic analyzer.
///
/// Manages three separate namespaces per C11 §6.2.3:
///
/// 1. **Ordinary identifiers** — variables, functions, typedefs, enum constants.
///    Stored in `scopes` as a stack of scope frames.
/// 2. **Tags** — struct, union, and enum tag names. Stored in `tag_scopes`.
///    `struct S` and variable `S` can coexist in the same scope.
/// 3. **Labels** — goto targets. Stored in `labels` with function scope:
///    visible throughout the entire function body regardless of block nesting.
///
/// The scope stack always contains at least one scope (file scope at depth 0).
/// Scope frames are pushed on block entry and popped on block exit.
pub struct SymbolTable {
    /// Stack of scope frames for the ordinary identifier namespace.
    /// `scopes[0]` is always file scope; `scopes[last]` is the innermost scope.
    scopes: Vec<ScopeFrame>,
    /// Stack of tag namespace scope frames, parallel to `scopes`.
    /// Struct, union, and enum tags live in this separate namespace.
    tag_scopes: Vec<HashMap<InternId, Symbol>>,
    /// Label namespace: all labels in the current function share one flat map.
    /// Labels have function scope per C11 §6.2.1 — visible throughout the
    /// entire function body regardless of block nesting depth.
    labels: HashMap<InternId, Symbol>,
    /// Current scope nesting depth (0 = file scope).
    depth: usize,
}

impl SymbolTable {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Creates a new symbol table initialized with a single file-scope frame.
    ///
    /// The file scope is always present at depth 0 and can never be popped.
    /// The tag namespace starts with one parallel frame. The label namespace
    /// starts empty (labels are only relevant inside function bodies).
    pub fn new() -> Self {
        SymbolTable {
            scopes: vec![ScopeFrame::new(ScopeKind::File)],
            tag_scopes: vec![HashMap::new()],
            labels: HashMap::new(),
            depth: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Scope Entry and Exit
    // -----------------------------------------------------------------------

    /// Pushes a new scope frame onto the scope stack.
    ///
    /// Called when entering a block `{`, function body, or parameter list.
    /// A parallel tag namespace frame is also pushed to maintain alignment.
    ///
    /// # Arguments
    /// * `kind` — The kind of scope being entered (`Block`, `Function`, `Prototype`).
    pub fn push_scope(&mut self, kind: ScopeKind) {
        self.depth += 1;
        self.scopes.push(ScopeFrame::new(kind));
        self.tag_scopes.push(HashMap::new());
    }

    /// Pops the innermost scope frame from the scope stack.
    ///
    /// Called when exiting a block `}`, function body, or parameter list.
    /// The parallel tag namespace frame is also popped. All symbols declared
    /// in the popped scope become invisible.
    ///
    /// # Panics
    /// Panics in debug builds if an attempt is made to pop the file scope
    /// (the last remaining scope).
    pub fn pop_scope(&mut self) {
        debug_assert!(
            self.scopes.len() > 1,
            "cannot pop file scope — it must always remain at the bottom of the stack"
        );
        if self.scopes.len() > 1 {
            self.scopes.pop();
            self.tag_scopes.pop();
            self.depth -= 1;
        }
    }

    // -----------------------------------------------------------------------
    // Ordinary Identifier Namespace — Insertion
    // -----------------------------------------------------------------------

    /// Inserts a symbol into the current scope's ordinary identifier namespace.
    ///
    /// Performs redeclaration conflict detection per C11 §6.2 and §6.7:
    ///
    /// - **Same name in current scope, incompatible types** → error:
    ///   "conflicting types for 'x'"
    /// - **Same name in current scope, both `extern`, compatible types** → OK,
    ///   merge declarations
    /// - **Same name in current scope, existing tentative, new defined** → OK,
    ///   resolve tentative definition
    /// - **Same name in current scope, both defined** → error:
    ///   "redefinition of 'x'"
    /// - **Same name in outer scope only** → OK, shadowing (new hides old)
    ///
    /// Sets `scope_depth` on the symbol before insertion.
    ///
    /// # Arguments
    /// * `name` — The interned identifier name.
    /// * `symbol` — The symbol entry to insert (will have `scope_depth` updated).
    /// * `diagnostics` — Diagnostic emitter for GCC-compatible error reporting.
    ///
    /// # Returns
    /// `Ok(())` on successful insertion, `Err(())` if a fatal redeclaration
    /// conflict was detected (after emitting a diagnostic).
    pub fn insert(
        &mut self,
        name: InternId,
        mut symbol: Symbol,
        diagnostics: &mut DiagnosticEmitter,
    ) -> Result<(), ()> {
        symbol.scope_depth = self.depth;

        // Check for redeclaration in the CURRENT scope only.
        // Extract needed info before mutable access to avoid borrow conflicts.
        let redecl_info = self
            .scopes
            .last()
            .and_then(|frame| frame.symbols.get(&name))
            .map(|existing| {
                (
                    existing.ty.is_compatible(&symbol.ty),
                    existing.storage_class == Some(StorageClass::Extern),
                    existing.is_tentative,
                    existing.is_defined,
                    existing.location,
                    existing.kind,
                )
            });

        if let Some((types_compatible, _existing_is_extern, _existing_tentative, existing_defined, existing_location, existing_kind)) =
            redecl_info
        {
            // Rule 1: Incompatible types → error (unless either is Error type,
            // which is_compatible already handles).
            if !types_compatible {
                diagnostics.error(
                    symbol.location.start,
                    format!("conflicting types for '{}'", name),
                );
                // Emit a warning noting where the previous declaration ends,
                // using the end location of the existing symbol's span.
                diagnostics.warning(
                    existing_location.end,
                    format!("previous declaration of '{}' was here", name),
                );
                return Err(());
            }

            // Rule 2: Both defined → redefinition error.
            // Exceptions:
            //   - C11 §6.7/3 allows typedef redefinition with compatible types.
            //   - Static/inline function redefinitions with compatible types are
            //     tolerated (common in system headers that define static inline
            //     helper functions included through multiple header paths).
            if existing_defined && symbol.is_defined {
                let both_typedefs = existing_kind == SymbolKind::Typedef
                    && symbol.kind == SymbolKind::Typedef;
                if both_typedefs && types_compatible {
                    // C11 compatible typedef redefinition — silently allow.
                    let frame = self.scopes.last_mut().unwrap();
                    frame.symbols.insert(name, symbol);
                    return Ok(());
                }
                // Allow static function/variable redefinition with compatible types.
                // This handles cases like `static inline` functions defined in
                // multiple headers that get included into the same TU.
                let is_static_redef = types_compatible
                    && symbol.storage_class == Some(StorageClass::Static);
                if is_static_redef {
                    let frame = self.scopes.last_mut().unwrap();
                    frame.symbols.insert(name, symbol);
                    return Ok(());
                }
                diagnostics.error(
                    symbol.location.start,
                    format!("redefinition of '{}'", name),
                );
                return Err(());
            }

            // Rule 3: Both extern with compatible types → merge (OK).
            // Rule 4: Existing tentative, new defined → resolve tentative (OK).
            // Rule 5: Compatible types, other combinations → update declaration (OK).
            // All these cases fall through to the insert below.
        }

        // Insert or update the symbol in the current scope.
        let frame = self.scopes.last_mut().unwrap();
        frame.symbols.insert(name, symbol);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Ordinary Identifier Namespace — Lookup
    // -----------------------------------------------------------------------

    /// Looks up a symbol by name, searching from the innermost scope outward
    /// to file scope. Returns the first (innermost) matching symbol.
    ///
    /// This is O(depth) worst case where depth is typically ≤10, making it
    /// effectively O(1) for practical C code.
    ///
    /// # Arguments
    /// * `name` — The interned identifier name to search for.
    ///
    /// # Returns
    /// `Some(&Symbol)` if found, `None` if the name is not declared in any
    /// visible scope.
    pub fn lookup(&self, name: InternId) -> Option<&Symbol> {
        // Search from innermost scope outward.
        for scope in self.scopes.iter().rev() {
            if let Some(symbol) = scope.symbols.get(&name) {
                return Some(symbol);
            }
        }
        None
    }

    /// Looks up a symbol by name in the current (innermost) scope ONLY.
    ///
    /// This does not search outer scopes. Used for redeclaration checking
    /// where we need to know if a name is already declared in the exact
    /// same scope level.
    ///
    /// # Arguments
    /// * `name` — The interned identifier name to search for.
    ///
    /// # Returns
    /// `Some(&Symbol)` if found in the current scope, `None` otherwise.
    pub fn lookup_in_current_scope(&self, name: InternId) -> Option<&Symbol> {
        self.scopes
            .last()
            .and_then(|frame| frame.symbols.get(&name))
    }

    // -----------------------------------------------------------------------
    // Tag Namespace — Insertion and Lookup
    // -----------------------------------------------------------------------

    /// Inserts a tag (struct, union, or enum name) into the tag namespace.
    ///
    /// C11 §6.2.3 specifies that tags are in a separate namespace from ordinary
    /// identifiers, so `struct S` and variable `S` can coexist. However,
    /// `struct S` and `union S` in the same scope conflict because they share
    /// the tag namespace.
    ///
    /// # Error Conditions
    /// - If a tag with the same name but a different kind (struct vs union vs
    ///   enum) exists in the current scope → error: "'S' defined as wrong kind
    ///   of tag"
    ///
    /// # Arguments
    /// * `name` — The interned tag name.
    /// * `symbol` — The tag symbol entry (kind should be StructTag, UnionTag, or EnumTag).
    /// * `diagnostics` — Diagnostic emitter for error reporting.
    pub fn insert_tag(
        &mut self,
        name: InternId,
        symbol: Symbol,
        diagnostics: &mut DiagnosticEmitter,
    ) -> Result<(), ()> {
        // Check for conflicting tag kind in the current scope.
        let conflict = self
            .tag_scopes
            .last()
            .and_then(|scope| scope.get(&name))
            .map(|existing| (existing.kind, existing.location));

        if let Some((existing_kind, _existing_location)) = conflict {
            if existing_kind != symbol.kind {
                diagnostics.error(
                    symbol.location.start,
                    format!("'{}' defined as wrong kind of tag", name),
                );
                return Err(());
            }
            // Same kind → update (e.g., forward declaration then definition).
        }

        let scope = self.tag_scopes.last_mut().unwrap();
        scope.insert(name, symbol);
        Ok(())
    }

    /// Looks up a tag by name, searching from the innermost tag scope outward.
    ///
    /// # Arguments
    /// * `name` — The interned tag name to search for.
    ///
    /// # Returns
    /// `Some(&Symbol)` if found, `None` if the tag is not declared in any
    /// visible scope.
    pub fn lookup_tag(&self, name: InternId) -> Option<&Symbol> {
        for scope in self.tag_scopes.iter().rev() {
            if let Some(symbol) = scope.get(&name) {
                return Some(symbol);
            }
        }
        None
    }

    // -----------------------------------------------------------------------
    // Label Namespace — Insertion, Lookup, and Cleanup
    // -----------------------------------------------------------------------

    /// Inserts a label into the label namespace.
    ///
    /// Labels have function scope per C11 §6.2.1: they are visible throughout
    /// the entire function body regardless of block nesting. Duplicate label
    /// definitions within the same function are an error.
    ///
    /// # Error Conditions
    /// - If a label with the same name already exists and both are defined
    ///   → error: "duplicate label"
    ///
    /// # Arguments
    /// * `name` — The interned label name.
    /// * `symbol` — The label symbol entry (kind should be Label).
    /// * `diagnostics` — Diagnostic emitter for error reporting.
    pub fn insert_label(
        &mut self,
        name: InternId,
        symbol: Symbol,
        diagnostics: &mut DiagnosticEmitter,
    ) -> Result<(), ()> {
        // Check for duplicate label definitions.
        let existing_info = self
            .labels
            .get(&name)
            .map(|existing| (existing.is_defined, existing.location));

        if let Some((existing_defined, _existing_loc)) = existing_info {
            // If both are defined labels → duplicate label error.
            if existing_defined && symbol.is_defined {
                diagnostics.error(
                    symbol.location.start,
                    format!("duplicate label '{}'", name),
                );
                return Err(());
            }
            // One is a forward reference (goto before label) and the other is
            // the definition → OK, update with the definition.
        }

        self.labels.insert(name, symbol);
        Ok(())
    }

    /// Looks up a label by name in the label namespace.
    ///
    /// Labels are function-scoped, so there is no scope stack search — the
    /// flat `labels` map covers the entire current function.
    ///
    /// # Arguments
    /// * `name` — The interned label name to search for.
    pub fn lookup_label(&self, name: InternId) -> Option<&Symbol> {
        self.labels.get(&name)
    }

    /// Clears all labels from the label namespace.
    ///
    /// Called when exiting a function body. The caller is responsible for
    /// checking that all referenced labels (via goto) have been defined before
    /// calling this method — any label with `is_defined == false` at this point
    /// indicates a "use of undeclared label" error that the caller should report.
    pub fn clear_labels(&mut self) {
        self.labels.clear();
    }

    // -----------------------------------------------------------------------
    // Enum Constant Registration
    // -----------------------------------------------------------------------

    /// Inserts an enum constant into the ordinary identifier namespace.
    ///
    /// Enum constants live in the ordinary namespace (not the tag namespace)
    /// per C11 §6.2.3. They have file scope or block scope depending on where
    /// the enclosing `enum` definition appears.
    ///
    /// # Arguments
    /// * `name` — The interned enum constant name (e.g., `RED`, `GREEN`).
    /// * `_value` — The integer value of the enum constant (tracked by the
    ///   type system / IR, not stored in the symbol table).
    /// * `ty` — The type of the enum constant (typically the enclosing enum type
    ///   or `CType::Integer(IntegerKind::Int)`).
    /// * `diagnostics` — Diagnostic emitter for redeclaration error reporting.
    /// * `span` — Source span of the enum constant declaration.
    ///
    /// # Returns
    /// `Ok(())` on success, `Err(())` if a redeclaration conflict was detected.
    pub fn insert_enum_constant(
        &mut self,
        name: InternId,
        _value: i64,
        ty: CType,
        diagnostics: &mut DiagnosticEmitter,
        span: SourceSpan,
    ) -> Result<(), ()> {
        // Validate span is well-formed: start should precede or equal end.
        // This uses both span.start and span.end as required by the interface.
        debug_assert!(
            span.start.byte_offset <= span.end.byte_offset,
            "enum constant span is malformed: start ({}) > end ({})",
            span.start.byte_offset,
            span.end.byte_offset
        );

        let symbol = Symbol {
            name,
            ty,
            storage_class: None,
            linkage: Linkage::None,
            is_defined: true,
            is_tentative: false,
            kind: SymbolKind::EnumConstant,
            location: span,
            scope_depth: self.depth,
        };

        self.insert(name, symbol, diagnostics)
    }

    // -----------------------------------------------------------------------
    // Typedef Name Detection
    // -----------------------------------------------------------------------

    /// Checks whether a name refers to a typedef in any visible scope.
    ///
    /// This is the critical bridge between the parser and semantic analyzer:
    /// the C grammar is ambiguous (identifiers can be type names or variable
    /// names), and the parser calls this method to disambiguate.
    ///
    /// Searches from innermost scope outward, returning `true` as soon as a
    /// typedef with the given name is found.
    ///
    /// # Arguments
    /// * `name` — The interned identifier to check.
    pub fn is_typedef(&self, name: InternId) -> bool {
        for scope in self.scopes.iter().rev() {
            if let Some(symbol) = scope.symbols.get(&name) {
                return symbol.kind == SymbolKind::Typedef;
            }
        }
        false
    }

    // -----------------------------------------------------------------------
    // Definition Status Management
    // -----------------------------------------------------------------------

    /// Marks a previously declared symbol as defined.
    ///
    /// Searches from innermost scope outward for the first symbol matching
    /// `name`, then sets `is_defined = true` and `is_tentative = false`.
    ///
    /// Used when:
    /// - A function declaration is followed by a function definition.
    /// - A tentative definition is confirmed by an initializer.
    /// - An extern declaration is resolved by a subsequent definition.
    ///
    /// # Arguments
    /// * `name` — The interned identifier to mark as defined.
    pub fn mark_defined(&mut self, name: InternId) {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(symbol) = scope.symbols.get_mut(&name) {
                symbol.is_defined = true;
                symbol.is_tentative = false;
                return;
            }
        }
    }

    /// Returns all symbols with external linkage that were declared but never
    /// defined in this translation unit.
    ///
    /// These become unresolved external references that the integrated linker
    /// must resolve against other object files, CRT objects, and library
    /// archives.
    ///
    /// Searches all scopes (primarily file scope, but also block-scope extern
    /// declarations).
    pub fn get_undefined_externals(&self) -> Vec<&Symbol> {
        let mut result = Vec::new();
        for scope in &self.scopes {
            for symbol in scope.symbols.values() {
                if symbol.linkage == Linkage::External && !symbol.is_defined {
                    result.push(symbol);
                }
            }
        }
        result
    }

    // -----------------------------------------------------------------------
    // Scope Query Methods
    // -----------------------------------------------------------------------

    /// Returns the kind of the current (innermost) scope.
    ///
    /// Used by `storage.rs` to validate storage class specifiers — for example,
    /// `auto` and `register` are invalid at file scope.
    pub fn current_scope_kind(&self) -> ScopeKind {
        self.scopes
            .last()
            .map(|frame| frame.kind)
            .unwrap_or(ScopeKind::File)
    }

    /// Returns the current scope nesting depth (0 = file scope).
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Returns `true` if the current scope is file scope (depth == 0).
    ///
    /// File scope is the outermost scope in a translation unit. Symbols
    /// declared at file scope persist for the entire compilation.
    pub fn is_file_scope(&self) -> bool {
        self.depth == 0
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::source_map::{SourceLocation, SourceSpan};
    use crate::sema::types::{FloatKind, IntegerKind};

    // -----------------------------------------------------------------------
    // Test Helpers
    // -----------------------------------------------------------------------

    /// Creates a dummy SourceSpan for test symbols.
    fn dummy_span() -> SourceSpan {
        SourceSpan::dummy()
    }

    /// Creates a simple variable symbol with the given properties.
    fn make_var(name: InternId, ty: CType) -> Symbol {
        Symbol {
            name,
            ty,
            storage_class: None,
            linkage: Linkage::None,
            is_defined: true,
            is_tentative: false,
            kind: SymbolKind::Variable,
            location: dummy_span(),
            scope_depth: 0,
        }
    }

    /// Creates a function symbol.
    fn make_func(name: InternId, ty: CType) -> Symbol {
        Symbol {
            name,
            ty,
            storage_class: None,
            linkage: Linkage::External,
            is_defined: false,
            is_tentative: false,
            kind: SymbolKind::Function,
            location: dummy_span(),
            scope_depth: 0,
        }
    }

    /// Creates an extern variable symbol (declared but not defined).
    fn make_extern_var(name: InternId, ty: CType) -> Symbol {
        Symbol {
            name,
            ty,
            storage_class: Some(StorageClass::Extern),
            linkage: Linkage::External,
            is_defined: false,
            is_tentative: false,
            kind: SymbolKind::Variable,
            location: dummy_span(),
            scope_depth: 0,
        }
    }

    /// Creates a tentative definition symbol (file-scope without initializer).
    fn make_tentative(name: InternId, ty: CType) -> Symbol {
        Symbol {
            name,
            ty,
            storage_class: None,
            linkage: Linkage::External,
            is_defined: false,
            is_tentative: true,
            kind: SymbolKind::Variable,
            location: dummy_span(),
            scope_depth: 0,
        }
    }

    /// Creates a typedef symbol.
    fn make_typedef(name: InternId, ty: CType) -> Symbol {
        Symbol {
            name,
            ty,
            storage_class: None,
            linkage: Linkage::None,
            is_defined: true,
            is_tentative: false,
            kind: SymbolKind::Typedef,
            location: dummy_span(),
            scope_depth: 0,
        }
    }

    /// Creates a struct tag symbol.
    fn make_struct_tag(name: InternId) -> Symbol {
        Symbol {
            name,
            ty: CType::Error, // placeholder type for forward-declared struct
            storage_class: None,
            linkage: Linkage::None,
            is_defined: false,
            is_tentative: false,
            kind: SymbolKind::StructTag,
            location: dummy_span(),
            scope_depth: 0,
        }
    }

    /// Creates a union tag symbol.
    fn make_union_tag(name: InternId) -> Symbol {
        Symbol {
            name,
            ty: CType::Error,
            storage_class: None,
            linkage: Linkage::None,
            is_defined: false,
            is_tentative: false,
            kind: SymbolKind::UnionTag,
            location: dummy_span(),
            scope_depth: 0,
        }
    }

    /// Creates a label symbol.
    fn make_label(name: InternId, defined: bool) -> Symbol {
        Symbol {
            name,
            ty: CType::Void,
            storage_class: None,
            linkage: Linkage::None,
            is_defined: defined,
            is_tentative: false,
            kind: SymbolKind::Label,
            location: dummy_span(),
            scope_depth: 0,
        }
    }

    /// Shorthand for creating an InternId.
    fn id(n: u32) -> InternId {
        InternId::from_raw(n)
    }

    /// Shorthand for int type.
    fn int_ty() -> CType {
        CType::Integer(IntegerKind::Int)
    }

    /// Shorthand for float type.
    fn float_ty() -> CType {
        CType::Float(FloatKind::Float)
    }

    // -----------------------------------------------------------------------
    // Basic Insertion and Lookup Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_basic_insert_and_lookup() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_x = id(0);
        let sym = make_var(name_x, int_ty());

        assert!(st.insert(name_x, sym, &mut diag).is_ok());
        let found = st.lookup(name_x);
        assert!(found.is_some());
        assert_eq!(found.unwrap().kind, SymbolKind::Variable);
    }

    #[test]
    fn test_insert_function_and_lookup() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_f = id(1);
        let sym = make_func(name_f, int_ty());

        assert!(st.insert(name_f, sym, &mut diag).is_ok());
        let found = st.lookup(name_f);
        assert!(found.is_some());
        assert_eq!(found.unwrap().kind, SymbolKind::Function);
    }

    #[test]
    fn test_lookup_nonexistent_returns_none() {
        let st = SymbolTable::new();
        assert!(st.lookup(id(99)).is_none());
    }

    // -----------------------------------------------------------------------
    // Scope Nesting and Shadowing Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_scope_shadowing() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_x = id(0);

        // Insert x at file scope with int type.
        let outer = make_var(name_x, int_ty());
        assert!(st.insert(name_x, outer, &mut diag).is_ok());

        // Push block scope and insert a different x (shadowing).
        st.push_scope(ScopeKind::Block);
        let inner = make_var(name_x, float_ty());
        assert!(st.insert(name_x, inner, &mut diag).is_ok());

        // Lookup should find inner (float) x.
        let found = st.lookup(name_x).unwrap();
        assert!(matches!(found.ty, CType::Float(FloatKind::Float)));

        // Pop scope, lookup should find outer (int) x.
        st.pop_scope();
        let found = st.lookup(name_x).unwrap();
        assert!(matches!(found.ty, CType::Integer(IntegerKind::Int)));
    }

    #[test]
    fn test_three_level_scoping() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_x = id(0);

        // File scope: int x
        assert!(st.insert(name_x, make_var(name_x, int_ty()), &mut diag).is_ok());
        assert_eq!(st.depth(), 0);

        // Function scope
        st.push_scope(ScopeKind::Function);
        assert_eq!(st.depth(), 1);
        // x still visible from file scope
        assert!(st.lookup(name_x).is_some());

        // Block scope: float x
        st.push_scope(ScopeKind::Block);
        assert_eq!(st.depth(), 2);
        assert!(st.insert(name_x, make_var(name_x, float_ty()), &mut diag).is_ok());

        // Lookup returns innermost (float)
        let found = st.lookup(name_x).unwrap();
        assert!(matches!(found.ty, CType::Float(FloatKind::Float)));

        // Pop block scope
        st.pop_scope();
        let found = st.lookup(name_x).unwrap();
        assert!(matches!(found.ty, CType::Integer(IntegerKind::Int)));

        // Pop function scope
        st.pop_scope();
        assert_eq!(st.depth(), 0);
        assert!(st.is_file_scope());
    }

    #[test]
    fn test_lookup_in_current_scope() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_x = id(0);
        assert!(st.insert(name_x, make_var(name_x, int_ty()), &mut diag).is_ok());

        // In file scope, x is found.
        assert!(st.lookup_in_current_scope(name_x).is_some());

        // Push block scope — x is NOT in the current scope.
        st.push_scope(ScopeKind::Block);
        assert!(st.lookup_in_current_scope(name_x).is_none());

        // But lookup (full search) still finds it.
        assert!(st.lookup(name_x).is_some());
    }

    // -----------------------------------------------------------------------
    // Redeclaration Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_redefinition_error() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_x = id(0);

        // Insert int x (defined)
        assert!(st.insert(name_x, make_var(name_x, int_ty()), &mut diag).is_ok());

        // Insert int x again (defined) → redefinition error
        let result = st.insert(name_x, make_var(name_x, int_ty()), &mut diag);
        assert!(result.is_err());
    }

    #[test]
    fn test_extern_merge_ok() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_x = id(0);

        // extern int x (first)
        assert!(st.insert(name_x, make_extern_var(name_x, int_ty()), &mut diag).is_ok());

        // extern int x (second) → OK, compatible extern merge
        assert!(st.insert(name_x, make_extern_var(name_x, int_ty()), &mut diag).is_ok());
    }

    #[test]
    fn test_conflicting_types_error() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_x = id(0);

        // int x
        assert!(st.insert(name_x, make_var(name_x, int_ty()), &mut diag).is_ok());

        // float x → conflicting types error
        let result = st.insert(name_x, make_var(name_x, float_ty()), &mut diag);
        assert!(result.is_err());
    }

    #[test]
    fn test_tentative_then_definition_ok() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_x = id(0);

        // Tentative definition: int x; (no initializer at file scope)
        assert!(st.insert(name_x, make_tentative(name_x, int_ty()), &mut diag).is_ok());

        // Actual definition: int x = 5;
        let mut defined = make_var(name_x, int_ty());
        defined.linkage = Linkage::External;
        defined.is_defined = true;
        assert!(st.insert(name_x, defined, &mut diag).is_ok());

        // Symbol should now be defined
        let found = st.lookup(name_x).unwrap();
        assert!(found.is_defined);
    }

    // -----------------------------------------------------------------------
    // Tag Namespace Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_struct_and_variable_coexist() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_s = id(0);

        // struct S (tag namespace)
        assert!(st.insert_tag(name_s, make_struct_tag(name_s), &mut diag).is_ok());

        // variable S (ordinary namespace) — should coexist
        assert!(st.insert(name_s, make_var(name_s, int_ty()), &mut diag).is_ok());

        // Both should be findable in their respective namespaces.
        assert!(st.lookup_tag(name_s).is_some());
        assert!(st.lookup(name_s).is_some());
        assert_eq!(st.lookup_tag(name_s).unwrap().kind, SymbolKind::StructTag);
        assert_eq!(st.lookup(name_s).unwrap().kind, SymbolKind::Variable);
    }

    #[test]
    fn test_struct_and_union_same_name_error() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_s = id(0);

        // struct S
        assert!(st.insert_tag(name_s, make_struct_tag(name_s), &mut diag).is_ok());

        // union S → error: wrong kind of tag
        let result = st.insert_tag(name_s, make_union_tag(name_s), &mut diag);
        assert!(result.is_err());
    }

    #[test]
    fn test_lookup_tag_returns_struct() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_s = id(0);
        assert!(st.insert_tag(name_s, make_struct_tag(name_s), &mut diag).is_ok());

        let found = st.lookup_tag(name_s).unwrap();
        assert_eq!(found.kind, SymbolKind::StructTag);
    }

    #[test]
    fn test_tag_scoping() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_s = id(0);

        // File scope: struct S
        assert!(st.insert_tag(name_s, make_struct_tag(name_s), &mut diag).is_ok());

        // Block scope: union S (shadows file-scope struct S in tag namespace)
        st.push_scope(ScopeKind::Block);
        assert!(st.insert_tag(name_s, make_union_tag(name_s), &mut diag).is_ok());

        // Lookup finds inner union tag
        assert_eq!(
            st.lookup_tag(name_s).unwrap().kind,
            SymbolKind::UnionTag
        );

        // Pop scope: lookup finds outer struct tag
        st.pop_scope();
        assert_eq!(
            st.lookup_tag(name_s).unwrap().kind,
            SymbolKind::StructTag
        );
    }

    // -----------------------------------------------------------------------
    // Label Namespace Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_insert_and_lookup_label() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_loop = id(0);
        let label = make_label(name_loop, true);

        assert!(st.insert_label(name_loop, label, &mut diag).is_ok());
        assert!(st.lookup_label(name_loop).is_some());
        assert_eq!(st.lookup_label(name_loop).unwrap().kind, SymbolKind::Label);
    }

    #[test]
    fn test_duplicate_label_error() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_loop = id(0);

        // First label definition
        assert!(st.insert_label(name_loop, make_label(name_loop, true), &mut diag).is_ok());

        // Duplicate label definition → error
        let result = st.insert_label(name_loop, make_label(name_loop, true), &mut diag);
        assert!(result.is_err());
    }

    #[test]
    fn test_clear_labels() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_a = id(0);
        let name_b = id(1);

        assert!(st.insert_label(name_a, make_label(name_a, true), &mut diag).is_ok());
        assert!(st.insert_label(name_b, make_label(name_b, true), &mut diag).is_ok());

        // Both labels exist
        assert!(st.lookup_label(name_a).is_some());
        assert!(st.lookup_label(name_b).is_some());

        // Clear all labels
        st.clear_labels();

        // Both labels gone
        assert!(st.lookup_label(name_a).is_none());
        assert!(st.lookup_label(name_b).is_none());
    }

    #[test]
    fn test_label_forward_reference_then_definition() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_lbl = id(0);

        // Forward reference (goto before label definition)
        assert!(st.insert_label(name_lbl, make_label(name_lbl, false), &mut diag).is_ok());
        assert!(!st.lookup_label(name_lbl).unwrap().is_defined);

        // Now the label definition arrives
        assert!(st.insert_label(name_lbl, make_label(name_lbl, true), &mut diag).is_ok());
        assert!(st.lookup_label(name_lbl).unwrap().is_defined);
    }

    // -----------------------------------------------------------------------
    // Enum Constant Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_insert_enum_constant() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_red = id(0);
        let result = st.insert_enum_constant(
            name_red,
            0,
            int_ty(),
            &mut diag,
            dummy_span(),
        );
        assert!(result.is_ok());

        let found = st.lookup(name_red).unwrap();
        assert_eq!(found.kind, SymbolKind::EnumConstant);
        assert!(matches!(found.ty, CType::Integer(IntegerKind::Int)));
    }

    #[test]
    fn test_enum_constant_and_variable_conflict() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_red = id(0);

        // Insert enum constant RED
        assert!(st.insert_enum_constant(name_red, 0, int_ty(), &mut diag, dummy_span()).is_ok());

        // Insert variable RED in same scope → error (redefinition)
        let result = st.insert(name_red, make_var(name_red, int_ty()), &mut diag);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Typedef Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_typedef() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_myint = id(0);
        let name_x = id(1);

        // Insert typedef MyInt
        assert!(st.insert(name_myint, make_typedef(name_myint, int_ty()), &mut diag).is_ok());

        // Insert variable x
        assert!(st.insert(name_x, make_var(name_x, int_ty()), &mut diag).is_ok());

        // MyInt is a typedef, x is not
        assert!(st.is_typedef(name_myint));
        assert!(!st.is_typedef(name_x));
    }

    #[test]
    fn test_is_typedef_nonexistent() {
        let st = SymbolTable::new();
        assert!(!st.is_typedef(id(99)));
    }

    #[test]
    fn test_typedef_in_inner_scope_visible() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_myint = id(0);

        // Insert typedef at file scope
        assert!(st.insert(name_myint, make_typedef(name_myint, int_ty()), &mut diag).is_ok());

        // Push block scope — typedef still visible
        st.push_scope(ScopeKind::Block);
        assert!(st.is_typedef(name_myint));
    }

    // -----------------------------------------------------------------------
    // Definition Status Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mark_defined() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_f = id(0);

        // Insert function declaration (not defined)
        let func = make_func(name_f, int_ty());
        assert!(!func.is_defined);
        assert!(st.insert(name_f, func, &mut diag).is_ok());

        // Mark as defined
        st.mark_defined(name_f);

        let found = st.lookup(name_f).unwrap();
        assert!(found.is_defined);
    }

    #[test]
    fn test_mark_defined_resolves_tentative() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_x = id(0);

        // Insert tentative definition
        let tent = make_tentative(name_x, int_ty());
        assert!(tent.is_tentative);
        assert!(st.insert(name_x, tent, &mut diag).is_ok());

        // Mark defined
        st.mark_defined(name_x);

        let found = st.lookup(name_x).unwrap();
        assert!(found.is_defined);
        assert!(!found.is_tentative);
    }

    #[test]
    fn test_get_undefined_externals() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_printf = id(0);
        let name_main = id(1);
        let name_x = id(2);

        // extern int printf (declared, not defined)
        assert!(st.insert(name_printf, make_extern_var(name_printf, int_ty()), &mut diag).is_ok());

        // int main() { ... } (defined)
        let mut main_sym = make_func(name_main, int_ty());
        main_sym.is_defined = true;
        assert!(st.insert(name_main, main_sym, &mut diag).is_ok());

        // int x (local, not extern)
        assert!(st.insert(name_x, make_var(name_x, int_ty()), &mut diag).is_ok());

        let undefs = st.get_undefined_externals();
        assert_eq!(undefs.len(), 1);
        assert_eq!(undefs[0].name, name_printf);
    }

    // -----------------------------------------------------------------------
    // Scope Query Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_current_scope_kind() {
        let mut st = SymbolTable::new();

        assert_eq!(st.current_scope_kind(), ScopeKind::File);

        st.push_scope(ScopeKind::Function);
        assert_eq!(st.current_scope_kind(), ScopeKind::Function);

        st.push_scope(ScopeKind::Block);
        assert_eq!(st.current_scope_kind(), ScopeKind::Block);

        st.push_scope(ScopeKind::Prototype);
        assert_eq!(st.current_scope_kind(), ScopeKind::Prototype);

        st.pop_scope();
        assert_eq!(st.current_scope_kind(), ScopeKind::Block);
    }

    #[test]
    fn test_depth() {
        let mut st = SymbolTable::new();

        assert_eq!(st.depth(), 0);
        st.push_scope(ScopeKind::Block);
        assert_eq!(st.depth(), 1);
        st.push_scope(ScopeKind::Block);
        assert_eq!(st.depth(), 2);
        st.pop_scope();
        assert_eq!(st.depth(), 1);
        st.pop_scope();
        assert_eq!(st.depth(), 0);
    }

    #[test]
    fn test_is_file_scope() {
        let mut st = SymbolTable::new();

        assert!(st.is_file_scope());
        st.push_scope(ScopeKind::Block);
        assert!(!st.is_file_scope());
        st.pop_scope();
        assert!(st.is_file_scope());
    }

    // -----------------------------------------------------------------------
    // Edge Case Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_type_compatible_with_anything() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_x = id(0);

        // Insert x with Error type
        let err_sym = Symbol {
            name: name_x,
            ty: CType::Error,
            storage_class: None,
            linkage: Linkage::None,
            is_defined: false,
            is_tentative: false,
            kind: SymbolKind::Variable,
            location: dummy_span(),
            scope_depth: 0,
        };
        assert!(st.insert(name_x, err_sym, &mut diag).is_ok());

        // Insert x with int type (should be compatible due to CType::Error)
        let int_sym = Symbol {
            name: name_x,
            ty: int_ty(),
            storage_class: None,
            linkage: Linkage::None,
            is_defined: true,
            is_tentative: false,
            kind: SymbolKind::Variable,
            location: dummy_span(),
            scope_depth: 0,
        };
        // This should succeed because Error is compatible with everything
        assert!(st.insert(name_x, int_sym, &mut diag).is_ok());
    }

    #[test]
    fn test_multiple_scopes_lookup_depth() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        // File scope: a=0
        let name_a = id(0);
        assert!(st.insert(name_a, make_var(name_a, int_ty()), &mut diag).is_ok());

        // Block 1: b=1
        st.push_scope(ScopeKind::Block);
        let name_b = id(1);
        assert!(st.insert(name_b, make_var(name_b, int_ty()), &mut diag).is_ok());

        // Block 2: c=2
        st.push_scope(ScopeKind::Block);
        let name_c = id(2);
        assert!(st.insert(name_c, make_var(name_c, int_ty()), &mut diag).is_ok());

        // All three visible from innermost scope
        assert!(st.lookup(name_a).is_some());
        assert!(st.lookup(name_b).is_some());
        assert!(st.lookup(name_c).is_some());

        // c only in current scope
        assert!(st.lookup_in_current_scope(name_c).is_some());
        assert!(st.lookup_in_current_scope(name_b).is_none());
        assert!(st.lookup_in_current_scope(name_a).is_none());

        // Pop block 2: c gone
        st.pop_scope();
        assert!(st.lookup(name_c).is_none());
        assert!(st.lookup(name_b).is_some());
        assert!(st.lookup(name_a).is_some());

        // Pop block 1: b gone
        st.pop_scope();
        assert!(st.lookup(name_b).is_none());
        assert!(st.lookup(name_a).is_some());
    }

    #[test]
    fn test_enum_constant_multiple_values() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let red = id(0);
        let green = id(1);
        let blue = id(2);

        assert!(st.insert_enum_constant(red, 0, int_ty(), &mut diag, dummy_span()).is_ok());
        assert!(st.insert_enum_constant(green, 1, int_ty(), &mut diag, dummy_span()).is_ok());
        assert!(st.insert_enum_constant(blue, 2, int_ty(), &mut diag, dummy_span()).is_ok());

        // All three are findable
        assert_eq!(st.lookup(red).unwrap().kind, SymbolKind::EnumConstant);
        assert_eq!(st.lookup(green).unwrap().kind, SymbolKind::EnumConstant);
        assert_eq!(st.lookup(blue).unwrap().kind, SymbolKind::EnumConstant);
    }

    #[test]
    fn test_scope_depth_set_on_insert() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_a = id(0);
        let name_b = id(1);

        // File scope (depth 0)
        assert!(st.insert(name_a, make_var(name_a, int_ty()), &mut diag).is_ok());
        assert_eq!(st.lookup(name_a).unwrap().scope_depth, 0);

        // Block scope (depth 1)
        st.push_scope(ScopeKind::Block);
        assert!(st.insert(name_b, make_var(name_b, int_ty()), &mut diag).is_ok());
        assert_eq!(st.lookup(name_b).unwrap().scope_depth, 1);
    }

    #[test]
    fn test_get_undefined_externals_empty() {
        let st = SymbolTable::new();
        assert!(st.get_undefined_externals().is_empty());
    }

    #[test]
    fn test_mark_defined_in_nested_scope() {
        let mut st = SymbolTable::new();
        let mut diag = DiagnosticEmitter::new();

        let name_f = id(0);

        // Declare function at file scope
        assert!(st.insert(name_f, make_func(name_f, int_ty()), &mut diag).is_ok());

        // Push block scope, mark defined from within block scope
        st.push_scope(ScopeKind::Block);
        st.mark_defined(name_f);

        // Pop scope, verify the definition stuck
        st.pop_scope();
        assert!(st.lookup(name_f).unwrap().is_defined);
    }
}
