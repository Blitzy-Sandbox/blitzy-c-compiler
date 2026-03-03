// src/sema/storage.rs — Storage Class Specifier Validation
//
// This module validates C11 storage class specifiers (`static`, `extern`,
// `auto`, `register`, `_Thread_local`), detects conflicting specifier
// combinations, resolves linkage (external, internal, none) per C11 §6.2.2,
// and enforces scope-specific rules per C11 §6.7.1.
//
// # C11 Storage Class Rules
//
// C11 §6.7.1 defines five storage class specifiers. At most one may appear
// in a declaration's specifiers, with the exception that `_Thread_local` may
// combine with `static` or `extern`.
//
// Storage class validity depends on scope:
// - **File scope**: `static` (internal linkage), `extern` (external linkage),
//   or none (external linkage). `auto` and `register` are errors.
// - **Block scope**: all five are valid. `static` creates a local-static,
//   `extern` references an external symbol, `auto`/`register` are automatic,
//   and `_Thread_local` provides thread storage.
// - **Function declarations**: only `static` and `extern` are permitted.
// - **Prototype scope** (parameters): only `register` is permitted.
//
// # Design
//
// The module exports:
// - `Linkage` enum: External, Internal, None
// - `StorageDuration` enum: Static, Thread, Automatic, Allocated
// - `validate_storage_class`: scope-dependent specifier validation
// - `check_conflicting_specifiers`: multi-specifier conflict detection
// - `resolve_linkage`: redeclaration linkage consistency per C11 §6.2.2
// - `check_register_address`: C11 §6.7.1p6 register address restriction
//
// Zero external crate dependencies — only the Rust standard library is used.

use crate::common::diagnostics::DiagnosticEmitter;
use crate::common::source_map::SourceSpan;
use crate::frontend::parser::ast::StorageClass;
use crate::sema::scope::ScopeKind;

// ─────────────────────────────────────────────────────────────────────────────
// Linkage — C11 §6.2.2 "Linkages of identifiers"
// ─────────────────────────────────────────────────────────────────────────────

/// Represents the three linkage categories defined by C11 §6.2.2.
///
/// Linkage determines whether multiple declarations of the same identifier
/// in different scopes or translation units refer to the same entity.
///
/// # C11 §6.2.2 Summary
///
/// - **External linkage**: the identifier refers to the same object or
///   function across all translation units. This is the default for functions
///   and file-scope variable declarations without `static`.
/// - **Internal linkage**: the identifier refers to the same object or
///   function only within the current translation unit. Achieved by declaring
///   with `static` at file scope.
/// - **No linkage**: the identifier refers to a unique entity. This applies
///   to block-scope variables (including `static` locals), function parameters,
///   and any entity not visible outside its declaration scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Linkage {
    /// External linkage: visible across translation units.
    ///
    /// Default for functions and file-scope variables without `static`.
    /// Multiple translation units can reference the same entity.
    External,

    /// Internal linkage: visible only within the current translation unit.
    ///
    /// Declared with `static` at file scope. Each translation unit gets
    /// its own independent copy of the entity.
    Internal,

    /// No linkage: local to the enclosing scope.
    ///
    /// Block-scope variables (including `static` locals), function parameters,
    /// and other entities that are not visible outside their declaration scope.
    None,
}

impl std::fmt::Display for Linkage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Linkage::External => write!(f, "external"),
            Linkage::Internal => write!(f, "internal"),
            Linkage::None => write!(f, "none"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StorageDuration — C11 §6.2.4 "Storage durations of objects"
// ─────────────────────────────────────────────────────────────────────────────

/// Represents the four storage duration categories defined by C11 §6.2.4.
///
/// Storage duration determines the lifetime of an object: when it is created,
/// how long it persists, and when it is destroyed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageDuration {
    /// Static storage duration: exists for the entire program execution.
    ///
    /// File-scope variables, `static` local variables, and string literals
    /// all have static storage duration. They are initialized before `main()`
    /// and persist until program termination.
    Static,

    /// Thread storage duration: exists for the lifetime of the creating thread.
    ///
    /// Objects declared with `_Thread_local` have thread storage duration.
    /// Each thread gets its own independent copy, initialized when the thread
    /// starts and destroyed when the thread exits.
    Thread,

    /// Automatic storage duration: exists during enclosing block execution.
    ///
    /// Default for block-scope variables without `static` or `extern`.
    /// Created when execution enters the enclosing block, destroyed when
    /// execution leaves the block. The `register` specifier is a hint that
    /// does not change semantics beyond prohibiting `&`.
    Automatic,

    /// Allocated storage duration: managed via `malloc`/`free`.
    ///
    /// Not directly associated with any storage class specifier. Included
    /// for completeness in the C11 §6.2.4 model but never produced by
    /// storage class validation functions in this module.
    Allocated,
}

impl std::fmt::Display for StorageDuration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageDuration::Static => write!(f, "static"),
            StorageDuration::Thread => write!(f, "thread"),
            StorageDuration::Automatic => write!(f, "automatic"),
            StorageDuration::Allocated => write!(f, "allocated"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// validate_storage_class — Per-context storage class validation
// ─────────────────────────────────────────────────────────────────────────────

/// Validates a storage class specifier in its declaration context and returns
/// the resulting linkage and storage duration.
///
/// This function implements the scope-dependent validation rules from C11
/// §6.7.1 "Storage-class specifiers":
///
/// - **File scope**: `static` → internal linkage, static duration; `extern` →
///   external linkage, static duration; default → external linkage, static
///   duration; `auto`/`register` → errors.
/// - **Block scope**: `static` → no linkage, static duration; `extern` →
///   external linkage, static duration; `auto`/`register` → no linkage,
///   automatic duration; default → no linkage, automatic duration;
///   `_Thread_local` alone → no linkage, thread duration.
/// - **Function declarations**: only `static` and `extern` allowed;
///   `auto`/`register`/`_Thread_local` → errors.
/// - **Prototype scope** (parameters): only `register` allowed.
///
/// # Arguments
///
/// * `storage_class` — The effective storage class specifier after conflict
///   resolution by [`check_conflicting_specifiers`], or `None` if no storage
///   class was specified.
/// * `scope_kind` — The current lexical scope kind.
/// * `is_function_declaration` — Whether the declaration declares a function.
/// * `diagnostics` — Diagnostic emitter for GCC-compatible error reporting.
/// * `span` — Source span for error location.
///
/// # Returns
///
/// `Ok((linkage, storage_duration))` on success, or `Err(())` if the
/// specifier is invalid in the given context (after emitting a diagnostic).
pub fn validate_storage_class(
    storage_class: Option<StorageClass>,
    scope_kind: ScopeKind,
    is_function_declaration: bool,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) -> Result<(Linkage, StorageDuration), ()> {
    // Function declarations have the most restrictive rules — handle first.
    if is_function_declaration {
        return validate_function_storage(storage_class, scope_kind, diagnostics, span);
    }

    // Parameter declarations only allow `register`.
    if scope_kind == ScopeKind::Prototype {
        return validate_parameter_storage(storage_class, diagnostics, span);
    }

    // Dispatch by scope kind for variable declarations.
    match scope_kind {
        ScopeKind::File => validate_file_scope_storage(storage_class, diagnostics, span),
        ScopeKind::Block | ScopeKind::Function => {
            validate_block_scope_storage(storage_class, diagnostics, span)
        }
        // Already handled above, but exhaustiveness requires this arm.
        ScopeKind::Prototype => validate_parameter_storage(storage_class, diagnostics, span),
    }
}

/// Validates storage class for function declarations per C11 §6.7.1.
///
/// Functions may only have `static` or `extern` storage class specifiers.
/// `auto`, `register`, and `_Thread_local` are not permitted on functions.
fn validate_function_storage(
    storage_class: Option<StorageClass>,
    scope_kind: ScopeKind,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) -> Result<(Linkage, StorageDuration), ()> {
    match storage_class {
        // Default for functions: external linkage, static storage duration.
        // Functions always have static storage duration regardless of scope.
        None => {
            let linkage = if scope_kind == ScopeKind::File {
                Linkage::External
            } else {
                // Block-scope function declarations without a storage class
                // have external linkage by default per C11 §6.2.2p5.
                Linkage::External
            };
            Ok((linkage, StorageDuration::Static))
        }
        Some(StorageClass::Static) => {
            // `static` function: internal linkage, static duration.
            // Valid at file scope. At block scope, the function has internal
            // linkage (visible only within this translation unit).
            Ok((Linkage::Internal, StorageDuration::Static))
        }
        Some(StorageClass::Extern) => {
            // `extern` function: external linkage, static duration.
            Ok((Linkage::External, StorageDuration::Static))
        }
        Some(StorageClass::Auto) => {
            diagnostics.error(
                span.start,
                "function declared with 'auto' storage class",
            );
            Err(())
        }
        Some(StorageClass::Register) => {
            diagnostics.error(
                span.start,
                "function declared with 'register' storage class",
            );
            Err(())
        }
        Some(StorageClass::ThreadLocal) => {
            diagnostics.error(
                span.start,
                "function declared with '_Thread_local' storage class",
            );
            Err(())
        }
    }
}

/// Validates storage class for function parameters per C11 §6.7.1.
///
/// Only `register` is permitted for function parameters. All other storage
/// class specifiers produce an error.
fn validate_parameter_storage(
    storage_class: Option<StorageClass>,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) -> Result<(Linkage, StorageDuration), ()> {
    match storage_class {
        // Parameters have no linkage and automatic storage duration.
        // `register` is a hint; it does not change semantics beyond
        // preventing the address-of operator on the parameter.
        None | Some(StorageClass::Register) => {
            Ok((Linkage::None, StorageDuration::Automatic))
        }
        Some(StorageClass::Static) => {
            diagnostics.error(
                span.start,
                "'static' storage class not allowed in function parameter",
            );
            Err(())
        }
        Some(StorageClass::Extern) => {
            diagnostics.error(
                span.start,
                "'extern' storage class not allowed in function parameter",
            );
            Err(())
        }
        Some(StorageClass::Auto) => {
            diagnostics.error(
                span.start,
                "'auto' storage class not allowed in function parameter",
            );
            Err(())
        }
        Some(StorageClass::ThreadLocal) => {
            diagnostics.error(
                span.start,
                "'_Thread_local' storage class not allowed in function parameter",
            );
            Err(())
        }
    }
}

/// Validates storage class at file scope per C11 §6.7.1.
///
/// At file scope, `static` gives internal linkage, `extern` gives external
/// linkage, and the default is external linkage with static storage duration.
/// `auto` and `register` are errors. `_Thread_local` alone is an error
/// (requires `static` or `extern` companion handled by
/// [`check_conflicting_specifiers`]).
fn validate_file_scope_storage(
    storage_class: Option<StorageClass>,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) -> Result<(Linkage, StorageDuration), ()> {
    match storage_class {
        // Default at file scope: external linkage, static storage duration.
        None => Ok((Linkage::External, StorageDuration::Static)),
        // `static` at file scope: internal linkage, static storage duration.
        Some(StorageClass::Static) => Ok((Linkage::Internal, StorageDuration::Static)),
        // `extern` at file scope: external linkage, static storage duration.
        Some(StorageClass::Extern) => Ok((Linkage::External, StorageDuration::Static)),
        Some(StorageClass::Auto) => {
            diagnostics.error(
                span.start,
                "'auto' storage class not allowed at file scope",
            );
            Err(())
        }
        Some(StorageClass::Register) => {
            diagnostics.error(
                span.start,
                "'register' storage class not allowed at file scope",
            );
            Err(())
        }
        Some(StorageClass::ThreadLocal) => {
            // `_Thread_local` alone at file scope per C11 §6.7.1¶3 is
            // valid — it implies external linkage with thread storage
            // duration. Treat as external linkage, thread-local.
            Ok((Linkage::External, StorageDuration::Thread))
        }
    }
}

/// Validates storage class at block scope per C11 §6.7.1.
///
/// All five storage class specifiers are valid at block scope:
/// - `static`: no linkage, static duration (local static variable).
/// - `extern`: external linkage, static duration (references external symbol).
/// - `auto`: no linkage, automatic duration (explicit default).
/// - `register`: no linkage, automatic duration (register hint).
/// - `_Thread_local` alone: no linkage, thread duration (implicit static-like).
fn validate_block_scope_storage(
    storage_class: Option<StorageClass>,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) -> Result<(Linkage, StorageDuration), ()> {
    match storage_class {
        // Default at block scope: no linkage, automatic storage duration.
        None => Ok((Linkage::None, StorageDuration::Automatic)),
        // `static` at block scope: no linkage, static storage duration.
        // Creates a "local static" variable that persists across calls.
        Some(StorageClass::Static) => Ok((Linkage::None, StorageDuration::Static)),
        // `extern` at block scope: external linkage, static storage duration.
        // References a symbol defined at file scope or in another TU.
        Some(StorageClass::Extern) => Ok((Linkage::External, StorageDuration::Static)),
        // `auto` at block scope: explicit version of the default.
        Some(StorageClass::Auto) => Ok((Linkage::None, StorageDuration::Automatic)),
        // `register` at block scope: no linkage, automatic storage duration.
        // The `&` operator is prohibited on register variables.
        Some(StorageClass::Register) => Ok((Linkage::None, StorageDuration::Automatic)),
        // `_Thread_local` alone at block scope: no linkage, thread duration.
        // Implicitly behaves like `static` but with thread storage duration.
        // Emit a warning because `_Thread_local` without explicit `static`
        // or `extern` is unusual and may indicate a missing companion
        // specifier — GCC-compatible diagnostic behavior.
        Some(StorageClass::ThreadLocal) => {
            diagnostics.warning(
                span.start,
                "'_Thread_local' at block scope without 'static' or 'extern' has implicit static storage behavior",
            );
            Ok((Linkage::None, StorageDuration::Thread))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// check_conflicting_specifiers — Multi-specifier conflict detection
// ─────────────────────────────────────────────────────────────────────────────

/// Validates a list of storage class specifiers for conflicts and returns
/// the single effective specifier.
///
/// C11 §6.7.1 constraint: "At most, one storage-class specifier may be given
/// in the declaration specifiers in a declaration, except that `_Thread_local`
/// may appear with `static` or `extern`."
///
/// # `_Thread_local` Handling
///
/// When `_Thread_local` combines with a companion specifier, this function
/// returns the companion (`Static` or `Extern`) so that
/// [`validate_storage_class`] can determine the correct linkage. The caller
/// should separately detect the presence of `_Thread_local` in the original
/// specifier list and override the storage duration to
/// [`StorageDuration::Thread`] accordingly.
///
/// - `_Thread_local` + `static` → returns `Some(StorageClass::Static)`.
/// - `_Thread_local` + `extern` → returns `Some(StorageClass::Extern)`.
/// - `_Thread_local` + `auto` → error.
/// - `_Thread_local` + `register` → error.
/// - `_Thread_local` alone → returns `Some(StorageClass::ThreadLocal)`.
///   Further scope validation happens in `validate_storage_class`.
///
/// # Arguments
///
/// * `specifiers` — All storage class specifiers in the declaration.
/// * `diagnostics` — Diagnostic emitter for GCC-compatible error reporting.
/// * `span` — Source span for error location.
///
/// # Returns
///
/// `Some(effective)` if a valid single specifier was determined, or `None`
/// if either no specifiers were provided or a conflict was detected (with
/// diagnostic emitted).
pub fn check_conflicting_specifiers(
    specifiers: &[StorageClass],
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) -> Option<StorageClass> {
    if specifiers.is_empty() {
        return None;
    }

    // Single specifier: always valid at the combination level.
    // Scope-level validation happens in `validate_storage_class`.
    if specifiers.len() == 1 {
        return Some(specifiers[0]);
    }

    // Partition specifiers into _Thread_local and non-_Thread_local groups.
    let mut has_thread_local = false;
    let mut companions: Vec<StorageClass> = Vec::new();

    for &spec in specifiers {
        if spec == StorageClass::ThreadLocal {
            if has_thread_local {
                diagnostics.error(
                    span.start,
                    "duplicate '_Thread_local' storage class specifier",
                );
                return None;
            }
            has_thread_local = true;
        } else {
            companions.push(spec);
        }
    }

    if has_thread_local {
        // `_Thread_local` can combine with exactly one of `static`/`extern`.
        if companions.is_empty() {
            // `_Thread_local` alone — scope validation deferred.
            return Some(StorageClass::ThreadLocal);
        }

        if companions.len() > 1 {
            // `_Thread_local` + multiple other specifiers.
            diagnostics.error(
                span.start,
                format!(
                    "cannot combine '{}' with '{}' alongside '_Thread_local'",
                    storage_class_name(companions[0]),
                    storage_class_name(companions[1]),
                ),
            );
            return None;
        }

        // `_Thread_local` + exactly one companion.
        let companion = companions[0];
        match companion {
            StorageClass::Static => Some(StorageClass::Static),
            StorageClass::Extern => Some(StorageClass::Extern),
            StorageClass::Auto => {
                diagnostics.error(
                    span.start,
                    "cannot combine '_Thread_local' with 'auto'",
                );
                None
            }
            StorageClass::Register => {
                diagnostics.error(
                    span.start,
                    "cannot combine '_Thread_local' with 'register'",
                );
                None
            }
            StorageClass::ThreadLocal => {
                // Duplicate: already caught above.
                diagnostics.error(
                    span.start,
                    "duplicate '_Thread_local' storage class specifier",
                );
                None
            }
        }
    } else {
        // No `_Thread_local` — at most one specifier is allowed.
        if companions.len() > 1 {
            diagnostics.error(
                span.start,
                format!(
                    "cannot combine '{}' with '{}'",
                    storage_class_name(companions[0]),
                    storage_class_name(companions[1]),
                ),
            );
            return None;
        }

        // Exactly one non-_Thread_local specifier.
        Some(companions[0])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// resolve_linkage — Redeclaration linkage consistency (C11 §6.2.2)
// ─────────────────────────────────────────────────────────────────────────────

/// Resolves linkage for a redeclaration based on the prior declaration's
/// linkage and the new declaration's storage class.
///
/// Implements C11 §6.2.2 linkage resolution rules for identifiers that are
/// declared more than once within the same or nested scopes:
///
/// - If the prior declaration has **external** linkage and the new declaration
///   uses `extern`, the result is external linkage.
/// - If the prior declaration has **internal** linkage and the new declaration
///   uses `extern` at **block scope**, the result inherits the internal linkage
///   from the prior file-scope `static` declaration (C11 §6.2.2p4).
/// - If the prior declaration has **internal** linkage (`static`) and the new
///   declaration at **file scope** lacks `static`, this is a linkage conflict
///   error: "non-static declaration follows static declaration."
/// - If the prior declaration has **external** linkage and the new declaration
///   uses `static` at **file scope**, this is a linkage conflict error:
///   "static declaration follows non-static declaration."
/// - If there is no prior declaration, linkage is determined by the storage
///   class and scope alone.
///
/// # Arguments
///
/// * `new_storage` — Storage class of the new declaration.
/// * `previous_linkage` — Linkage of the prior declaration, or `None` if
///   this is the first declaration of the identifier.
/// * `scope_kind` — Current scope kind.
/// * `diagnostics` — Diagnostic emitter.
/// * `span` — Source span for error location.
///
/// # Returns
///
/// The resolved linkage for the new declaration.
pub fn resolve_linkage(
    new_storage: Option<StorageClass>,
    previous_linkage: Option<Linkage>,
    scope_kind: ScopeKind,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) -> Linkage {
    match previous_linkage {
        // ── First declaration: determine linkage from storage class + scope ──
        Option::None => match new_storage {
            Some(StorageClass::Static) => {
                if scope_kind == ScopeKind::File {
                    Linkage::Internal
                } else {
                    // `static` at block scope → no linkage (local static).
                    Linkage::None
                }
            }
            Some(StorageClass::Extern) => Linkage::External,
            Option::None => {
                if scope_kind == ScopeKind::File {
                    Linkage::External
                } else {
                    Linkage::None
                }
            }
            // `auto`, `register`, `_Thread_local` → no linkage.
            Some(_) => Linkage::None,
        },

        // ── Redeclaration: resolve linkage consistency per C11 §6.2.2 ──
        Some(prev) => resolve_with_prior(prev, new_storage, scope_kind, diagnostics, span),
    }
}

/// Internal helper: resolves linkage when a prior declaration exists.
///
/// Separated from `resolve_linkage` for clarity and to keep the match arms
/// manageable.
fn resolve_with_prior(
    prev: Linkage,
    new_storage: Option<StorageClass>,
    scope_kind: ScopeKind,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) -> Linkage {
    match (prev, new_storage) {
        // ── Prior external linkage ──

        // Prior external + new `extern` → stays external.
        (Linkage::External, Some(StorageClass::Extern)) => Linkage::External,

        // Prior external + no storage at file scope → stays external
        // (tentative definition / redeclaration without storage class).
        (Linkage::External, Option::None) if scope_kind == ScopeKind::File => Linkage::External,

        // Prior external + new `static` at file scope → CONFLICT.
        (Linkage::External, Some(StorageClass::Static)) if scope_kind == ScopeKind::File => {
            diagnostics.error(
                span.start,
                "static declaration follows non-static declaration",
            );
            // Return Internal as best-effort despite the error.
            Linkage::Internal
        }

        // Prior external + new `static` at non-file scope → new entity,
        // no linkage (block-scope `static` creates a separate local static).
        (Linkage::External, Some(StorageClass::Static)) => Linkage::None,

        // ── Prior internal linkage ──

        // Prior internal + new `extern` → depends on scope.
        (Linkage::Internal, Some(StorageClass::Extern)) => {
            if scope_kind == ScopeKind::File {
                // File scope: `extern` after `static` → CONFLICT.
                diagnostics.error(
                    span.start,
                    "non-static declaration follows static declaration",
                );
                Linkage::Internal
            } else {
                // Block scope: `extern` inherits internal linkage from
                // the prior file-scope `static` declaration (C11 §6.2.2p4).
                Linkage::Internal
            }
        }

        // Prior internal + no storage at file scope → CONFLICT
        // (implicit external linkage conflicts with prior `static`).
        (Linkage::Internal, Option::None) if scope_kind == ScopeKind::File => {
            diagnostics.error(
                span.start,
                "non-static declaration follows static declaration",
            );
            Linkage::Internal
        }

        // Prior internal + new `static` → consistent (stays internal).
        (Linkage::Internal, Some(StorageClass::Static)) => Linkage::Internal,

        // ── Fallback ──

        // Prior external + no storage at non-file scope → external.
        (Linkage::External, Option::None) => Linkage::External,

        // Any other combination: preserve prior linkage.
        (prior, _) => prior,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// check_register_address — C11 §6.7.1p6 register address restriction
// ─────────────────────────────────────────────────────────────────────────────

/// Checks whether the address-of operator (`&`) was applied to a variable
/// declared with the `register` storage class.
///
/// C11 §6.7.1 constraint 6: "The operand of the unary `&` operator shall
/// be either a function designator, the result of a `[]` or unary `*`
/// operator, or an lvalue that designates an object that is not a bit-field
/// and is not declared with the `register` storage-class specifier."
///
/// If the address was taken on a register variable, an error diagnostic is
/// emitted.
///
/// # Arguments
///
/// * `storage` — The storage class of the variable, or `None`.
/// * `has_address_taken` — Whether the address-of (`&`) operator was applied.
/// * `diagnostics` — Diagnostic emitter.
/// * `span` — Source span for error location.
pub fn check_register_address(
    storage: Option<StorageClass>,
    has_address_taken: bool,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) {
    if has_address_taken {
        if let Some(StorageClass::Register) = storage {
            diagnostics.error(
                span.start,
                "address of register variable requested",
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// validate_struct_member_storage — Struct/union member validation
// ─────────────────────────────────────────────────────────────────────────────

/// Validates that a struct or union member does not carry a storage class
/// specifier.
///
/// C11 §6.7.2.1: Struct and union members cannot have storage class specifiers.
/// For example, `struct S { static int x; }` is a constraint violation.
///
/// Enum constants are not variables and do not have storage classes, so this
/// function is not applicable to enum declarations.
///
/// # Arguments
///
/// * `storage_class` — The storage class on the member declaration, if any.
/// * `diagnostics` — Diagnostic emitter.
/// * `span` — Source span for error location.
///
/// # Returns
///
/// `true` if valid (no storage class), `false` if invalid (diagnostic emitted).
pub(crate) fn validate_struct_member_storage(
    storage_class: Option<StorageClass>,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) -> bool {
    if let Some(sc) = storage_class {
        diagnostics.error(
            span.start,
            format!(
                "storage class specifier '{}' not allowed in struct/union member",
                storage_class_name(sc),
            ),
        );
        false
    } else {
        true
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions
// ─────────────────────────────────────────────────────────────────────────────

/// Returns the human-readable name of a storage class specifier for use in
/// diagnostic messages.
fn storage_class_name(sc: StorageClass) -> &'static str {
    match sc {
        StorageClass::Static => "static",
        StorageClass::Extern => "extern",
        StorageClass::Auto => "auto",
        StorageClass::Register => "register",
        StorageClass::ThreadLocal => "_Thread_local",
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::source_map::SourceSpan;

    // ── Test Helpers ─────────────────────────────────────────────────────

    /// Creates a dummy source span for test assertions.
    fn dummy_span() -> SourceSpan {
        SourceSpan::dummy()
    }

    /// Creates a fresh diagnostic emitter for each test.
    fn emitter() -> DiagnosticEmitter {
        DiagnosticEmitter::new()
    }

    // ── File Scope: Variable Declarations ────────────────────────────────

    #[test]
    fn file_scope_no_storage_class_gives_external_static() {
        let mut diag = emitter();
        let result = validate_storage_class(
            None, ScopeKind::File, false, &mut diag, dummy_span(),
        );
        assert_eq!(result, Ok((Linkage::External, StorageDuration::Static)));
        assert!(!diag.has_errors());
    }

    #[test]
    fn file_scope_static_gives_internal_static() {
        let mut diag = emitter();
        let result = validate_storage_class(
            Some(StorageClass::Static), ScopeKind::File, false,
            &mut diag, dummy_span(),
        );
        assert_eq!(result, Ok((Linkage::Internal, StorageDuration::Static)));
        assert!(!diag.has_errors());
    }

    #[test]
    fn file_scope_extern_gives_external_static() {
        let mut diag = emitter();
        let result = validate_storage_class(
            Some(StorageClass::Extern), ScopeKind::File, false,
            &mut diag, dummy_span(),
        );
        assert_eq!(result, Ok((Linkage::External, StorageDuration::Static)));
        assert!(!diag.has_errors());
    }

    #[test]
    fn file_scope_auto_is_error() {
        let mut diag = emitter();
        let result = validate_storage_class(
            Some(StorageClass::Auto), ScopeKind::File, false,
            &mut diag, dummy_span(),
        );
        assert!(result.is_err());
        assert!(diag.has_errors());
        assert_eq!(diag.error_count(), 1);
    }

    #[test]
    fn file_scope_register_is_error() {
        let mut diag = emitter();
        let result = validate_storage_class(
            Some(StorageClass::Register), ScopeKind::File, false,
            &mut diag, dummy_span(),
        );
        assert!(result.is_err());
        assert!(diag.has_errors());
        assert_eq!(diag.error_count(), 1);
    }

    #[test]
    fn file_scope_thread_local_alone_is_valid() {
        // Per C11 §6.7.1¶3, `_Thread_local` alone at file scope is valid
        // and implies external linkage with thread storage duration.
        let mut diag = emitter();
        let result = validate_storage_class(
            Some(StorageClass::ThreadLocal), ScopeKind::File, false,
            &mut diag, dummy_span(),
        );
        assert_eq!(result, Ok((Linkage::External, StorageDuration::Thread)));
        assert!(!diag.has_errors());
    }

    // ── Block Scope: Variable Declarations ───────────────────────────────

    #[test]
    fn block_scope_no_storage_class_gives_none_automatic() {
        let mut diag = emitter();
        let result = validate_storage_class(
            None, ScopeKind::Block, false, &mut diag, dummy_span(),
        );
        assert_eq!(result, Ok((Linkage::None, StorageDuration::Automatic)));
        assert!(!diag.has_errors());
    }

    #[test]
    fn block_scope_static_gives_none_static() {
        let mut diag = emitter();
        let result = validate_storage_class(
            Some(StorageClass::Static), ScopeKind::Block, false,
            &mut diag, dummy_span(),
        );
        assert_eq!(result, Ok((Linkage::None, StorageDuration::Static)));
        assert!(!diag.has_errors());
    }

    #[test]
    fn block_scope_extern_gives_external_static() {
        let mut diag = emitter();
        let result = validate_storage_class(
            Some(StorageClass::Extern), ScopeKind::Block, false,
            &mut diag, dummy_span(),
        );
        assert_eq!(result, Ok((Linkage::External, StorageDuration::Static)));
        assert!(!diag.has_errors());
    }

    #[test]
    fn block_scope_auto_gives_none_automatic() {
        let mut diag = emitter();
        let result = validate_storage_class(
            Some(StorageClass::Auto), ScopeKind::Block, false,
            &mut diag, dummy_span(),
        );
        assert_eq!(result, Ok((Linkage::None, StorageDuration::Automatic)));
        assert!(!diag.has_errors());
    }

    #[test]
    fn block_scope_register_gives_none_automatic() {
        let mut diag = emitter();
        let result = validate_storage_class(
            Some(StorageClass::Register), ScopeKind::Block, false,
            &mut diag, dummy_span(),
        );
        assert_eq!(result, Ok((Linkage::None, StorageDuration::Automatic)));
        assert!(!diag.has_errors());
    }

    #[test]
    fn block_scope_thread_local_alone_gives_none_thread() {
        let mut diag = emitter();
        let result = validate_storage_class(
            Some(StorageClass::ThreadLocal), ScopeKind::Block, false,
            &mut diag, dummy_span(),
        );
        assert_eq!(result, Ok((Linkage::None, StorageDuration::Thread)));
        assert!(!diag.has_errors());
    }

    // ── Conflicting Specifier Detection ──────────────────────────────────

    #[test]
    fn conflict_empty_specifiers_returns_none() {
        let mut diag = emitter();
        let result = check_conflicting_specifiers(&[], &mut diag, dummy_span());
        assert_eq!(result, None);
        assert!(!diag.has_errors());
    }

    #[test]
    fn conflict_single_specifier_returns_it() {
        let mut diag = emitter();
        let result = check_conflicting_specifiers(
            &[StorageClass::Static], &mut diag, dummy_span(),
        );
        assert_eq!(result, Some(StorageClass::Static));
        assert!(!diag.has_errors());
    }

    #[test]
    fn conflict_static_extern_is_error() {
        let mut diag = emitter();
        let result = check_conflicting_specifiers(
            &[StorageClass::Static, StorageClass::Extern],
            &mut diag, dummy_span(),
        );
        assert!(result.is_none());
        assert!(diag.has_errors());
    }

    #[test]
    fn conflict_auto_register_is_error() {
        let mut diag = emitter();
        let result = check_conflicting_specifiers(
            &[StorageClass::Auto, StorageClass::Register],
            &mut diag, dummy_span(),
        );
        assert!(result.is_none());
        assert!(diag.has_errors());
    }

    #[test]
    fn conflict_extern_static_is_error() {
        let mut diag = emitter();
        let result = check_conflicting_specifiers(
            &[StorageClass::Extern, StorageClass::Static],
            &mut diag, dummy_span(),
        );
        assert!(result.is_none());
        assert!(diag.has_errors());
    }

    #[test]
    fn thread_local_static_is_ok() {
        let mut diag = emitter();
        let result = check_conflicting_specifiers(
            &[StorageClass::ThreadLocal, StorageClass::Static],
            &mut diag, dummy_span(),
        );
        assert_eq!(result, Some(StorageClass::Static));
        assert!(!diag.has_errors());
    }

    #[test]
    fn thread_local_extern_is_ok() {
        let mut diag = emitter();
        let result = check_conflicting_specifiers(
            &[StorageClass::ThreadLocal, StorageClass::Extern],
            &mut diag, dummy_span(),
        );
        assert_eq!(result, Some(StorageClass::Extern));
        assert!(!diag.has_errors());
    }

    #[test]
    fn thread_local_auto_is_error() {
        let mut diag = emitter();
        let result = check_conflicting_specifiers(
            &[StorageClass::ThreadLocal, StorageClass::Auto],
            &mut diag, dummy_span(),
        );
        assert!(result.is_none());
        assert!(diag.has_errors());
    }

    #[test]
    fn thread_local_register_is_error() {
        let mut diag = emitter();
        let result = check_conflicting_specifiers(
            &[StorageClass::ThreadLocal, StorageClass::Register],
            &mut diag, dummy_span(),
        );
        assert!(result.is_none());
        assert!(diag.has_errors());
    }

    #[test]
    fn thread_local_alone_returns_thread_local() {
        let mut diag = emitter();
        let result = check_conflicting_specifiers(
            &[StorageClass::ThreadLocal], &mut diag, dummy_span(),
        );
        assert_eq!(result, Some(StorageClass::ThreadLocal));
        assert!(!diag.has_errors());
    }

    #[test]
    fn duplicate_thread_local_is_error() {
        let mut diag = emitter();
        let result = check_conflicting_specifiers(
            &[StorageClass::ThreadLocal, StorageClass::ThreadLocal],
            &mut diag, dummy_span(),
        );
        assert!(result.is_none());
        assert!(diag.has_errors());
    }

    // ── Function-Specific Validation ─────────────────────────────────────

    #[test]
    fn function_default_gives_external_static() {
        let mut diag = emitter();
        let result = validate_storage_class(
            None, ScopeKind::File, true, &mut diag, dummy_span(),
        );
        assert_eq!(result, Ok((Linkage::External, StorageDuration::Static)));
        assert!(!diag.has_errors());
    }

    #[test]
    fn function_static_gives_internal_static() {
        let mut diag = emitter();
        let result = validate_storage_class(
            Some(StorageClass::Static), ScopeKind::File, true,
            &mut diag, dummy_span(),
        );
        assert_eq!(result, Ok((Linkage::Internal, StorageDuration::Static)));
        assert!(!diag.has_errors());
    }

    #[test]
    fn function_extern_gives_external_static() {
        let mut diag = emitter();
        let result = validate_storage_class(
            Some(StorageClass::Extern), ScopeKind::File, true,
            &mut diag, dummy_span(),
        );
        assert_eq!(result, Ok((Linkage::External, StorageDuration::Static)));
        assert!(!diag.has_errors());
    }

    #[test]
    fn function_auto_is_error() {
        let mut diag = emitter();
        let result = validate_storage_class(
            Some(StorageClass::Auto), ScopeKind::File, true,
            &mut diag, dummy_span(),
        );
        assert!(result.is_err());
        assert!(diag.has_errors());
    }

    #[test]
    fn function_register_is_error() {
        let mut diag = emitter();
        let result = validate_storage_class(
            Some(StorageClass::Register), ScopeKind::File, true,
            &mut diag, dummy_span(),
        );
        assert!(result.is_err());
        assert!(diag.has_errors());
    }

    #[test]
    fn function_thread_local_is_error() {
        let mut diag = emitter();
        let result = validate_storage_class(
            Some(StorageClass::ThreadLocal), ScopeKind::File, true,
            &mut diag, dummy_span(),
        );
        assert!(result.is_err());
        assert!(diag.has_errors());
    }

    // ── Parameter-Specific Validation ────────────────────────────────────

    #[test]
    fn param_default_gives_none_automatic() {
        let mut diag = emitter();
        let result = validate_storage_class(
            None, ScopeKind::Prototype, false, &mut diag, dummy_span(),
        );
        assert_eq!(result, Ok((Linkage::None, StorageDuration::Automatic)));
        assert!(!diag.has_errors());
    }

    #[test]
    fn param_register_gives_none_automatic() {
        let mut diag = emitter();
        let result = validate_storage_class(
            Some(StorageClass::Register), ScopeKind::Prototype, false,
            &mut diag, dummy_span(),
        );
        assert_eq!(result, Ok((Linkage::None, StorageDuration::Automatic)));
        assert!(!diag.has_errors());
    }

    #[test]
    fn param_static_is_error() {
        let mut diag = emitter();
        let result = validate_storage_class(
            Some(StorageClass::Static), ScopeKind::Prototype, false,
            &mut diag, dummy_span(),
        );
        assert!(result.is_err());
        assert!(diag.has_errors());
    }

    #[test]
    fn param_extern_is_error() {
        let mut diag = emitter();
        let result = validate_storage_class(
            Some(StorageClass::Extern), ScopeKind::Prototype, false,
            &mut diag, dummy_span(),
        );
        assert!(result.is_err());
        assert!(diag.has_errors());
    }

    #[test]
    fn param_auto_is_error() {
        let mut diag = emitter();
        let result = validate_storage_class(
            Some(StorageClass::Auto), ScopeKind::Prototype, false,
            &mut diag, dummy_span(),
        );
        assert!(result.is_err());
        assert!(diag.has_errors());
    }

    #[test]
    fn param_thread_local_is_error() {
        let mut diag = emitter();
        let result = validate_storage_class(
            Some(StorageClass::ThreadLocal), ScopeKind::Prototype, false,
            &mut diag, dummy_span(),
        );
        assert!(result.is_err());
        assert!(diag.has_errors());
    }

    // ── Linkage Resolution ───────────────────────────────────────────────

    #[test]
    fn linkage_first_declaration_static_file_gives_internal() {
        let mut diag = emitter();
        let result = resolve_linkage(
            Some(StorageClass::Static), None,
            ScopeKind::File, &mut diag, dummy_span(),
        );
        assert_eq!(result, Linkage::Internal);
        assert!(!diag.has_errors());
    }

    #[test]
    fn linkage_first_declaration_extern_gives_external() {
        let mut diag = emitter();
        let result = resolve_linkage(
            Some(StorageClass::Extern), None,
            ScopeKind::File, &mut diag, dummy_span(),
        );
        assert_eq!(result, Linkage::External);
        assert!(!diag.has_errors());
    }

    #[test]
    fn linkage_first_declaration_default_file_gives_external() {
        let mut diag = emitter();
        let result = resolve_linkage(
            None, None, ScopeKind::File, &mut diag, dummy_span(),
        );
        assert_eq!(result, Linkage::External);
        assert!(!diag.has_errors());
    }

    #[test]
    fn linkage_first_declaration_default_block_gives_none() {
        let mut diag = emitter();
        let result = resolve_linkage(
            None, None, ScopeKind::Block, &mut diag, dummy_span(),
        );
        assert_eq!(result, Linkage::None);
        assert!(!diag.has_errors());
    }

    #[test]
    fn linkage_extern_after_extern_stays_external() {
        let mut diag = emitter();
        let result = resolve_linkage(
            Some(StorageClass::Extern), Some(Linkage::External),
            ScopeKind::File, &mut diag, dummy_span(),
        );
        assert_eq!(result, Linkage::External);
        assert!(!diag.has_errors());
    }

    #[test]
    fn linkage_static_after_static_stays_internal() {
        let mut diag = emitter();
        let result = resolve_linkage(
            Some(StorageClass::Static), Some(Linkage::Internal),
            ScopeKind::File, &mut diag, dummy_span(),
        );
        assert_eq!(result, Linkage::Internal);
        assert!(!diag.has_errors());
    }

    #[test]
    fn linkage_static_after_extern_file_is_conflict() {
        let mut diag = emitter();
        let result = resolve_linkage(
            Some(StorageClass::Static), Some(Linkage::External),
            ScopeKind::File, &mut diag, dummy_span(),
        );
        assert!(diag.has_errors());
        // Best-effort: returns Internal despite the error.
        assert_eq!(result, Linkage::Internal);
    }

    #[test]
    fn linkage_extern_after_static_block_inherits_internal() {
        let mut diag = emitter();
        // First `static` at file scope → internal linkage.
        // Then `extern` at block scope → inherits internal.
        let result = resolve_linkage(
            Some(StorageClass::Extern), Some(Linkage::Internal),
            ScopeKind::Block, &mut diag, dummy_span(),
        );
        assert_eq!(result, Linkage::Internal);
        assert!(!diag.has_errors());
    }

    #[test]
    fn linkage_extern_after_static_file_is_conflict() {
        let mut diag = emitter();
        let result = resolve_linkage(
            Some(StorageClass::Extern), Some(Linkage::Internal),
            ScopeKind::File, &mut diag, dummy_span(),
        );
        assert!(diag.has_errors());
        assert_eq!(result, Linkage::Internal);
    }

    #[test]
    fn linkage_no_storage_after_static_file_is_conflict() {
        let mut diag = emitter();
        // Prior static (internal), then no storage at file scope (implicit external).
        let result = resolve_linkage(
            None, Some(Linkage::Internal),
            ScopeKind::File, &mut diag, dummy_span(),
        );
        assert!(diag.has_errors());
        assert_eq!(result, Linkage::Internal);
    }

    #[test]
    fn linkage_no_storage_after_external_file_stays_external() {
        let mut diag = emitter();
        let result = resolve_linkage(
            None, Some(Linkage::External),
            ScopeKind::File, &mut diag, dummy_span(),
        );
        assert_eq!(result, Linkage::External);
        assert!(!diag.has_errors());
    }

    // ── Register Address Restriction ─────────────────────────────────────

    #[test]
    fn register_with_address_taken_is_error() {
        let mut diag = emitter();
        check_register_address(
            Some(StorageClass::Register), true, &mut diag, dummy_span(),
        );
        assert!(diag.has_errors());
        assert_eq!(diag.error_count(), 1);
    }

    #[test]
    fn register_without_address_is_ok() {
        let mut diag = emitter();
        check_register_address(
            Some(StorageClass::Register), false, &mut diag, dummy_span(),
        );
        assert!(!diag.has_errors());
    }

    #[test]
    fn non_register_with_address_is_ok() {
        let mut diag = emitter();
        check_register_address(
            Some(StorageClass::Auto), true, &mut diag, dummy_span(),
        );
        assert!(!diag.has_errors());
    }

    #[test]
    fn no_storage_with_address_is_ok() {
        let mut diag = emitter();
        check_register_address(None, true, &mut diag, dummy_span());
        assert!(!diag.has_errors());
    }

    #[test]
    fn static_with_address_is_ok() {
        let mut diag = emitter();
        check_register_address(
            Some(StorageClass::Static), true, &mut diag, dummy_span(),
        );
        assert!(!diag.has_errors());
    }

    #[test]
    fn extern_with_address_is_ok() {
        let mut diag = emitter();
        check_register_address(
            Some(StorageClass::Extern), true, &mut diag, dummy_span(),
        );
        assert!(!diag.has_errors());
    }

    // ── Struct/Union Member Storage ──────────────────────────────────────

    #[test]
    fn struct_member_no_storage_is_ok() {
        let mut diag = emitter();
        assert!(validate_struct_member_storage(None, &mut diag, dummy_span()));
        assert!(!diag.has_errors());
    }

    #[test]
    fn struct_member_static_is_error() {
        let mut diag = emitter();
        assert!(!validate_struct_member_storage(
            Some(StorageClass::Static), &mut diag, dummy_span(),
        ));
        assert!(diag.has_errors());
    }

    #[test]
    fn struct_member_extern_is_error() {
        let mut diag = emitter();
        assert!(!validate_struct_member_storage(
            Some(StorageClass::Extern), &mut diag, dummy_span(),
        ));
        assert!(diag.has_errors());
    }

    #[test]
    fn struct_member_register_is_error() {
        let mut diag = emitter();
        assert!(!validate_struct_member_storage(
            Some(StorageClass::Register), &mut diag, dummy_span(),
        ));
        assert!(diag.has_errors());
    }

    // ── Edge Cases ───────────────────────────────────────────────────────

    #[test]
    fn function_scope_block_scope_default_automatic() {
        // Function scope (label scope) treated like block scope for variables.
        let mut diag = emitter();
        let result = validate_storage_class(
            None, ScopeKind::Function, false, &mut diag, dummy_span(),
        );
        assert_eq!(result, Ok((Linkage::None, StorageDuration::Automatic)));
        assert!(!diag.has_errors());
    }

    #[test]
    fn function_at_block_scope_default_external() {
        // A function declared inside a block (block-scope function declaration).
        let mut diag = emitter();
        let result = validate_storage_class(
            None, ScopeKind::Block, true, &mut diag, dummy_span(),
        );
        assert_eq!(result, Ok((Linkage::External, StorageDuration::Static)));
        assert!(!diag.has_errors());
    }

    #[test]
    fn first_declaration_auto_block() {
        let mut diag = emitter();
        let result = resolve_linkage(
            Some(StorageClass::Auto), None,
            ScopeKind::Block, &mut diag, dummy_span(),
        );
        assert_eq!(result, Linkage::None);
        assert!(!diag.has_errors());
    }

    #[test]
    fn first_declaration_register_block() {
        let mut diag = emitter();
        let result = resolve_linkage(
            Some(StorageClass::Register), None,
            ScopeKind::Block, &mut diag, dummy_span(),
        );
        assert_eq!(result, Linkage::None);
        assert!(!diag.has_errors());
    }

    #[test]
    fn static_at_block_scope_no_linkage() {
        let mut diag = emitter();
        let result = resolve_linkage(
            Some(StorageClass::Static), None,
            ScopeKind::Block, &mut diag, dummy_span(),
        );
        assert_eq!(result, Linkage::None);
        assert!(!diag.has_errors());
    }

    #[test]
    fn thread_local_static_combination_flow() {
        // Simulate the full flow: check_conflicting_specifiers → validate_storage_class
        // for `_Thread_local static` at file scope.
        let mut diag = emitter();

        // Step 1: check_conflicting_specifiers returns the companion (Static).
        let effective = check_conflicting_specifiers(
            &[StorageClass::ThreadLocal, StorageClass::Static],
            &mut diag, dummy_span(),
        );
        assert_eq!(effective, Some(StorageClass::Static));
        assert!(!diag.has_errors());

        // Step 2: validate_storage_class with Static at file scope.
        let result = validate_storage_class(
            effective, ScopeKind::File, false, &mut diag, dummy_span(),
        );
        // Returns internal linkage, static duration. The caller would then
        // override duration to Thread because _Thread_local was present.
        assert_eq!(result, Ok((Linkage::Internal, StorageDuration::Static)));
        assert!(!diag.has_errors());
    }

    #[test]
    fn thread_local_extern_combination_flow() {
        // Simulate `_Thread_local extern` at file scope.
        let mut diag = emitter();

        let effective = check_conflicting_specifiers(
            &[StorageClass::ThreadLocal, StorageClass::Extern],
            &mut diag, dummy_span(),
        );
        assert_eq!(effective, Some(StorageClass::Extern));

        let result = validate_storage_class(
            effective, ScopeKind::File, false, &mut diag, dummy_span(),
        );
        assert_eq!(result, Ok((Linkage::External, StorageDuration::Static)));
        assert!(!diag.has_errors());
    }

    #[test]
    fn conflicting_specifiers_reverse_order() {
        // Ensure order doesn't matter for ThreadLocal combinations.
        let mut diag = emitter();
        let result = check_conflicting_specifiers(
            &[StorageClass::Static, StorageClass::ThreadLocal],
            &mut diag, dummy_span(),
        );
        assert_eq!(result, Some(StorageClass::Static));
        assert!(!diag.has_errors());
    }

    #[test]
    fn three_specifiers_no_thread_local_error() {
        let mut diag = emitter();
        let result = check_conflicting_specifiers(
            &[StorageClass::Static, StorageClass::Extern, StorageClass::Auto],
            &mut diag, dummy_span(),
        );
        assert!(result.is_none());
        assert!(diag.has_errors());
    }

    #[test]
    fn linkage_external_static_non_file_scope() {
        // Prior external, new `static` at block scope → new entity (no linkage).
        let mut diag = emitter();
        let result = resolve_linkage(
            Some(StorageClass::Static), Some(Linkage::External),
            ScopeKind::Block, &mut diag, dummy_span(),
        );
        assert_eq!(result, Linkage::None);
        assert!(!diag.has_errors());
    }

    #[test]
    fn linkage_none_prior_any_new() {
        // Prior no-linkage, any new storage → preserves prior.
        let mut diag = emitter();
        let result = resolve_linkage(
            Some(StorageClass::Static), Some(Linkage::None),
            ScopeKind::Block, &mut diag, dummy_span(),
        );
        assert_eq!(result, Linkage::None);
        assert!(!diag.has_errors());
    }
}
