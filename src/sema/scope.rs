// src/sema/scope.rs — Lexical Scope Management
//
// This module implements the lexical scope stack for the bcc semantic analyzer,
// following C11 §6.2.1 "Scopes of identifiers" precisely:
//
// - **File scope**: From declaration point to end of translation unit. This is
//   the outermost scope, always present at the bottom of the scope stack.
//
// - **Function scope**: Used exclusively for labels (goto targets). Labels are
//   visible throughout the entire function body regardless of block nesting.
//   Per C11, the scope of a label is the entire function in which it appears.
//
// - **Block scope**: Enclosed by `{ }`. Identifiers are visible from their
//   declaration point to the closing `}`. Includes function bodies, if/else
//   branches, for/while/do-while loop bodies, and switch statement bodies.
//
// - **Function prototype scope**: Parameter names in a function declaration
//   (not definition). These names are only visible within the prototype itself.
//   Example: `void f(int x, int y);` — x and y have prototype scope.
//
// The scope stack supports variable shadowing (inner scope declarations shadow
// outer scope), tracks scope entry/exit for proper symbol lifetime management,
// and provides label scope helpers for goto target resolution.
//
// Zero external crate dependencies — only the Rust standard library is used.

// ─────────────────────────────────────────────────────────────────────────────
// ScopeKind — Classifies the four kinds of C11 lexical scopes
// ─────────────────────────────────────────────────────────────────────────────

/// Represents the kind of a lexical scope in C11.
///
/// C11 §6.2.1 defines four scope categories for identifiers. Each scope kind
/// has distinct rules for identifier visibility, lifetime, and namespace behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScopeKind {
    /// File scope: top-level declarations, visible throughout the translation unit.
    ///
    /// This is the outermost scope. Identifiers declared here have file scope
    /// and are visible from the point of declaration to the end of the
    /// translation unit. File scope is always present at the bottom of the
    /// scope stack and can never be popped.
    File,

    /// Function scope: used ONLY for labels (goto targets).
    ///
    /// Labels are visible throughout the entire function body regardless of
    /// block nesting. This means `goto forward_label;` at the beginning of a
    /// function can jump to `forward_label:` at the end. The label namespace
    /// is per-function, not per-block.
    ///
    /// A Function scope is pushed when entering a function definition and
    /// popped when exiting. It sits between file scope and the function body's
    /// block scope in the stack.
    Function,

    /// Block scope: enclosed by `{ }`, includes function bodies, if/else
    /// branches, for/while/do-while loop bodies, and switch bodies.
    ///
    /// Identifiers declared in a block are visible from the declaration point
    /// to the closing `}`. Inner block scopes shadow outer scopes — a variable
    /// declared in an inner block with the same name as one in an outer block
    /// hides the outer declaration within the inner block.
    Block,

    /// Function prototype scope: parameter names in a function declaration
    /// (not a function definition).
    ///
    /// These names are only visible within the prototype itself and do not
    /// carry over to the function body or any other context.
    ///
    /// Example: `void f(int x, int y);` — `x` and `y` have prototype scope
    /// and are discarded after the closing `)`.
    Prototype,
}

impl std::fmt::Display for ScopeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScopeKind::File => write!(f, "file"),
            ScopeKind::Function => write!(f, "function"),
            ScopeKind::Block => write!(f, "block"),
            ScopeKind::Prototype => write!(f, "prototype"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scope — Represents a single scope level in the scope stack
// ─────────────────────────────────────────────────────────────────────────────

/// Represents a single scope level in the scope stack.
///
/// Each scope tracks its kind, nesting depth, symbol count, and unreachable
/// code status. The `depth` field corresponds to the scope's position in the
/// stack (0 = file scope, incrementing inward). The `symbol_count` tracks how
/// many symbols have been declared within this scope. The `has_unreachable`
/// flag is set after unconditional control flow transfers (`return`, `break`,
/// `continue`, unconditional `goto`) to enable unreachable code warnings.
#[derive(Debug, Clone)]
pub struct Scope {
    /// The kind of this scope (File, Function, Block, or Prototype).
    pub kind: ScopeKind,

    /// Nesting depth of this scope (0 = file scope, incrementing inward).
    ///
    /// The depth corresponds to the index in the scope stack vector. File scope
    /// is always at depth 0. Each nested scope increments the depth by 1.
    pub depth: usize,

    /// Number of symbols declared in this scope.
    ///
    /// Incremented by the semantic analyzer each time a new symbol is registered
    /// in this scope. Used for tracking and diagnostics.
    pub symbol_count: usize,

    /// Whether code in this scope has become unreachable.
    ///
    /// Set to `true` after encountering `return`, `break`, `continue`, or an
    /// unconditional `goto`. Subsequent code in this scope (before a new label
    /// or scope entry) is unreachable and should trigger a diagnostic warning.
    pub has_unreachable: bool,
}

impl Scope {
    /// Creates a new scope with the given kind and depth.
    ///
    /// The scope starts with zero symbols and reachable code.
    fn new(kind: ScopeKind, depth: usize) -> Self {
        Scope {
            kind,
            depth,
            symbol_count: 0,
            has_unreachable: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ScopeStack — Manages the stack of lexical scopes during semantic analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Manages the stack of lexical scopes during semantic analysis.
///
/// The scope stack is initialized with a single file scope at depth 0, which
/// is always present and can never be popped. As the analyzer traverses the
/// AST, scopes are pushed and popped to reflect the lexical structure of the
/// C source:
///
/// - **Entering a function definition**: push `Function` scope, then push
///   `Block` scope for the function body.
/// - **Entering a block `{`**: push `Block` scope.
/// - **Entering a parameter list `(` in a declaration (not definition)**:
///   push `Prototype` scope.
/// - **Exiting a block `}`**: pop `Block` scope.
/// - **Exiting a function body `}`**: pop `Block` scope, then pop `Function`
///   scope.
/// - **Exiting a parameter list `)`**: pop `Prototype` scope.
///
/// The scope stack also tracks loop and switch nesting depth for validating
/// `break` and `continue` statements.
///
/// # Invariants
///
/// - The scope stack always contains at least one scope (file scope at depth 0).
/// - File scope (the bottom entry) can never be popped.
/// - Depth values are monotonically increasing from bottom (0) to top.
pub struct ScopeStack {
    /// Stack of scopes. `scopes[0]` is always the file scope.
    scopes: Vec<Scope>,

    /// Nesting depth of loop constructs (for/while/do-while).
    /// Incremented when entering a loop body, decremented when exiting.
    loop_depth: usize,

    /// Nesting depth of switch statements.
    /// Incremented when entering a switch body, decremented when exiting.
    switch_depth: usize,
}

impl ScopeStack {
    // ─────────────────────────────────────────────────────────────────────
    // Construction
    // ─────────────────────────────────────────────────────────────────────

    /// Creates a new scope stack initialized with file scope at depth 0.
    ///
    /// The file scope is always present and represents the translation unit's
    /// top-level scope. It can never be popped.
    ///
    /// # Examples
    ///
    /// ```
    /// # use bcc::sema::scope::ScopeStack;
    /// let stack = ScopeStack::new();
    /// assert_eq!(stack.depth(), 0);
    /// assert!(stack.is_file_scope());
    /// ```
    pub fn new() -> Self {
        let mut scopes = Vec::with_capacity(16); // Pre-allocate for typical nesting depth
        scopes.push(Scope::new(ScopeKind::File, 0));
        ScopeStack {
            scopes,
            loop_depth: 0,
            switch_depth: 0,
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Scope Entry and Exit
    // ─────────────────────────────────────────────────────────────────────

    /// Pushes a new scope onto the stack and returns a reference to it.
    ///
    /// The new scope's depth is set to the current stack length (i.e., one more
    /// than the current top scope's depth). The scope starts with zero symbols
    /// and reachable code.
    ///
    /// # Arguments
    ///
    /// * `kind` — The kind of scope to push (`Block`, `Function`, `Prototype`).
    ///
    /// # Returns
    ///
    /// A reference to the newly pushed scope.
    ///
    /// # Usage patterns
    ///
    /// - Entering a function body: `push(Function)` then `push(Block)`
    /// - Entering a `{ }` block: `push(Block)`
    /// - Entering a parameter list in a declaration: `push(Prototype)`
    pub fn push(&mut self, kind: ScopeKind) -> &Scope {
        let depth = self.scopes.len();
        self.scopes.push(Scope::new(kind, depth));
        // Return reference to the scope we just pushed (now at the top)
        &self.scopes[self.scopes.len() - 1]
    }

    /// Pops the innermost scope from the stack and returns it.
    ///
    /// The returned scope can be inspected for cleanup purposes (e.g., the
    /// symbol table can use the scope depth to remove symbols that are going
    /// out of scope).
    ///
    /// # Panics
    ///
    /// Panics if an attempt is made to pop the file scope (the last remaining
    /// scope). File scope must always remain at the bottom of the stack.
    ///
    /// # Usage patterns
    ///
    /// - Exiting a `}` block: `pop()` removes the `Block` scope
    /// - Exiting a function body: `pop()` for `Block`, then `pop()` for `Function`
    /// - Exiting a parameter list `)`: `pop()` removes the `Prototype` scope
    pub fn pop(&mut self) -> Scope {
        if self.scopes.len() <= 1 {
            panic!(
                "attempted to pop file scope from the scope stack; \
                 file scope must always remain as the bottom entry"
            );
        }
        // The unwrap is safe because we've verified len > 1 above.
        self.scopes.pop().expect("scope stack is non-empty after length check")
    }

    // ─────────────────────────────────────────────────────────────────────
    // Query Methods
    // ─────────────────────────────────────────────────────────────────────

    /// Returns a reference to the current (innermost) scope.
    ///
    /// This is always valid because the file scope is always present.
    ///
    /// # Panics
    ///
    /// Cannot panic — the scope stack always contains at least the file scope.
    pub fn current(&self) -> &Scope {
        // File scope is always present, so `last()` always succeeds.
        self.scopes.last().expect("scope stack always has at least file scope")
    }

    /// Returns the kind of the current (innermost) scope.
    pub fn current_kind(&self) -> ScopeKind {
        self.current().kind
    }

    /// Returns the current nesting depth (0 = file scope).
    ///
    /// The depth equals the index of the topmost scope in the stack. File scope
    /// has depth 0, the first nested scope has depth 1, etc.
    pub fn depth(&self) -> usize {
        self.current().depth
    }

    /// Returns `true` if the current scope is file scope.
    ///
    /// File scope is indicated by the stack containing exactly one entry
    /// (the file scope at depth 0).
    pub fn is_file_scope(&self) -> bool {
        self.scopes.len() == 1
    }

    /// Returns `true` if the current (innermost) scope is a block scope.
    pub fn is_block_scope(&self) -> bool {
        self.current_kind() == ScopeKind::Block
    }

    /// Returns `true` if the current (innermost) scope is a function scope.
    pub fn is_function_scope(&self) -> bool {
        self.current_kind() == ScopeKind::Function
    }

    /// Returns `true` if we are anywhere inside a function.
    ///
    /// This checks whether a `Function` scope exists anywhere in the scope
    /// stack. This is used to validate that certain constructs (like `return`,
    /// `goto`, labels, and local variable declarations) only appear inside
    /// function bodies.
    ///
    /// # Examples
    ///
    /// - At file scope: returns `false`
    /// - Inside a function body (Function + Block): returns `true`
    /// - Inside a nested block within a function: returns `true`
    pub fn is_in_function(&self) -> bool {
        self.scopes.iter().any(|s| s.kind == ScopeKind::Function)
    }

    /// Returns `true` if we are inside a loop or switch statement.
    ///
    /// Used to validate `break` and `continue` statements — `break` is valid
    /// inside loops and switch statements, while `continue` is valid only
    /// inside loops. This method returns `true` if either a loop or switch
    /// is currently active.
    ///
    /// Loop and switch nesting is tracked via `enter_loop()`/`exit_loop()` and
    /// `enter_switch()`/`exit_switch()` respectively.
    pub fn is_in_loop_or_switch(&self) -> bool {
        self.loop_depth > 0 || self.switch_depth > 0
    }

    /// Returns `true` if we are currently inside a loop body.
    ///
    /// Used to validate `continue` statements, which are only valid in loops
    /// (not in switch statements).
    pub fn is_in_loop(&self) -> bool {
        self.loop_depth > 0
    }

    /// Returns `true` if we are currently inside a switch body.
    ///
    /// Used to validate `case` and `default` labels, which are only valid
    /// directly within a switch statement.
    pub fn is_in_switch(&self) -> bool {
        self.switch_depth > 0
    }

    // ─────────────────────────────────────────────────────────────────────
    // Scope Iteration
    // ─────────────────────────────────────────────────────────────────────

    /// Returns an iterator over scopes from innermost to outermost.
    ///
    /// This is the primary lookup order for the symbol table: inner scopes
    /// are searched first (for variable shadowing), then progressively outer
    /// scopes up to file scope.
    ///
    /// # Returns
    ///
    /// An iterator yielding `&Scope` references from the innermost (top of
    /// stack) to the outermost (file scope at depth 0).
    pub fn iter_from_inner(&self) -> impl Iterator<Item = &Scope> {
        self.scopes.iter().rev()
    }

    /// Finds the nearest enclosing scope of the given kind, searching from
    /// the current (innermost) scope outward.
    ///
    /// Returns `None` if no scope of the requested kind exists in the stack.
    ///
    /// # Arguments
    ///
    /// * `kind` — The `ScopeKind` to search for.
    ///
    /// # Examples
    ///
    /// ```text
    /// // Stack: [File, Function, Block, Block]
    /// find_enclosing(Function) => Some(&Scope { kind: Function, depth: 1, .. })
    /// find_enclosing(Prototype) => None
    /// ```
    pub fn find_enclosing(&self, kind: ScopeKind) -> Option<&Scope> {
        self.scopes.iter().rev().find(|s| s.kind == kind)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Scope Tracking Utilities
    // ─────────────────────────────────────────────────────────────────────

    /// Increments the symbol count of the current (innermost) scope.
    ///
    /// Called by the semantic analyzer each time a new symbol is registered
    /// in the current scope. Useful for tracking scope size and for diagnostics.
    pub fn increment_symbol_count(&mut self) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.symbol_count += 1;
        }
    }

    /// Marks the current scope as having unreachable code.
    ///
    /// Called after encountering `return`, `break`, `continue`, or an
    /// unconditional `goto`. Subsequent statements in this scope (before
    /// encountering a new label that could be a jump target) are unreachable
    /// and should trigger a "unreachable code" warning.
    pub fn set_unreachable(&mut self) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.has_unreachable = true;
        }
    }

    /// Returns `true` if the current scope is marked as having unreachable code.
    ///
    /// This flag is set by `set_unreachable()` and automatically cleared when
    /// a new scope is pushed (since new scopes start with reachable code).
    pub fn is_unreachable(&self) -> bool {
        self.current().has_unreachable
    }

    // ─────────────────────────────────────────────────────────────────────
    // Label Scope Helpers
    // ─────────────────────────────────────────────────────────────────────

    /// Checks whether two scope depths are within the same function.
    ///
    /// In C, labels have function scope — they are visible throughout the
    /// entire function body regardless of block nesting. This means a `goto`
    /// can target any label within the same function. This helper verifies
    /// that two depths (e.g., the depth of a `goto` statement and the depth
    /// of its target label) are within the same function.
    ///
    /// Two depths are considered to be in the same function if the nearest
    /// enclosing `Function` scope for each depth is the same. If neither depth
    /// has an enclosing function scope (i.e., both are at file scope), they
    /// are trivially in the same "scope" but labels at file scope are invalid
    /// in C.
    ///
    /// # Arguments
    ///
    /// * `depth_a` — The scope depth of the first location.
    /// * `depth_b` — The scope depth of the second location.
    ///
    /// # Returns
    ///
    /// `true` if both depths are enclosed by the same `Function` scope (or
    /// both have no enclosing function scope).
    pub fn is_same_function_scope(&self, depth_a: usize, depth_b: usize) -> bool {
        let func_depth_for_a = self.find_enclosing_function_depth(depth_a);
        let func_depth_for_b = self.find_enclosing_function_depth(depth_b);
        func_depth_for_a == func_depth_for_b
    }

    // ─────────────────────────────────────────────────────────────────────
    // Loop and Switch Tracking
    // ─────────────────────────────────────────────────────────────────────

    /// Signals entry into a loop body (for/while/do-while).
    ///
    /// Increments the internal loop nesting counter. Must be paired with a
    /// corresponding `exit_loop()` call when the loop body is exited.
    pub fn enter_loop(&mut self) {
        self.loop_depth += 1;
    }

    /// Signals exit from a loop body.
    ///
    /// Decrements the internal loop nesting counter. Must be paired with a
    /// preceding `enter_loop()` call.
    ///
    /// # Panics
    ///
    /// Panics if called when not inside a loop (loop_depth is already 0).
    pub fn exit_loop(&mut self) {
        debug_assert!(
            self.loop_depth > 0,
            "exit_loop() called when not inside a loop"
        );
        self.loop_depth = self.loop_depth.saturating_sub(1);
    }

    /// Signals entry into a switch statement body.
    ///
    /// Increments the internal switch nesting counter. Must be paired with a
    /// corresponding `exit_switch()` call when the switch body is exited.
    pub fn enter_switch(&mut self) {
        self.switch_depth += 1;
    }

    /// Signals exit from a switch statement body.
    ///
    /// Decrements the internal switch nesting counter. Must be paired with a
    /// preceding `enter_switch()` call.
    ///
    /// # Panics
    ///
    /// Panics if called when not inside a switch (switch_depth is already 0).
    pub fn exit_switch(&mut self) {
        debug_assert!(
            self.switch_depth > 0,
            "exit_switch() called when not inside a switch"
        );
        self.switch_depth = self.switch_depth.saturating_sub(1);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Internal Helpers
    // ─────────────────────────────────────────────────────────────────────

    /// Finds the depth of the nearest enclosing `Function` scope at or below
    /// the given depth.
    ///
    /// Scans backward through the scope stack from `depth` toward depth 0,
    /// looking for a `Function` scope. Returns `None` if no function scope
    /// is found (i.e., the depth is within file scope only).
    ///
    /// If the given depth exceeds the current stack size, it is clamped to
    /// the maximum valid index.
    fn find_enclosing_function_depth(&self, depth: usize) -> Option<usize> {
        // Clamp to valid stack range
        let max_idx = if self.scopes.is_empty() {
            return None;
        } else {
            depth.min(self.scopes.len() - 1)
        };

        // Scan from the given depth backward toward file scope
        for i in (0..=max_idx).rev() {
            if self.scopes[i].kind == ScopeKind::Function {
                return Some(i);
            }
        }
        None
    }

    /// Returns the total number of scopes currently on the stack.
    ///
    /// This is always >= 1 because file scope is always present.
    pub fn len(&self) -> usize {
        self.scopes.len()
    }

    /// Returns `true` if the scope stack contains only the file scope.
    ///
    /// Equivalent to `self.len() == 1` and `self.is_file_scope()`.
    pub fn is_at_file_level(&self) -> bool {
        self.scopes.len() == 1
    }

    /// Returns a mutable reference to the current (innermost) scope.
    ///
    /// Useful for directly manipulating scope properties (e.g., resetting
    /// unreachable status when a label is encountered).
    pub fn current_mut(&mut self) -> &mut Scope {
        self.scopes.last_mut().expect("scope stack always has at least file scope")
    }

    /// Clears the unreachable flag on the current scope.
    ///
    /// Called when encountering a label that could be a jump target, making
    /// subsequent code potentially reachable again.
    pub fn clear_unreachable(&mut self) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.has_unreachable = false;
        }
    }
}

impl Default for ScopeStack {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ScopeStack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScopeStack")
            .field("depth", &self.depth())
            .field("current_kind", &self.current_kind())
            .field("scopes_count", &self.scopes.len())
            .field("loop_depth", &self.loop_depth)
            .field("switch_depth", &self.switch_depth)
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Construction Tests ──────────────────────────────────────────────

    #[test]
    fn new_scope_stack_starts_at_file_scope() {
        let stack = ScopeStack::new();
        assert_eq!(stack.depth(), 0);
        assert!(stack.is_file_scope());
        assert_eq!(stack.current_kind(), ScopeKind::File);
        assert_eq!(stack.len(), 1);
    }

    #[test]
    fn new_scope_stack_is_not_in_function() {
        let stack = ScopeStack::new();
        assert!(!stack.is_in_function());
    }

    #[test]
    fn new_scope_stack_is_not_in_loop_or_switch() {
        let stack = ScopeStack::new();
        assert!(!stack.is_in_loop_or_switch());
        assert!(!stack.is_in_loop());
        assert!(!stack.is_in_switch());
    }

    #[test]
    fn new_scope_stack_has_zero_symbol_count() {
        let stack = ScopeStack::new();
        assert_eq!(stack.current().symbol_count, 0);
    }

    #[test]
    fn new_scope_stack_is_reachable() {
        let stack = ScopeStack::new();
        assert!(!stack.is_unreachable());
    }

    #[test]
    fn default_creates_same_as_new() {
        let stack = ScopeStack::default();
        assert_eq!(stack.depth(), 0);
        assert!(stack.is_file_scope());
    }

    // ─── Push/Pop Tests ──────────────────────────────────────────────────

    #[test]
    fn push_block_scope_increases_depth() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeKind::Block);
        assert_eq!(stack.depth(), 1);
        assert!(!stack.is_file_scope());
        assert!(stack.is_block_scope());
    }

    #[test]
    fn pop_block_scope_decreases_depth() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeKind::Block);
        assert_eq!(stack.depth(), 1);
        let popped = stack.pop();
        assert_eq!(popped.kind, ScopeKind::Block);
        assert_eq!(popped.depth, 1);
        assert_eq!(stack.depth(), 0);
        assert!(stack.is_file_scope());
    }

    #[test]
    fn push_function_then_block() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeKind::Function);
        assert_eq!(stack.depth(), 1);
        assert!(stack.is_function_scope());

        stack.push(ScopeKind::Block);
        assert_eq!(stack.depth(), 2);
        assert!(stack.is_block_scope());

        // Pop both
        stack.pop();
        assert_eq!(stack.depth(), 1);
        assert!(stack.is_function_scope());

        stack.pop();
        assert_eq!(stack.depth(), 0);
        assert!(stack.is_file_scope());
    }

    #[test]
    fn push_prototype_scope() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeKind::Prototype);
        assert_eq!(stack.depth(), 1);
        assert_eq!(stack.current_kind(), ScopeKind::Prototype);

        let popped = stack.pop();
        assert_eq!(popped.kind, ScopeKind::Prototype);
        assert!(stack.is_file_scope());
    }

    #[test]
    fn push_returns_reference_to_new_scope() {
        let mut stack = ScopeStack::new();
        let scope = stack.push(ScopeKind::Block);
        assert_eq!(scope.kind, ScopeKind::Block);
        assert_eq!(scope.depth, 1);
        assert_eq!(scope.symbol_count, 0);
        assert!(!scope.has_unreachable);
    }

    #[test]
    fn pop_returns_popped_scope_with_correct_data() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeKind::Block);
        stack.increment_symbol_count();
        stack.increment_symbol_count();
        stack.set_unreachable();

        let popped = stack.pop();
        assert_eq!(popped.kind, ScopeKind::Block);
        assert_eq!(popped.symbol_count, 2);
        assert!(popped.has_unreachable);
    }

    // ─── File Scope Protection ──────────────────────────────────────────

    #[test]
    #[should_panic(expected = "attempted to pop file scope")]
    fn cannot_pop_file_scope() {
        let mut stack = ScopeStack::new();
        stack.pop(); // Should panic
    }

    #[test]
    #[should_panic(expected = "attempted to pop file scope")]
    fn cannot_pop_file_scope_after_push_pop() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeKind::Block);
        stack.pop();
        stack.pop(); // Should panic — file scope is all that remains
    }

    // ─── Kind Query Tests ────────────────────────────────────────────────

    #[test]
    fn current_kind_reflects_top_of_stack() {
        let mut stack = ScopeStack::new();
        assert_eq!(stack.current_kind(), ScopeKind::File);

        stack.push(ScopeKind::Function);
        assert_eq!(stack.current_kind(), ScopeKind::Function);

        stack.push(ScopeKind::Block);
        assert_eq!(stack.current_kind(), ScopeKind::Block);

        stack.push(ScopeKind::Prototype);
        assert_eq!(stack.current_kind(), ScopeKind::Prototype);
    }

    #[test]
    fn is_block_scope_only_for_block() {
        let mut stack = ScopeStack::new();
        assert!(!stack.is_block_scope());

        stack.push(ScopeKind::Block);
        assert!(stack.is_block_scope());

        stack.push(ScopeKind::Function);
        assert!(!stack.is_block_scope());
    }

    #[test]
    fn is_function_scope_only_for_function() {
        let mut stack = ScopeStack::new();
        assert!(!stack.is_function_scope());

        stack.push(ScopeKind::Function);
        assert!(stack.is_function_scope());

        stack.push(ScopeKind::Block);
        assert!(!stack.is_function_scope());
    }

    // ─── Nesting Tests ───────────────────────────────────────────────────

    #[test]
    fn deep_nesting_tracks_depth_correctly() {
        let mut stack = ScopeStack::new();
        // File → Function → Block → Block → Block
        stack.push(ScopeKind::Function);
        assert_eq!(stack.depth(), 1);

        stack.push(ScopeKind::Block);
        assert_eq!(stack.depth(), 2);

        stack.push(ScopeKind::Block);
        assert_eq!(stack.depth(), 3);

        stack.push(ScopeKind::Block);
        assert_eq!(stack.depth(), 4);
        assert_eq!(stack.len(), 5); // File + Function + 3 Blocks

        // Pop all three blocks
        stack.pop();
        assert_eq!(stack.depth(), 3);
        stack.pop();
        assert_eq!(stack.depth(), 2);
        stack.pop();
        assert_eq!(stack.depth(), 1);

        // Should be back at Function scope
        assert!(stack.is_function_scope());
        assert_eq!(stack.current_kind(), ScopeKind::Function);

        // Pop function
        stack.pop();
        assert!(stack.is_file_scope());
    }

    // ─── is_in_function Tests ────────────────────────────────────────────

    #[test]
    fn is_in_function_at_file_scope() {
        let stack = ScopeStack::new();
        assert!(!stack.is_in_function());
    }

    #[test]
    fn is_in_function_inside_function_body() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeKind::Function);
        assert!(stack.is_in_function());

        stack.push(ScopeKind::Block);
        assert!(stack.is_in_function());
    }

    #[test]
    fn is_in_function_nested_blocks() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeKind::Function);
        stack.push(ScopeKind::Block);
        stack.push(ScopeKind::Block);
        stack.push(ScopeKind::Block);
        assert!(stack.is_in_function());

        // Even deeply nested, still in a function
        assert_eq!(stack.depth(), 4);
        assert!(stack.is_in_function());
    }

    #[test]
    fn is_in_function_false_after_function_popped() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeKind::Function);
        stack.push(ScopeKind::Block);
        assert!(stack.is_in_function());

        stack.pop(); // Pop Block
        stack.pop(); // Pop Function
        assert!(!stack.is_in_function());
    }

    // ─── Loop/Switch Tracking Tests ──────────────────────────────────────

    #[test]
    fn loop_tracking() {
        let mut stack = ScopeStack::new();
        assert!(!stack.is_in_loop());
        assert!(!stack.is_in_loop_or_switch());

        stack.enter_loop();
        assert!(stack.is_in_loop());
        assert!(stack.is_in_loop_or_switch());

        stack.enter_loop(); // Nested loop
        assert!(stack.is_in_loop());

        stack.exit_loop();
        assert!(stack.is_in_loop()); // Still in outer loop

        stack.exit_loop();
        assert!(!stack.is_in_loop());
        assert!(!stack.is_in_loop_or_switch());
    }

    #[test]
    fn switch_tracking() {
        let mut stack = ScopeStack::new();
        assert!(!stack.is_in_switch());
        assert!(!stack.is_in_loop_or_switch());

        stack.enter_switch();
        assert!(stack.is_in_switch());
        assert!(stack.is_in_loop_or_switch());

        stack.exit_switch();
        assert!(!stack.is_in_switch());
        assert!(!stack.is_in_loop_or_switch());
    }

    #[test]
    fn loop_and_switch_combined() {
        let mut stack = ScopeStack::new();
        stack.enter_loop();
        stack.enter_switch();
        assert!(stack.is_in_loop());
        assert!(stack.is_in_switch());
        assert!(stack.is_in_loop_or_switch());

        stack.exit_switch();
        assert!(stack.is_in_loop());
        assert!(!stack.is_in_switch());
        assert!(stack.is_in_loop_or_switch()); // Still in loop

        stack.exit_loop();
        assert!(!stack.is_in_loop_or_switch());
    }

    // ─── Unreachable Marking Tests ───────────────────────────────────────

    #[test]
    fn set_unreachable_marks_current_scope() {
        let mut stack = ScopeStack::new();
        assert!(!stack.is_unreachable());

        stack.set_unreachable();
        assert!(stack.is_unreachable());
    }

    #[test]
    fn new_scope_is_reachable_even_if_parent_unreachable() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeKind::Block);
        stack.set_unreachable();
        assert!(stack.is_unreachable());

        // Push a new block — it starts as reachable
        stack.push(ScopeKind::Block);
        assert!(!stack.is_unreachable());

        // Pop inner block — parent is still unreachable
        stack.pop();
        assert!(stack.is_unreachable());
    }

    #[test]
    fn clear_unreachable_resets_flag() {
        let mut stack = ScopeStack::new();
        stack.set_unreachable();
        assert!(stack.is_unreachable());

        stack.clear_unreachable();
        assert!(!stack.is_unreachable());
    }

    // ─── find_enclosing Tests ────────────────────────────────────────────

    #[test]
    fn find_enclosing_function_in_nested_blocks() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeKind::Function);
        stack.push(ScopeKind::Block);
        stack.push(ScopeKind::Block);

        let func = stack.find_enclosing(ScopeKind::Function);
        assert!(func.is_some());
        let func = func.unwrap();
        assert_eq!(func.kind, ScopeKind::Function);
        assert_eq!(func.depth, 1);
    }

    #[test]
    fn find_enclosing_function_at_file_scope_returns_none() {
        let stack = ScopeStack::new();
        assert!(stack.find_enclosing(ScopeKind::Function).is_none());
    }

    #[test]
    fn find_enclosing_file_always_found() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeKind::Function);
        stack.push(ScopeKind::Block);

        let file = stack.find_enclosing(ScopeKind::File);
        assert!(file.is_some());
        assert_eq!(file.unwrap().depth, 0);
    }

    #[test]
    fn find_enclosing_prototype_returns_none_when_absent() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeKind::Function);
        stack.push(ScopeKind::Block);

        assert!(stack.find_enclosing(ScopeKind::Prototype).is_none());
    }

    #[test]
    fn find_enclosing_returns_nearest() {
        let mut stack = ScopeStack::new();
        // Two function scopes (nested functions are not C11, but test the logic)
        stack.push(ScopeKind::Function);
        stack.push(ScopeKind::Block);
        stack.push(ScopeKind::Function); // depth 3
        stack.push(ScopeKind::Block);

        let func = stack.find_enclosing(ScopeKind::Function);
        assert!(func.is_some());
        // Should find the inner (nearest) function scope at depth 3
        assert_eq!(func.unwrap().depth, 3);
    }

    // ─── iter_from_inner Tests ───────────────────────────────────────────

    #[test]
    fn iter_from_inner_visits_scopes_in_reverse_order() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeKind::Function);
        stack.push(ScopeKind::Block);
        stack.push(ScopeKind::Block);

        let kinds: Vec<ScopeKind> = stack.iter_from_inner().map(|s| s.kind).collect();
        assert_eq!(
            kinds,
            vec![
                ScopeKind::Block,
                ScopeKind::Block,
                ScopeKind::Function,
                ScopeKind::File,
            ]
        );
    }

    #[test]
    fn iter_from_inner_file_scope_only() {
        let stack = ScopeStack::new();
        let kinds: Vec<ScopeKind> = stack.iter_from_inner().map(|s| s.kind).collect();
        assert_eq!(kinds, vec![ScopeKind::File]);
    }

    // ─── Symbol Count Tests ──────────────────────────────────────────────

    #[test]
    fn symbol_count_starts_at_zero() {
        let stack = ScopeStack::new();
        assert_eq!(stack.current().symbol_count, 0);
    }

    #[test]
    fn increment_symbol_count_increases_count() {
        let mut stack = ScopeStack::new();
        stack.increment_symbol_count();
        assert_eq!(stack.current().symbol_count, 1);

        stack.increment_symbol_count();
        assert_eq!(stack.current().symbol_count, 2);
    }

    #[test]
    fn symbol_count_is_per_scope() {
        let mut stack = ScopeStack::new();
        stack.increment_symbol_count(); // File scope: 1 symbol

        stack.push(ScopeKind::Block);
        assert_eq!(stack.current().symbol_count, 0); // New scope starts at 0

        stack.increment_symbol_count();
        stack.increment_symbol_count();
        assert_eq!(stack.current().symbol_count, 2);

        stack.pop();
        assert_eq!(stack.current().symbol_count, 1); // Back to file scope
    }

    // ─── is_same_function_scope Tests ────────────────────────────────────

    #[test]
    fn same_function_scope_in_simple_function() {
        let mut stack = ScopeStack::new();
        // File(0) → Function(1) → Block(2) → Block(3)
        stack.push(ScopeKind::Function);
        stack.push(ScopeKind::Block);
        stack.push(ScopeKind::Block);

        // Depths 2 and 3 are both in the function at depth 1
        assert!(stack.is_same_function_scope(2, 3));
        assert!(stack.is_same_function_scope(3, 2)); // Symmetric

        // Depth 1 (Function) and depth 3 (Block) are in the same function
        assert!(stack.is_same_function_scope(1, 3));
    }

    #[test]
    fn same_function_scope_at_file_level() {
        let stack = ScopeStack::new();
        // Both at file scope with no function — trivially same "scope"
        assert!(stack.is_same_function_scope(0, 0));
    }

    #[test]
    fn same_function_scope_file_vs_function() {
        let mut stack = ScopeStack::new();
        // File(0) → Function(1) → Block(2)
        stack.push(ScopeKind::Function);
        stack.push(ScopeKind::Block);

        // Depth 0 (file scope, no function) vs depth 2 (inside function)
        assert!(!stack.is_same_function_scope(0, 2));
    }

    #[test]
    fn same_function_scope_with_clamped_depth() {
        let mut stack = ScopeStack::new();
        // File(0) → Function(1) → Block(2)
        stack.push(ScopeKind::Function);
        stack.push(ScopeKind::Block);

        // Depth 100 is clamped to the stack top (depth 2)
        // Both clamped to depth 2, same function
        assert!(stack.is_same_function_scope(100, 2));
    }

    // ─── Scope Display Tests ─────────────────────────────────────────────

    #[test]
    fn scope_kind_display() {
        assert_eq!(format!("{}", ScopeKind::File), "file");
        assert_eq!(format!("{}", ScopeKind::Function), "function");
        assert_eq!(format!("{}", ScopeKind::Block), "block");
        assert_eq!(format!("{}", ScopeKind::Prototype), "prototype");
    }

    // ─── Debug Formatting Tests ──────────────────────────────────────────

    #[test]
    fn scope_stack_debug_format() {
        let stack = ScopeStack::new();
        let debug_str = format!("{:?}", stack);
        assert!(debug_str.contains("ScopeStack"));
        assert!(debug_str.contains("depth"));
    }

    #[test]
    fn scope_kind_debug_format() {
        let debug_str = format!("{:?}", ScopeKind::File);
        assert_eq!(debug_str, "File");
    }

    #[test]
    fn scope_debug_format() {
        let scope = Scope::new(ScopeKind::Block, 3);
        let debug_str = format!("{:?}", scope);
        assert!(debug_str.contains("Block"));
        assert!(debug_str.contains("3"));
    }

    // ─── Clone and Copy Tests ────────────────────────────────────────────

    #[test]
    fn scope_kind_is_copy() {
        let kind = ScopeKind::Block;
        let kind2 = kind; // Copy
        assert_eq!(kind, kind2);
    }

    #[test]
    fn scope_is_clone() {
        let scope = Scope::new(ScopeKind::Function, 2);
        let scope2 = scope.clone();
        assert_eq!(scope2.kind, ScopeKind::Function);
        assert_eq!(scope2.depth, 2);
    }

    // ─── Edge Case Tests ─────────────────────────────────────────────────

    #[test]
    fn rapid_push_pop_cycles() {
        let mut stack = ScopeStack::new();
        for _ in 0..100 {
            stack.push(ScopeKind::Block);
            stack.pop();
        }
        assert!(stack.is_file_scope());
        assert_eq!(stack.depth(), 0);
    }

    #[test]
    fn many_nested_scopes() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeKind::Function);
        for i in 0..50 {
            stack.push(ScopeKind::Block);
            assert_eq!(stack.depth(), i + 2); // +1 for Function, +1 for zero-indexing
        }
        assert_eq!(stack.len(), 52); // File + Function + 50 Blocks

        for _ in 0..50 {
            stack.pop();
        }
        assert!(stack.is_function_scope());
        stack.pop();
        assert!(stack.is_file_scope());
    }

    #[test]
    fn current_mut_allows_modification() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeKind::Block);

        {
            let scope = stack.current_mut();
            scope.symbol_count = 42;
            scope.has_unreachable = true;
        }

        assert_eq!(stack.current().symbol_count, 42);
        assert!(stack.is_unreachable());
    }

    #[test]
    fn len_and_is_at_file_level() {
        let mut stack = ScopeStack::new();
        assert_eq!(stack.len(), 1);
        assert!(stack.is_at_file_level());

        stack.push(ScopeKind::Block);
        assert_eq!(stack.len(), 2);
        assert!(!stack.is_at_file_level());

        stack.pop();
        assert_eq!(stack.len(), 1);
        assert!(stack.is_at_file_level());
    }

    // ─── Comprehensive Workflow Tests ────────────────────────────────────

    #[test]
    fn typical_function_analysis_workflow() {
        let mut stack = ScopeStack::new();

        // Analyze: void foo(int x) { int y; if (x) { int z; } return; }

        // Enter function definition
        stack.push(ScopeKind::Function); // depth 1
        assert!(stack.is_in_function());

        // Enter function body block
        stack.push(ScopeKind::Block); // depth 2
        assert!(stack.is_block_scope());

        // Declare local variable y
        stack.increment_symbol_count();
        assert_eq!(stack.current().symbol_count, 1);

        // Enter if-block
        stack.push(ScopeKind::Block); // depth 3
        assert_eq!(stack.depth(), 3);

        // Declare local variable z
        stack.increment_symbol_count();
        assert_eq!(stack.current().symbol_count, 1);

        // Exit if-block
        let if_scope = stack.pop();
        assert_eq!(if_scope.symbol_count, 1);
        assert_eq!(stack.depth(), 2);

        // return statement makes remaining code unreachable
        stack.set_unreachable();
        assert!(stack.is_unreachable());

        // Exit function body
        let body_scope = stack.pop();
        assert_eq!(body_scope.symbol_count, 1);

        // Exit function scope
        stack.pop();
        assert!(stack.is_file_scope());
        assert!(!stack.is_in_function());
    }

    #[test]
    fn function_with_loop_and_switch() {
        let mut stack = ScopeStack::new();

        // Enter function
        stack.push(ScopeKind::Function);
        stack.push(ScopeKind::Block);

        // Enter for-loop
        stack.enter_loop();
        stack.push(ScopeKind::Block);
        assert!(stack.is_in_loop());
        assert!(stack.is_in_loop_or_switch());
        assert!(!stack.is_in_switch());

        // Enter switch inside loop
        stack.enter_switch();
        stack.push(ScopeKind::Block);
        assert!(stack.is_in_loop());
        assert!(stack.is_in_switch());
        assert!(stack.is_in_loop_or_switch());

        // Exit switch
        stack.pop();
        stack.exit_switch();
        assert!(stack.is_in_loop());
        assert!(!stack.is_in_switch());

        // Exit loop
        stack.pop();
        stack.exit_loop();
        assert!(!stack.is_in_loop());
        assert!(!stack.is_in_loop_or_switch());

        // Clean up
        stack.pop(); // Block
        stack.pop(); // Function
        assert!(stack.is_file_scope());
    }

    #[test]
    fn prototype_scope_workflow() {
        let mut stack = ScopeStack::new();

        // Analyzing: void f(int x, int y);
        stack.push(ScopeKind::Prototype);
        assert_eq!(stack.current_kind(), ScopeKind::Prototype);

        // Register parameter symbols
        stack.increment_symbol_count(); // x
        stack.increment_symbol_count(); // y
        assert_eq!(stack.current().symbol_count, 2);

        // End of prototype
        let proto = stack.pop();
        assert_eq!(proto.kind, ScopeKind::Prototype);
        assert_eq!(proto.symbol_count, 2);
        assert!(stack.is_file_scope());
    }

    // ─── ScopeKind Equality Tests ────────────────────────────────────────

    #[test]
    fn scope_kind_equality() {
        assert_eq!(ScopeKind::File, ScopeKind::File);
        assert_eq!(ScopeKind::Function, ScopeKind::Function);
        assert_eq!(ScopeKind::Block, ScopeKind::Block);
        assert_eq!(ScopeKind::Prototype, ScopeKind::Prototype);

        assert_ne!(ScopeKind::File, ScopeKind::Function);
        assert_ne!(ScopeKind::File, ScopeKind::Block);
        assert_ne!(ScopeKind::File, ScopeKind::Prototype);
        assert_ne!(ScopeKind::Function, ScopeKind::Block);
        assert_ne!(ScopeKind::Function, ScopeKind::Prototype);
        assert_ne!(ScopeKind::Block, ScopeKind::Prototype);
    }

    #[test]
    fn scope_kind_hash_equality() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ScopeKind::File);
        set.insert(ScopeKind::Function);
        set.insert(ScopeKind::Block);
        set.insert(ScopeKind::Prototype);
        assert_eq!(set.len(), 4);

        // Inserting duplicate doesn't increase count
        set.insert(ScopeKind::File);
        assert_eq!(set.len(), 4);
    }
}
