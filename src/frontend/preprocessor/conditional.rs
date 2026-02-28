//! Conditional compilation state machine for the `bcc` preprocessor.
//!
//! This module manages the `#if`/`#ifdef`/`#ifndef`/`#elif`/`#else`/`#endif`
//! directive stack. It tracks nesting of conditional blocks and determines
//! which source lines are "active" (should be included in output) and which
//! are "inactive" (should be skipped).
//!
//! # Design
//!
//! The conditional stack is modeled as a `Vec<ConditionalLevel>` where each
//! entry represents one `#if`/`#ifdef`/`#ifndef` group. Within each group,
//! the current branch state tracks whether lines are active, inactive because
//! the condition was false, inactive because a prior branch was already taken,
//! or inactive because the entire parent conditional is inactive.
//!
//! # Correctness
//!
//! Incorrect conditional evaluation can cause correct code to be silently
//! dropped or wrong code to be compiled. This module handles all edge cases:
//! - Arbitrary nesting depth (no hardcoded limits)
//! - `#if 0` suppresses everything including nested `#if`/`#endif`
//! - `#elif` chains: only the first true branch after no prior taken is active
//! - Duplicate `#else` detection
//! - Unterminated conditional detection at EOF
//! - `#endif` without matching `#if` detection
//!
//! # Safety
//!
//! This module contains zero `unsafe` blocks. All operations use safe Rust.

use crate::common::diagnostics::DiagnosticEmitter;
use crate::common::source_map::SourceLocation;

// ---------------------------------------------------------------------------
// BranchState — the state of the current branch within a conditional group
// ---------------------------------------------------------------------------

/// The state of the current branch within a single `#if` group.
///
/// Each level on the conditional stack has a `BranchState` that determines
/// whether source lines at that nesting depth are active (emitted to output)
/// or inactive (skipped).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BranchState {
    /// The condition for this branch was true and no previous branch was taken,
    /// so lines are active and should be emitted to output.
    Active,

    /// A previous branch in this `#if` group was already taken. Even if the
    /// current branch's condition is true, lines are inactive because only
    /// one branch per group can be active.
    AlreadyTaken,

    /// The condition for this branch was false (and no previous branch was
    /// taken), so lines are inactive. A subsequent `#elif` or `#else` may
    /// still activate lines.
    Inactive,

    /// The parent conditional is inactive (`#if 0` or nested inside an
    /// inactive parent), so everything within this group is unconditionally
    /// inactive regardless of branch conditions.
    ParentInactive,
}

// ---------------------------------------------------------------------------
// ConditionalLevel — tracking struct for a single #if group
// ---------------------------------------------------------------------------

/// Tracks the state of a single `#if`/`#ifdef`/`#ifndef` group on the stack.
///
/// Each time a `#if`, `#ifdef`, or `#ifndef` directive is encountered, a new
/// `ConditionalLevel` is pushed onto the stack. The level records the current
/// branch state, whether any branch has been taken (for `#elif`/`#else`
/// logic), the source location of the opening directive (for unterminated
/// conditional diagnostics), and whether a `#else` has been seen (to detect
/// duplicate `#else` directives).
#[derive(Debug, Clone)]
struct ConditionalLevel {
    /// Current branch state within this `#if` group.
    state: BranchState,

    /// Whether any branch in this `#if` group has been taken. This is used
    /// by `#elif` and `#else` to determine whether they should activate:
    /// once a branch is taken, all subsequent branches become `AlreadyTaken`.
    any_branch_taken: bool,

    /// Source location of the opening `#if`/`#ifdef`/`#ifndef` directive.
    /// Stored for diagnostic messages when an unterminated conditional is
    /// detected at end-of-file.
    location: SourceLocation,

    /// Whether this `#if` group has seen a `#else` directive. Used to
    /// detect and report duplicate `#else` errors, and to reject `#elif`
    /// directives that appear after `#else`.
    seen_else: bool,
}

// ---------------------------------------------------------------------------
// ConditionalStack — the public conditional compilation state machine
// ---------------------------------------------------------------------------

/// The conditional compilation state machine for the preprocessor.
///
/// `ConditionalStack` maintains a stack of [`ConditionalLevel`] entries
/// representing nested `#if`/`#ifdef`/`#ifndef` groups. The preprocessor's
/// main loop queries [`is_active`](Self::is_active) on every source line to
/// determine whether the line should be processed and emitted, or silently
/// skipped.
///
/// # Usage
///
/// ```ignore
/// let mut cond = ConditionalStack::new();
///
/// // Entering #if 1
/// cond.push_if(true, loc);
/// assert!(cond.is_active());
///
/// // Entering nested #if 0
/// cond.push_if(false, loc2);
/// assert!(!cond.is_active());
///
/// // #endif for inner
/// cond.pop_endif(loc3, &mut diag).unwrap();
/// assert!(cond.is_active());
///
/// // #endif for outer
/// cond.pop_endif(loc4, &mut diag).unwrap();
/// assert!(cond.is_active()); // back to top-level
/// ```
///
/// # Integration
///
/// - **Called by**: `directives.rs` dispatches conditional directives here.
/// - **Consumed by**: `mod.rs` preprocessor main loop checks `is_active()`
///   to decide whether to process/emit lines.
pub struct ConditionalStack {
    /// Stack of conditional levels. An empty stack means we are at the
    /// top level, where all lines are unconditionally active.
    stack: Vec<ConditionalLevel>,
}

impl ConditionalStack {
    // -- Construction -------------------------------------------------------

    /// Creates a new, empty `ConditionalStack`.
    ///
    /// An empty stack represents top-level code, which is always active.
    #[inline]
    pub fn new() -> Self {
        ConditionalStack {
            stack: Vec::new(),
        }
    }

    // -- Activity Query -----------------------------------------------------

    /// Returns `true` if the current position is in an active branch.
    ///
    /// When the stack is empty (top-level code), this always returns `true`.
    /// When inside a conditional group, it returns `true` only if the
    /// top-of-stack branch state is [`BranchState::Active`].
    ///
    /// The preprocessor's main loop calls this on every source line to
    /// determine whether to process and emit the line.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.stack
            .last()
            .map_or(true, |level| level.state == BranchState::Active)
    }

    // -- Conditional Directive Processing -----------------------------------

    /// Pushes a new conditional level for `#if`, `#ifdef`, or `#ifndef`.
    ///
    /// The `condition` parameter is the evaluated boolean result of the
    /// directive's expression:
    /// - For `#if expr`: the result of evaluating `expr` (nonzero = true).
    /// - For `#ifdef NAME`: whether `NAME` is defined in the macro table.
    /// - For `#ifndef NAME`: whether `NAME` is **not** defined.
    ///
    /// The `location` parameter records where the directive appeared, for
    /// use in "unterminated `#if`" diagnostics at end-of-file.
    ///
    /// # State Transitions
    ///
    /// - If the parent is inactive → push `ParentInactive` regardless of
    ///   `condition`, because everything inside an inactive parent is inactive.
    /// - If the parent is active and `condition` is `true` → push `Active`,
    ///   mark `any_branch_taken = true`.
    /// - If the parent is active and `condition` is `false` → push `Inactive`,
    ///   mark `any_branch_taken = false`.
    pub fn push_if(&mut self, condition: bool, location: SourceLocation) {
        let parent_active = self.is_active();

        let state = if !parent_active {
            BranchState::ParentInactive
        } else if condition {
            BranchState::Active
        } else {
            BranchState::Inactive
        };

        self.stack.push(ConditionalLevel {
            state,
            any_branch_taken: parent_active && condition,
            location,
            seen_else: false,
        });
    }

    /// Processes a `#elif` directive.
    ///
    /// The `condition` parameter is the evaluated boolean result of the
    /// `#elif` expression. The `location` parameter is the source position
    /// of the `#elif` directive (used in error messages).
    ///
    /// # Errors
    ///
    /// Returns `Err(())` and emits a diagnostic if:
    /// - The stack is empty (`#elif without #if`).
    /// - A `#else` has already been seen in this group (`#elif after #else`).
    ///
    /// # State Transitions
    ///
    /// - If the parent was inactive → remain `ParentInactive`.
    /// - If any branch was already taken → set `AlreadyTaken`.
    /// - If no branch taken yet and `condition` is true → set `Active`,
    ///   mark `any_branch_taken = true`.
    /// - If no branch taken yet and `condition` is false → remain `Inactive`.
    pub fn process_elif(
        &mut self,
        condition: bool,
        location: SourceLocation,
        diagnostics: &mut DiagnosticEmitter,
    ) -> Result<(), ()> {
        let level = match self.stack.last_mut() {
            Some(level) => level,
            None => {
                diagnostics.error(location, "#elif without #if");
                return Err(());
            }
        };

        if level.seen_else {
            diagnostics.error(location, "#elif after #else");
            return Err(());
        }

        // Determine the new branch state based on parent activity and
        // whether a previous branch was already taken.
        level.state = if level.state == BranchState::ParentInactive {
            // Parent is inactive → everything remains inactive.
            BranchState::ParentInactive
        } else if level.any_branch_taken {
            // A previous branch in this group was already taken.
            BranchState::AlreadyTaken
        } else if condition {
            // No previous branch taken, and this condition is true → activate.
            level.any_branch_taken = true;
            BranchState::Active
        } else {
            // No previous branch taken, but this condition is false → inactive.
            BranchState::Inactive
        };

        Ok(())
    }

    /// Processes a `#else` directive.
    ///
    /// The `location` parameter is the source position of the `#else`
    /// directive (used in error messages).
    ///
    /// # Errors
    ///
    /// Returns `Err(())` and emits a diagnostic if:
    /// - The stack is empty (`#else without #if`).
    /// - A `#else` has already been seen in this group (`#else after #else`).
    ///
    /// # State Transitions
    ///
    /// - Sets `seen_else = true` to prevent subsequent `#elif` or `#else`.
    /// - If the parent was inactive → remain `ParentInactive`.
    /// - If any branch was already taken → set `AlreadyTaken`.
    /// - If no branch taken yet → set `Active`, mark `any_branch_taken = true`.
    pub fn process_else(
        &mut self,
        location: SourceLocation,
        diagnostics: &mut DiagnosticEmitter,
    ) -> Result<(), ()> {
        let level = match self.stack.last_mut() {
            Some(level) => level,
            None => {
                diagnostics.error(location, "#else without #if");
                return Err(());
            }
        };

        if level.seen_else {
            diagnostics.error(location, "#else after #else");
            return Err(());
        }

        level.seen_else = true;

        level.state = if level.state == BranchState::ParentInactive {
            // Parent is inactive → everything remains inactive.
            BranchState::ParentInactive
        } else if level.any_branch_taken {
            // A previous branch in this group was already taken.
            BranchState::AlreadyTaken
        } else {
            // No previous branch taken → this #else branch is active.
            level.any_branch_taken = true;
            BranchState::Active
        };

        Ok(())
    }

    /// Processes a `#endif` directive.
    ///
    /// Pops the top level from the conditional stack, ending the current
    /// `#if`/`#ifdef`/`#ifndef` group.
    ///
    /// The `location` parameter is the source position of the `#endif`
    /// directive (used in the error message if the stack is empty).
    ///
    /// # Errors
    ///
    /// Returns `Err(())` and emits a diagnostic if the stack is empty
    /// (`#endif without #if`).
    pub fn pop_endif(
        &mut self,
        location: SourceLocation,
        diagnostics: &mut DiagnosticEmitter,
    ) -> Result<(), ()> {
        if self.stack.pop().is_none() {
            diagnostics.error(location, "#endif without #if");
            return Err(());
        }
        Ok(())
    }

    /// Checks for unterminated conditional blocks at end of file.
    ///
    /// If the stack is non-empty, each remaining level represents an
    /// `#if`/`#ifdef`/`#ifndef` that was never closed by a matching `#endif`.
    /// An error diagnostic is emitted for each unterminated conditional,
    /// using the stored source location of the opening directive.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the stack is empty (all conditionals were properly closed).
    /// `Err(())` if any unterminated conditionals were found.
    pub fn check_unterminated(
        &self,
        diagnostics: &mut DiagnosticEmitter,
    ) -> Result<(), ()> {
        if self.stack.is_empty() {
            return Ok(());
        }

        // Emit an error for each unterminated conditional, using the stored
        // location of the opening #if/#ifdef/#ifndef directive.
        for level in &self.stack {
            diagnostics.error(level.location, "unterminated #if");
        }

        Err(())
    }

    // -- Helper Methods -----------------------------------------------------

    /// Returns the current nesting depth of the conditional stack.
    ///
    /// A depth of 0 means we are at top-level code (outside all conditionals).
    /// A depth of N means there are N nested, unclosed `#if`/`#ifdef`/`#ifndef`
    /// directives.
    #[inline]
    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// Returns `true` if we are inside any conditional block.
    ///
    /// Equivalent to `self.depth() > 0`.
    #[inline]
    pub fn is_in_conditional(&self) -> bool {
        !self.stack.is_empty()
    }

    /// Returns `true` if the preprocessor should process the current directive.
    ///
    /// Even inside inactive conditional blocks (e.g., inside `#if 0`), the
    /// preprocessor must still process conditional directives (`#if`, `#ifdef`,
    /// `#ifndef`, `#elif`, `#else`, `#endif`) to correctly track nesting depth.
    /// All other directives (`#define`, `#include`, `#undef`, etc.) should
    /// only be processed when the current branch is active.
    ///
    /// This method returns `true` when the current branch is active, meaning
    /// ALL directives should be processed. When inactive, the caller must
    /// still handle conditional directives (by checking directive kind) but
    /// should skip non-conditional directives.
    ///
    /// # Usage
    ///
    /// ```ignore
    /// if cond_stack.should_process_directive() {
    ///     // Process any directive (active block)
    ///     process_all_directives(line);
    /// } else {
    ///     // Only process conditional directives to track nesting
    ///     if is_conditional_directive(line) {
    ///         process_conditional_directive(line);
    ///     }
    ///     // Skip all other directives
    /// }
    /// ```
    #[inline]
    pub fn should_process_directive(&self) -> bool {
        self.is_active()
    }
}

impl Default for ConditionalStack {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::source_map::{FileId, SourceLocation};

    // -- Test Helpers -------------------------------------------------------

    /// Creates a test `SourceLocation` at the given line number.
    /// Uses a fixed file_id and byte_offset for simplicity.
    fn test_loc(line: u32) -> SourceLocation {
        SourceLocation {
            file_id: FileId(0),
            byte_offset: 0,
            line,
            column: 1,
        }
    }

    /// Creates a `DiagnosticEmitter` for testing.
    fn test_emitter() -> DiagnosticEmitter {
        DiagnosticEmitter::new()
    }

    // -- Empty Stack Tests --------------------------------------------------

    #[test]
    fn test_empty_stack_is_active() {
        let stack = ConditionalStack::new();
        assert!(stack.is_active());
    }

    #[test]
    fn test_empty_stack_depth_zero() {
        let stack = ConditionalStack::new();
        assert_eq!(stack.depth(), 0);
    }

    #[test]
    fn test_empty_stack_not_in_conditional() {
        let stack = ConditionalStack::new();
        assert!(!stack.is_in_conditional());
    }

    #[test]
    fn test_empty_stack_should_process_directive() {
        let stack = ConditionalStack::new();
        assert!(stack.should_process_directive());
    }

    #[test]
    fn test_default_is_new() {
        let stack = ConditionalStack::default();
        assert!(stack.is_active());
        assert_eq!(stack.depth(), 0);
    }

    // -- push_if(true) Tests ------------------------------------------------

    #[test]
    fn test_push_if_true_is_active() {
        let mut stack = ConditionalStack::new();
        stack.push_if(true, test_loc(1));
        assert!(stack.is_active());
    }

    #[test]
    fn test_push_if_true_depth_one() {
        let mut stack = ConditionalStack::new();
        stack.push_if(true, test_loc(1));
        assert_eq!(stack.depth(), 1);
    }

    #[test]
    fn test_push_if_true_in_conditional() {
        let mut stack = ConditionalStack::new();
        stack.push_if(true, test_loc(1));
        assert!(stack.is_in_conditional());
    }

    // -- push_if(false) Tests -----------------------------------------------

    #[test]
    fn test_push_if_false_is_inactive() {
        let mut stack = ConditionalStack::new();
        stack.push_if(false, test_loc(1));
        assert!(!stack.is_active());
    }

    #[test]
    fn test_push_if_false_should_not_process_directive() {
        let mut stack = ConditionalStack::new();
        stack.push_if(false, test_loc(1));
        assert!(!stack.should_process_directive());
    }

    // -- push_if(false) + process_else() Tests ------------------------------

    #[test]
    fn test_if_false_else_is_active() {
        let mut stack = ConditionalStack::new();
        let mut diag = test_emitter();
        stack.push_if(false, test_loc(1));
        assert!(!stack.is_active());
        stack.process_else(test_loc(3), &mut diag).unwrap();
        assert!(stack.is_active());
    }

    // -- push_if(true) + process_else() Tests -------------------------------

    #[test]
    fn test_if_true_else_is_inactive() {
        let mut stack = ConditionalStack::new();
        let mut diag = test_emitter();
        stack.push_if(true, test_loc(1));
        assert!(stack.is_active());
        stack.process_else(test_loc(3), &mut diag).unwrap();
        // Branch was already taken, so #else is inactive.
        assert!(!stack.is_active());
    }

    // -- Nested Conditional Tests -------------------------------------------

    #[test]
    fn test_nested_true_false_inner_inactive() {
        let mut stack = ConditionalStack::new();
        stack.push_if(true, test_loc(1));
        assert!(stack.is_active());
        stack.push_if(false, test_loc(2));
        assert!(!stack.is_active());
        assert_eq!(stack.depth(), 2);
    }

    #[test]
    fn test_nested_false_true_inner_parent_inactive() {
        // #if 0
        //   #if 1
        //     // This should NOT be active despite #if 1
        //   #endif
        // #endif
        let mut stack = ConditionalStack::new();
        stack.push_if(false, test_loc(1));
        assert!(!stack.is_active());
        // Even though condition is true, parent is inactive.
        stack.push_if(true, test_loc(2));
        assert!(!stack.is_active());
    }

    #[test]
    fn test_nested_false_false_inner_parent_inactive() {
        let mut stack = ConditionalStack::new();
        stack.push_if(false, test_loc(1));
        stack.push_if(false, test_loc(2));
        assert!(!stack.is_active());
    }

    // -- #elif Chain Tests --------------------------------------------------

    #[test]
    fn test_elif_first_true_wins() {
        // #if 0
        // #elif 0
        // #elif 1  ← this should be active
        // #elif 1  ← this should be inactive (already taken)
        // #else    ← this should be inactive
        // #endif
        let mut stack = ConditionalStack::new();
        let mut diag = test_emitter();

        stack.push_if(false, test_loc(1));
        assert!(!stack.is_active());

        stack.process_elif(false, test_loc(2), &mut diag).unwrap();
        assert!(!stack.is_active());

        stack.process_elif(true, test_loc(3), &mut diag).unwrap();
        assert!(stack.is_active()); // first true condition wins

        stack.process_elif(true, test_loc(4), &mut diag).unwrap();
        assert!(!stack.is_active()); // already taken

        stack.process_else(test_loc(5), &mut diag).unwrap();
        assert!(!stack.is_active()); // already taken
    }

    #[test]
    fn test_elif_after_active_if_is_inactive() {
        // #if 1
        //   // active
        // #elif 1
        //   // inactive (previous branch was taken)
        // #endif
        let mut stack = ConditionalStack::new();
        let mut diag = test_emitter();

        stack.push_if(true, test_loc(1));
        assert!(stack.is_active());

        stack.process_elif(true, test_loc(3), &mut diag).unwrap();
        assert!(!stack.is_active());
    }

    #[test]
    fn test_elif_false_after_false_if_remains_inactive() {
        let mut stack = ConditionalStack::new();
        let mut diag = test_emitter();

        stack.push_if(false, test_loc(1));
        assert!(!stack.is_active());

        stack.process_elif(false, test_loc(2), &mut diag).unwrap();
        assert!(!stack.is_active());
    }

    // -- pop_endif Tests ----------------------------------------------------

    #[test]
    fn test_pop_endif_restores_active() {
        let mut stack = ConditionalStack::new();
        let mut diag = test_emitter();

        stack.push_if(true, test_loc(1));
        stack.push_if(false, test_loc(2));
        assert!(!stack.is_active());

        stack.pop_endif(test_loc(3), &mut diag).unwrap();
        assert!(stack.is_active()); // back to outer active
        assert_eq!(stack.depth(), 1);

        stack.pop_endif(test_loc(4), &mut diag).unwrap();
        assert!(stack.is_active()); // back to top-level
        assert_eq!(stack.depth(), 0);
    }

    #[test]
    fn test_pop_endif_on_empty_stack_is_error() {
        let mut stack = ConditionalStack::new();
        let mut diag = test_emitter();

        let result = stack.pop_endif(test_loc(1), &mut diag);
        assert!(result.is_err());
        assert!(diag.has_errors());
        assert_eq!(diag.error_count(), 1);
    }

    // -- Error Condition Tests ----------------------------------------------

    #[test]
    fn test_elif_without_if_is_error() {
        let mut stack = ConditionalStack::new();
        let mut diag = test_emitter();

        let result = stack.process_elif(true, test_loc(1), &mut diag);
        assert!(result.is_err());
        assert!(diag.has_errors());
    }

    #[test]
    fn test_elif_after_else_is_error() {
        let mut stack = ConditionalStack::new();
        let mut diag = test_emitter();

        stack.push_if(false, test_loc(1));
        stack.process_else(test_loc(2), &mut diag).unwrap();
        let result = stack.process_elif(true, test_loc(3), &mut diag);
        assert!(result.is_err());
        assert!(diag.has_errors());
    }

    #[test]
    fn test_else_without_if_is_error() {
        let mut stack = ConditionalStack::new();
        let mut diag = test_emitter();

        let result = stack.process_else(test_loc(1), &mut diag);
        assert!(result.is_err());
        assert!(diag.has_errors());
    }

    #[test]
    fn test_else_after_else_is_error() {
        let mut stack = ConditionalStack::new();
        let mut diag = test_emitter();

        stack.push_if(true, test_loc(1));
        stack.process_else(test_loc(2), &mut diag).unwrap();
        let result = stack.process_else(test_loc(3), &mut diag);
        assert!(result.is_err());
        assert!(diag.has_errors());
    }

    // -- Unterminated Conditional Tests -------------------------------------

    #[test]
    fn test_check_unterminated_empty_stack_ok() {
        let stack = ConditionalStack::new();
        let mut diag = test_emitter();
        let result = stack.check_unterminated(&mut diag);
        assert!(result.is_ok());
        assert!(!diag.has_errors());
    }

    #[test]
    fn test_check_unterminated_with_open_if_is_error() {
        let mut stack = ConditionalStack::new();
        let mut diag = test_emitter();

        stack.push_if(true, test_loc(5));
        let result = stack.check_unterminated(&mut diag);
        assert!(result.is_err());
        assert!(diag.has_errors());
        assert_eq!(diag.error_count(), 1);
    }

    #[test]
    fn test_check_unterminated_multiple_open_ifs() {
        let mut stack = ConditionalStack::new();
        let mut diag = test_emitter();

        stack.push_if(true, test_loc(1));
        stack.push_if(false, test_loc(5));
        stack.push_if(true, test_loc(10));

        let result = stack.check_unterminated(&mut diag);
        assert!(result.is_err());
        // One error per unterminated conditional.
        assert_eq!(diag.error_count(), 3);
    }

    // -- Deep Nesting Tests -------------------------------------------------

    #[test]
    fn test_deeply_nested_conditionals() {
        let mut stack = ConditionalStack::new();
        let mut diag = test_emitter();
        let depth = 50;

        // Push 50 nested #if true
        for i in 0..depth {
            stack.push_if(true, test_loc(i as u32 + 1));
            assert!(stack.is_active());
            assert_eq!(stack.depth(), i + 1);
        }

        // Pop them all
        for i in (0..depth).rev() {
            stack.pop_endif(test_loc(100 + i as u32), &mut diag).unwrap();
            assert!(stack.is_active());
            assert_eq!(stack.depth(), i);
        }

        assert_eq!(stack.depth(), 0);
        assert!(!stack.is_in_conditional());
    }

    #[test]
    fn test_deeply_nested_inactive_parent() {
        let mut stack = ConditionalStack::new();
        let mut diag = test_emitter();

        // #if 0 at the outermost level
        stack.push_if(false, test_loc(1));
        assert!(!stack.is_active());

        // Nest 20 levels of #if 1 inside #if 0 — all should be ParentInactive
        for i in 0..20 {
            stack.push_if(true, test_loc(i as u32 + 2));
            assert!(!stack.is_active());
        }

        // Pop all 20 inner levels
        for _ in 0..20 {
            stack.pop_endif(test_loc(100), &mut diag).unwrap();
            assert!(!stack.is_active());
        }

        // Pop the outermost #if 0
        stack.pop_endif(test_loc(101), &mut diag).unwrap();
        assert!(stack.is_active());
        assert_eq!(stack.depth(), 0);
    }

    // -- Complex Scenario Tests ---------------------------------------------

    #[test]
    fn test_complex_elif_chain_with_nesting() {
        // #if 0
        //   // inactive
        //   #if 1
        //     // inactive (parent inactive)
        //   #endif
        // #elif 1
        //   // active
        //   #if 0
        //     // inactive
        //   #else
        //     // active
        //   #endif
        // #else
        //   // inactive (already taken)
        // #endif
        let mut stack = ConditionalStack::new();
        let mut diag = test_emitter();

        // #if 0
        stack.push_if(false, test_loc(1));
        assert!(!stack.is_active());

        // Nested #if 1 inside #if 0
        stack.push_if(true, test_loc(3));
        assert!(!stack.is_active()); // parent is inactive

        // #endif for inner
        stack.pop_endif(test_loc(5), &mut diag).unwrap();
        assert!(!stack.is_active()); // still in #if 0

        // #elif 1
        stack.process_elif(true, test_loc(7), &mut diag).unwrap();
        assert!(stack.is_active());

        // Nested #if 0 inside active #elif
        stack.push_if(false, test_loc(9));
        assert!(!stack.is_active());

        // #else inside nested #if 0
        stack.process_else(test_loc(11), &mut diag).unwrap();
        assert!(stack.is_active());

        // #endif for inner
        stack.pop_endif(test_loc(13), &mut diag).unwrap();
        assert!(stack.is_active());

        // #else (already taken)
        stack.process_else(test_loc(15), &mut diag).unwrap();
        assert!(!stack.is_active());

        // #endif for outer
        stack.pop_endif(test_loc(17), &mut diag).unwrap();
        assert!(stack.is_active());
        assert_eq!(stack.depth(), 0);
        assert!(!diag.has_errors());
    }

    #[test]
    fn test_empty_if_group() {
        // #if 1
        // #endif
        let mut stack = ConditionalStack::new();
        let mut diag = test_emitter();

        stack.push_if(true, test_loc(1));
        assert!(stack.is_active());
        stack.pop_endif(test_loc(2), &mut diag).unwrap();
        assert!(stack.is_active());
        assert_eq!(stack.depth(), 0);
    }

    #[test]
    fn test_elif_in_parent_inactive() {
        // #if 0
        //   #if 0
        //   #elif 1  ← still ParentInactive
        //   #endif
        // #endif
        let mut stack = ConditionalStack::new();
        let mut diag = test_emitter();

        stack.push_if(false, test_loc(1));
        stack.push_if(false, test_loc(2));
        assert!(!stack.is_active());

        stack.process_elif(true, test_loc(3), &mut diag).unwrap();
        assert!(!stack.is_active()); // parent is inactive

        stack.pop_endif(test_loc(4), &mut diag).unwrap();
        stack.pop_endif(test_loc(5), &mut diag).unwrap();
        assert!(stack.is_active());
    }

    #[test]
    fn test_else_in_parent_inactive() {
        // #if 0
        //   #if 0
        //   #else  ← still ParentInactive
        //   #endif
        // #endif
        let mut stack = ConditionalStack::new();
        let mut diag = test_emitter();

        stack.push_if(false, test_loc(1));
        stack.push_if(false, test_loc(2));
        assert!(!stack.is_active());

        stack.process_else(test_loc(3), &mut diag).unwrap();
        assert!(!stack.is_active()); // parent is inactive

        stack.pop_endif(test_loc(4), &mut diag).unwrap();
        stack.pop_endif(test_loc(5), &mut diag).unwrap();
        assert!(stack.is_active());
    }

    #[test]
    fn test_is_active_through_multiple_nesting_levels() {
        let mut stack = ConditionalStack::new();
        let mut diag = test_emitter();

        // Level 1: active
        stack.push_if(true, test_loc(1));
        assert!(stack.is_active());

        // Level 2: active
        stack.push_if(true, test_loc(2));
        assert!(stack.is_active());

        // Level 3: inactive
        stack.push_if(false, test_loc(3));
        assert!(!stack.is_active());

        // Level 4: parent inactive (despite condition true)
        stack.push_if(true, test_loc(4));
        assert!(!stack.is_active());

        // Pop level 4
        stack.pop_endif(test_loc(5), &mut diag).unwrap();
        assert!(!stack.is_active()); // level 3 still inactive

        // Pop level 3
        stack.pop_endif(test_loc(6), &mut diag).unwrap();
        assert!(stack.is_active()); // level 2 is active

        // Pop level 2
        stack.pop_endif(test_loc(7), &mut diag).unwrap();
        assert!(stack.is_active()); // level 1 is active

        // Pop level 1
        stack.pop_endif(test_loc(8), &mut diag).unwrap();
        assert!(stack.is_active()); // top-level
    }

    #[test]
    fn test_all_false_elif_chain_with_else() {
        // #if 0
        // #elif 0
        // #elif 0
        // #else  ← should be active
        // #endif
        let mut stack = ConditionalStack::new();
        let mut diag = test_emitter();

        stack.push_if(false, test_loc(1));
        assert!(!stack.is_active());

        stack.process_elif(false, test_loc(2), &mut diag).unwrap();
        assert!(!stack.is_active());

        stack.process_elif(false, test_loc(3), &mut diag).unwrap();
        assert!(!stack.is_active());

        stack.process_else(test_loc(4), &mut diag).unwrap();
        assert!(stack.is_active()); // no branch was taken, so #else activates

        stack.pop_endif(test_loc(5), &mut diag).unwrap();
        assert!(stack.is_active());
    }

    #[test]
    fn test_multiple_errors_accumulated() {
        let mut stack = ConditionalStack::new();
        let mut diag = test_emitter();

        // Multiple errors on empty stack
        let _ = stack.pop_endif(test_loc(1), &mut diag);
        let _ = stack.process_else(test_loc(2), &mut diag);
        let _ = stack.process_elif(true, test_loc(3), &mut diag);

        assert_eq!(diag.error_count(), 3);
    }

    #[test]
    fn test_depth_tracking_through_operations() {
        let mut stack = ConditionalStack::new();
        let mut diag = test_emitter();

        assert_eq!(stack.depth(), 0);

        stack.push_if(true, test_loc(1));
        assert_eq!(stack.depth(), 1);

        stack.push_if(false, test_loc(2));
        assert_eq!(stack.depth(), 2);

        // #elif does not change depth
        stack.process_elif(true, test_loc(3), &mut diag).unwrap();
        assert_eq!(stack.depth(), 2);

        // #else does not change depth
        stack.process_else(test_loc(4), &mut diag).unwrap();
        assert_eq!(stack.depth(), 2);

        stack.pop_endif(test_loc(5), &mut diag).unwrap();
        assert_eq!(stack.depth(), 1);

        stack.pop_endif(test_loc(6), &mut diag).unwrap();
        assert_eq!(stack.depth(), 0);
    }
}
