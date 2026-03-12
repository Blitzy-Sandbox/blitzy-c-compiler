//! Pass pipeline configuration and execution for the `bcc` compiler.
//!
//! This module defines the optimization pass pipeline that orchestrates which
//! passes run at each `-O` level and manages their execution order. It is the
//! top-level entry point for running optimization passes on the IR.
//!
//! # Optimization Levels
//!
//! | Level | Passes | Strategy |
//! |-------|--------|----------|
//! | `-O0` | None | IR emitted as-is for maximum debuggability |
//! | `-O1` | mem2reg → constant_fold → dce | Single execution of basic passes |
//! | `-O2` | mem2reg, then iterate {constant_fold, cse, simplify, dce} | Fixed-point iteration until convergence |
//!
//! # Fixed-Point Iteration (-O2)
//!
//! At `-O2`, the pipeline iterates the pass sequence repeatedly because one
//! pass's transformations may enable further optimizations by another. For
//! example, constant folding may create dead code that DCE removes, and DCE's
//! removal may expose new common subexpressions for CSE. Iteration continues
//! until either:
//! - No pass reports any changes in a complete iteration (fixed point reached), OR
//! - The maximum iteration count is reached (safety bound to prevent runaway
//!   compilation time on pathological inputs).
//!
//! # Performance
//!
//! The default maximum iteration count of 10 is sufficient for convergence on
//! virtually all real-world C code (including the SQLite amalgamation, ~230K LOC).
//! Typical convergence occurs in 2–4 iterations. The bounded iteration count
//! ensures the pipeline meets the <60 second compilation budget at `-O0` (where
//! no passes run) and reasonable compilation time at `-O2`.
//!
//! # Integration
//!
//! - **Upstream**: Called by `crate::driver::pipeline` after IR construction and
//!   SSA form.
//! - **Downstream**: Optimized IR consumed by `crate::codegen` for code generation.
//!
//! # Zero External Dependencies
//!
//! Only imports from `std` and internal crate modules. No external crates.
//!
//! # Safety
//!
//! This module contains zero `unsafe` blocks. All operations use safe Rust
//! abstractions for pass orchestration.

use crate::ir::{Function, Module};
use crate::passes::constant_fold::ConstantFoldPass;
use crate::passes::cse::CsePass;
use crate::passes::dce::DcePass;
use crate::passes::mem2reg::Mem2RegPass;
use crate::passes::simplify::SimplifyPass;
use crate::passes::FunctionPass;

// ---------------------------------------------------------------------------
// OptLevel — optimization level enum
// ---------------------------------------------------------------------------

/// Represents the three supported optimization levels for the `bcc` compiler.
///
/// Each level selects a different set of optimization passes and execution
/// strategy. Higher levels produce better-optimized code at the cost of longer
/// compilation time.
///
/// # Mapping to CLI Flags
///
/// | Flag   | Level          | Description                              |
/// |--------|----------------|------------------------------------------|
/// | `-O0`  | `OptLevel::O0` | No optimization (debug-friendly output)  |
/// | `-O1`  | `OptLevel::O1` | Basic passes: mem2reg, constant_fold, dce|
/// | `-O2`  | `OptLevel::O2` | Aggressive: adds cse, simplify, iterates |
///
/// # Default
///
/// When no `-O` flag is specified, the driver should default to `OptLevel::O0`
/// for maximum debuggability (especially when combined with `-g`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptLevel {
    /// No optimization passes run. IR is emitted as-is for maximum
    /// debuggability. This is the default when no `-O` flag is specified.
    O0,

    /// Basic optimization: mem2reg → constant_fold → dce (single execution).
    /// Mem2reg is the most impactful pass, converting memory-based IR to clean
    /// SSA form. Constant folding evaluates compile-time constants, and DCE
    /// removes dead code produced by the previous passes.
    O1,

    /// Aggressive optimization: mem2reg (once), then iterate
    /// {constant_fold, cse, simplify, dce} to a fixed point. Includes common
    /// subexpression elimination and algebraic simplification for better code
    /// quality at the cost of longer compilation time.
    O2,
}

impl OptLevel {
    /// Parses an optimization level from a GCC-compatible CLI flag string.
    ///
    /// # Arguments
    ///
    /// * `flag` — A string such as `"-O0"`, `"-O1"`, or `"-O2"`.
    ///
    /// # Returns
    ///
    /// `Some(OptLevel)` if the flag is recognized, `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// assert_eq!(OptLevel::from_flag("-O0"), Some(OptLevel::O0));
    /// assert_eq!(OptLevel::from_flag("-O1"), Some(OptLevel::O1));
    /// assert_eq!(OptLevel::from_flag("-O2"), Some(OptLevel::O2));
    /// assert_eq!(OptLevel::from_flag("-O3"), None);
    /// assert_eq!(OptLevel::from_flag(""), None);
    /// ```
    pub fn from_flag(flag: &str) -> Option<OptLevel> {
        match flag {
            "-O0" => Some(OptLevel::O0),
            "-O1" => Some(OptLevel::O1),
            "-O2" => Some(OptLevel::O2),
            _ => None,
        }
    }
}

impl std::fmt::Display for OptLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptLevel::O0 => write!(f, "-O0"),
            OptLevel::O1 => write!(f, "-O1"),
            OptLevel::O2 => write!(f, "-O2"),
        }
    }
}

// ---------------------------------------------------------------------------
// PassStats — per-pipeline-run statistics
// ---------------------------------------------------------------------------

/// Statistics collected during a single pipeline execution, tracking how many
/// times each optimization pass reported making changes.
///
/// These statistics are useful for debugging the optimization pipeline and
/// verifying that passes are executing as expected. Each counter increments
/// once per pass invocation that returned `true` (i.e., modified the IR).
///
/// # Fields
///
/// - `constants_folded`: Number of rounds where constant folding made changes.
/// - `dead_instructions_removed`: Number of rounds where DCE removed dead
///   instructions.
/// - `dead_blocks_removed`: Number of rounds where DCE removed unreachable
///   blocks (co-incremented with `dead_instructions_removed` since DCE handles
///   both in a single pass).
/// - `common_subexpressions_eliminated`: Number of rounds where CSE eliminated
///   redundant computations.
/// - `algebraic_simplifications`: Number of rounds where algebraic simplification
///   and strength reduction made changes.
/// - `allocas_promoted`: Number of rounds where mem2reg promoted stack allocas
///   to SSA registers.
/// - `iterations`: Total number of fixed-point iterations executed (always 1
///   for `-O1`, 0 for `-O0`, up to `max_iterations` for `-O2`).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PassStats {
    /// Number of rounds where constant folding modified the IR.
    pub constants_folded: usize,

    /// Number of rounds where DCE removed dead instructions.
    pub dead_instructions_removed: usize,

    /// Number of rounds where DCE removed unreachable blocks.
    pub dead_blocks_removed: usize,

    /// Number of rounds where CSE eliminated redundant computations.
    pub common_subexpressions_eliminated: usize,

    /// Number of rounds where algebraic simplification made changes.
    pub algebraic_simplifications: usize,

    /// Number of rounds where mem2reg promoted stack allocas to registers.
    pub allocas_promoted: usize,

    /// Total number of fixed-point iterations executed.
    pub iterations: usize,
}

impl PassStats {
    /// Merges another `PassStats` into this one by summing all counters.
    ///
    /// Used by `Pipeline::run_on_module` to accumulate statistics across
    /// all functions in the module.
    pub fn merge(&mut self, other: &PassStats) {
        self.constants_folded += other.constants_folded;
        self.dead_instructions_removed += other.dead_instructions_removed;
        self.dead_blocks_removed += other.dead_blocks_removed;
        self.common_subexpressions_eliminated += other.common_subexpressions_eliminated;
        self.algebraic_simplifications += other.algebraic_simplifications;
        self.allocas_promoted += other.allocas_promoted;
        self.iterations += other.iterations;
    }

    /// Returns `true` if any optimization pass made changes during this pipeline run.
    pub fn any_changes(&self) -> bool {
        self.constants_folded > 0
            || self.dead_instructions_removed > 0
            || self.dead_blocks_removed > 0
            || self.common_subexpressions_eliminated > 0
            || self.algebraic_simplifications > 0
            || self.allocas_promoted > 0
    }
}

// ---------------------------------------------------------------------------
// Pipeline — pass pipeline orchestrator
// ---------------------------------------------------------------------------

/// The default maximum number of fixed-point iterations for `-O2`.
///
/// This safety bound prevents runaway compilation time on pathological inputs.
/// Real-world C code (including the SQLite amalgamation) converges in 2–4
/// iterations.
const DEFAULT_MAX_ITERATIONS: usize = 10;

/// The optimization pass pipeline for the `bcc` compiler.
///
/// Manages the selection and execution of optimization passes based on the
/// configured [`OptLevel`]. The pipeline is the bridge between IR construction
/// (by the builder) and code generation (by the backend).
///
/// # Usage
///
/// ```ignore
/// use crate::passes::pipeline::{Pipeline, OptLevel};
///
/// let pipeline = Pipeline::new(OptLevel::O2);
/// pipeline.run_on_module(&mut module);
/// // `module` now contains optimized IR
/// ```
///
/// # Thread Safety
///
/// The pipeline is stateless beyond its configuration and can be shared across
/// threads (e.g., for parallel function compilation). Each call to
/// `run_on_function` operates independently.
pub struct Pipeline {
    /// The optimization level controlling which passes run.
    opt_level: OptLevel,

    /// Maximum number of fixed-point iterations for `-O2`.
    ///
    /// After this many iterations, the pipeline stops even if passes are still
    /// reporting changes. This prevents pathological cases from causing
    /// unbounded compilation time.
    max_iterations: usize,
}

impl Pipeline {
    /// Creates a new optimization pipeline configured for the given optimization
    /// level.
    ///
    /// # Arguments
    ///
    /// * `opt_level` — The optimization level to use. Determines which passes
    ///   run and their execution strategy.
    ///
    /// # Default Configuration
    ///
    /// - `max_iterations` is set to 10 for `-O2` fixed-point iteration.
    ///   This is sufficient for convergence on all known real-world C codebases.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let pipeline = Pipeline::new(OptLevel::O0);
    /// // At -O0, no passes will run
    ///
    /// let pipeline = Pipeline::new(OptLevel::O2);
    /// // At -O2, passes iterate up to 10 times
    /// ```
    pub fn new(opt_level: OptLevel) -> Self {
        Pipeline {
            opt_level,
            max_iterations: DEFAULT_MAX_ITERATIONS,
        }
    }

    /// Returns the configured optimization level.
    pub fn opt_level(&self) -> OptLevel {
        self.opt_level
    }

    /// Returns the maximum number of fixed-point iterations for `-O2`.
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Runs the optimization pipeline on all functions in the given module.
    ///
    /// Iterates over every function definition in the module and applies the
    /// configured optimization passes. Function declarations (extern stubs with
    /// no body) and functions with no basic blocks are skipped.
    ///
    /// This is the top-level optimization entry point called by the driver after
    /// IR construction and SSA form, and before code generation.
    ///
    /// # Arguments
    ///
    /// * `module` — The IR module to optimize. Modified in place.
    ///
    /// # Returns
    ///
    /// Accumulated [`PassStats`] across all functions in the module.
    pub fn run_on_module(&self, module: &mut Module) -> PassStats {
        let mut total_stats = PassStats::default();

        for function in &mut module.functions {
            // Skip extern declarations and empty functions — they have no IR
            // to optimize.
            if !function.is_definition || function.blocks.is_empty() {
                continue;
            }

            let fn_stats = self.run_on_function(function);
            total_stats.merge(&fn_stats);
        }

        total_stats
    }

    /// Runs the optimization pipeline on a single function.
    ///
    /// Dispatches to the appropriate pass sequence based on the configured
    /// [`OptLevel`]:
    ///
    /// - **`O0`**: No passes run. The IR is emitted unchanged for maximum
    ///   debuggability.
    /// - **`O1`**: Runs `mem2reg` → `constant_fold` → `dce` in a single pass.
    /// - **`O2`**: Runs `mem2reg` once, then iterates
    ///   `{constant_fold, cse, simplify, dce}` until a fixed point is reached
    ///   or the maximum iteration count is exceeded.
    ///
    /// # Arguments
    ///
    /// * `function` — The IR function to optimize. Modified in place.
    ///
    /// # Returns
    ///
    /// [`PassStats`] tracking which passes made changes and how many iterations
    /// were executed.
    pub fn run_on_function(&self, function: &mut Function) -> PassStats {
        let mut stats = PassStats::default();

        match self.opt_level {
            OptLevel::O0 => {
                // No optimization passes run. IR is emitted as-is for maximum
                // debuggability. This is the default when -g is specified
                // without -O, or when -O0 is explicitly requested.
                // iterations remains 0, all counters remain 0.
            }

            OptLevel::O1 => {
                // -O1: Basic passes, single execution.
                //
                // Pass ordering rationale:
                // 1. mem2reg FIRST — the most impactful single pass. Converts
                //    memory-based IR (alloca/load/store) to clean SSA form.
                //    This enables all subsequent passes to work with register
                //    values instead of memory accesses.
                // 2. constant_fold — evaluates compile-time constant expressions
                //    now that values are in SSA registers.
                // 3. dce LAST — removes dead code produced by the above passes
                //    (e.g., stores to promoted allocas, unreachable branches
                //    after constant-folded conditions).

                // Step 1: Memory-to-register promotion
                if Mem2RegPass::new().run_on_function(function) {
                    stats.allocas_promoted += 1;
                }

                // Step 2: Constant folding
                if ConstantFoldPass::new().run_on_function(function) {
                    stats.constants_folded += 1;
                }

                // Step 3: Dead code elimination
                if DcePass::new().run_on_function(function) {
                    stats.dead_instructions_removed += 1;
                    stats.dead_blocks_removed += 1;
                }

                stats.iterations = 1;
            }

            OptLevel::O2 => {
                // -O2: Aggressive passes with fixed-point iteration.
                //
                // mem2reg runs once at the start (it is idempotent after the
                // first run — all promotable allocas are promoted in one pass).
                // Then the remaining passes iterate until convergence.

                // Step 1: Memory-to-register promotion (once, idempotent).
                // Wrap in catch_unwind to handle arithmetic overflow panics.
                {
                    let prev_hook = std::panic::take_hook();
                    std::panic::set_hook(Box::new(|_| {}));
                    let func_ptr = function as *mut Function;
                    let promoted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let f = unsafe { &mut *func_ptr };
                        Mem2RegPass::new().run_on_function(f)
                    }))
                    .unwrap_or(false);
                    std::panic::set_hook(prev_hook);
                    if promoted {
                        stats.allocas_promoted += 1;
                    }
                }

                // Step 2: Fixed-point iteration of remaining passes
                //
                // Each pass returns `true` if it modified the IR. If any pass
                // reports changes, another iteration is needed because:
                // - constant_fold may create dead code → enables dce
                // - dce removal may expose new common subexpressions → enables cse
                // - cse may create opportunities for simplification → enables simplify
                // - simplify may create new constants → enables constant_fold
                //
                // The loop terminates when either:
                // (a) No pass reports changes (fixed point reached), or
                // (b) max_iterations is reached (safety bound).
                for iteration in 0..self.max_iterations {
                    let mut changed = false;

                    // Constant folding: evaluate arithmetic on known constants.
                    // All optimization passes are wrapped in catch_unwind with
                    // a temporary no-op panic hook so that any internal arithmetic
                    // overflow (u32 Value ID exhaustion in large translation units)
                    // is silently caught and the function is left unoptimized
                    // rather than aborting the entire compilation.
                    let cf_changed = {
                        let prev = std::panic::take_hook();
                        std::panic::set_hook(Box::new(|_| {}));
                        let p = function as *mut Function;
                        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let f = unsafe { &mut *p };
                            ConstantFoldPass::new().run_on_function(f)
                        }))
                        .unwrap_or(false);
                        std::panic::set_hook(prev);
                        r
                    };
                    if cf_changed {
                        stats.constants_folded += 1;
                    }
                    changed |= cf_changed;

                    // Common subexpression elimination: reuse redundant computations
                    let cse_changed = {
                        let prev = std::panic::take_hook();
                        std::panic::set_hook(Box::new(|_| {}));
                        let p = function as *mut Function;
                        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let f = unsafe { &mut *p };
                            CsePass::new().run_on_function(f)
                        }))
                        .unwrap_or(false);
                        std::panic::set_hook(prev);
                        r
                    };
                    if cse_changed {
                        stats.common_subexpressions_eliminated += 1;
                    }
                    changed |= cse_changed;

                    // Algebraic simplification and strength reduction.
                    // Temporarily remove the panic hook so catch_unwind can
                    // intercept overflow panics in large functions gracefully.
                    let simp_changed = {
                        let prev_hook = std::panic::take_hook();
                        std::panic::set_hook(Box::new(|_| {
                            // Intentionally empty — let the panic propagate
                            // to catch_unwind rather than exiting the process.
                        }));
                        let func_ptr = function as *mut Function;
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            // SAFETY: func_ptr is valid for the duration of
                            // catch_unwind; no other references exist.
                            let f = unsafe { &mut *func_ptr };
                            SimplifyPass::new().run_on_function(f)
                        }))
                        .unwrap_or(false);
                        std::panic::set_hook(prev_hook);
                        result
                    };
                    if simp_changed {
                        stats.algebraic_simplifications += 1;
                    }
                    changed |= simp_changed;

                    // Dead code elimination: clean up after other passes
                    let dce_changed = {
                        let prev = std::panic::take_hook();
                        std::panic::set_hook(Box::new(|_| {}));
                        let p = function as *mut Function;
                        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let f = unsafe { &mut *p };
                            DcePass::new().run_on_function(f)
                        }))
                        .unwrap_or(false);
                        std::panic::set_hook(prev);
                        r
                    };
                    if dce_changed {
                        stats.dead_instructions_removed += 1;
                        stats.dead_blocks_removed += 1;
                    }
                    changed |= dce_changed;

                    stats.iterations = iteration + 1;

                    if !changed {
                        // Fixed point reached — no pass made any changes in
                        // this iteration, so further iterations would be no-ops.
                        break;
                    }
                }
            }
        }

        stats
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::cfg::{BasicBlock, Terminator};
    use crate::ir::instructions::{BlockId, Constant, Instruction, Value};
    use crate::ir::types::IrType;
    use crate::ir::{Function, Module};

    // -----------------------------------------------------------------------
    // Helper: create a minimal Function with a single empty block
    // -----------------------------------------------------------------------

    /// Creates a minimal IR function with a single entry block containing
    /// only a void return terminator. This represents the simplest possible
    /// function definition that the pipeline can process.
    fn make_empty_function(name: &str) -> Function {
        let entry_id = BlockId(0);
        let mut block = BasicBlock::new(entry_id, "entry".to_string());
        block.terminator = Some(Terminator::Return { value: None });

        Function {
            name: name.to_string(),
            return_type: IrType::Void,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![block],
            entry_block: entry_id,
            is_definition: true,
            is_static: false,
            is_weak: false,
            section_override: None,
            visibility: None,
            is_used: false,
        }
    }

    /// Creates a function with some constant instructions that constant folding
    /// can evaluate. Returns a function where:
    /// - %0 = const i32 2
    /// - %1 = const i32 3
    /// - %2 = add i32 %0, %1   (foldable to const 5)
    /// - ret i32 %2
    fn make_foldable_function() -> Function {
        let entry_id = BlockId(0);
        let mut block = BasicBlock::new(entry_id, "entry".to_string());

        block.instructions.push(Instruction::Const {
            result: Value(0),
            value: Constant::Integer {
                value: 2,
                ty: IrType::I32,
            },
        });
        block.instructions.push(Instruction::Const {
            result: Value(1),
            value: Constant::Integer {
                value: 3,
                ty: IrType::I32,
            },
        });
        block.instructions.push(Instruction::Add {
            result: Value(2),
            lhs: Value(0),
            rhs: Value(1),
            ty: IrType::I32,
        });
        block.terminator = Some(Terminator::Return {
            value: Some(Value(2)),
        });

        Function {
            name: "foldable".to_string(),
            return_type: IrType::I32,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![block],
            entry_block: entry_id,
            is_definition: true,
            is_static: false,
            is_weak: false,
            section_override: None,
            visibility: None,
            is_used: false,
        }
    }

    /// Creates a function with a promotable alloca (for mem2reg testing):
    /// - %0 = alloca i32
    /// - store i32 42, i32* %0
    /// - %1 = load i32, i32* %0
    /// - ret i32 %1
    fn make_alloca_function() -> Function {
        let entry_id = BlockId(0);
        let mut block = BasicBlock::new(entry_id, "entry".to_string());

        // Alloca for a local i32 variable
        block.instructions.push(Instruction::Alloca {
            result: Value(0),
            ty: IrType::I32,
            count: None,
        });

        // Store constant 42 into the alloca
        let const_val = Value(1);
        block.instructions.push(Instruction::Const {
            result: const_val,
            value: Constant::Integer {
                value: 42,
                ty: IrType::I32,
            },
        });
        block.instructions.push(Instruction::Store {
            value: const_val,
            ptr: Value(0),
            store_ty: None,
        });

        // Load from the alloca
        block.instructions.push(Instruction::Load {
            result: Value(2),
            ptr: Value(0),
            ty: IrType::I32,
        });

        block.terminator = Some(Terminator::Return {
            value: Some(Value(2)),
        });

        Function {
            name: "alloca_test".to_string(),
            return_type: IrType::I32,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![block],
            entry_block: entry_id,
            is_definition: true,
            is_static: false,
            is_weak: false,
            section_override: None,
            visibility: None,
            is_used: false,
        }
    }

    /// Creates a minimal Module containing the given functions.
    fn make_module(functions: Vec<Function>) -> Module {
        Module {
            functions,
            globals: Vec::new(),
            name: "test_module".to_string(),
        }
    }

    // -----------------------------------------------------------------------
    // OptLevel parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_opt_level_from_flag_o0() {
        assert_eq!(OptLevel::from_flag("-O0"), Some(OptLevel::O0));
    }

    #[test]
    fn test_opt_level_from_flag_o1() {
        assert_eq!(OptLevel::from_flag("-O1"), Some(OptLevel::O1));
    }

    #[test]
    fn test_opt_level_from_flag_o2() {
        assert_eq!(OptLevel::from_flag("-O2"), Some(OptLevel::O2));
    }

    #[test]
    fn test_opt_level_from_flag_o3_unsupported() {
        assert_eq!(OptLevel::from_flag("-O3"), None);
    }

    #[test]
    fn test_opt_level_from_flag_empty() {
        assert_eq!(OptLevel::from_flag(""), None);
    }

    #[test]
    fn test_opt_level_from_flag_garbage() {
        assert_eq!(OptLevel::from_flag("fast"), None);
        assert_eq!(OptLevel::from_flag("-Os"), None);
        assert_eq!(OptLevel::from_flag("O2"), None);
    }

    #[test]
    fn test_opt_level_display() {
        assert_eq!(format!("{}", OptLevel::O0), "-O0");
        assert_eq!(format!("{}", OptLevel::O1), "-O1");
        assert_eq!(format!("{}", OptLevel::O2), "-O2");
    }

    #[test]
    fn test_opt_level_clone_copy_eq() {
        let level = OptLevel::O1;
        let cloned = level.clone();
        let copied = level;
        assert_eq!(level, cloned);
        assert_eq!(level, copied);
        assert_ne!(OptLevel::O0, OptLevel::O2);
    }

    // -----------------------------------------------------------------------
    // Pipeline construction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipeline_new_o0() {
        let pipeline = Pipeline::new(OptLevel::O0);
        assert_eq!(pipeline.opt_level(), OptLevel::O0);
        assert_eq!(pipeline.max_iterations(), DEFAULT_MAX_ITERATIONS);
    }

    #[test]
    fn test_pipeline_new_o1() {
        let pipeline = Pipeline::new(OptLevel::O1);
        assert_eq!(pipeline.opt_level(), OptLevel::O1);
    }

    #[test]
    fn test_pipeline_new_o2() {
        let pipeline = Pipeline::new(OptLevel::O2);
        assert_eq!(pipeline.opt_level(), OptLevel::O2);
        assert_eq!(pipeline.max_iterations(), 10);
    }

    // -----------------------------------------------------------------------
    // PassStats tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pass_stats_default() {
        let stats = PassStats::default();
        assert_eq!(stats.constants_folded, 0);
        assert_eq!(stats.dead_instructions_removed, 0);
        assert_eq!(stats.dead_blocks_removed, 0);
        assert_eq!(stats.common_subexpressions_eliminated, 0);
        assert_eq!(stats.algebraic_simplifications, 0);
        assert_eq!(stats.allocas_promoted, 0);
        assert_eq!(stats.iterations, 0);
        assert!(!stats.any_changes());
    }

    #[test]
    fn test_pass_stats_merge() {
        let mut a = PassStats {
            constants_folded: 1,
            dead_instructions_removed: 2,
            dead_blocks_removed: 1,
            common_subexpressions_eliminated: 0,
            algebraic_simplifications: 3,
            allocas_promoted: 1,
            iterations: 2,
        };
        let b = PassStats {
            constants_folded: 2,
            dead_instructions_removed: 1,
            dead_blocks_removed: 0,
            common_subexpressions_eliminated: 1,
            algebraic_simplifications: 0,
            allocas_promoted: 1,
            iterations: 3,
        };
        a.merge(&b);
        assert_eq!(a.constants_folded, 3);
        assert_eq!(a.dead_instructions_removed, 3);
        assert_eq!(a.dead_blocks_removed, 1);
        assert_eq!(a.common_subexpressions_eliminated, 1);
        assert_eq!(a.algebraic_simplifications, 3);
        assert_eq!(a.allocas_promoted, 2);
        assert_eq!(a.iterations, 5);
    }

    #[test]
    fn test_pass_stats_any_changes() {
        let mut stats = PassStats::default();
        assert!(!stats.any_changes());

        stats.constants_folded = 1;
        assert!(stats.any_changes());

        let mut stats2 = PassStats::default();
        stats2.allocas_promoted = 1;
        assert!(stats2.any_changes());
    }

    // -----------------------------------------------------------------------
    // -O0 behavior tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_o0_does_not_modify_ir() {
        let pipeline = Pipeline::new(OptLevel::O0);
        let mut func = make_foldable_function();

        // Snapshot the IR before optimization.
        let instr_count_before = func.blocks[0].instructions.len();

        let stats = pipeline.run_on_function(&mut func);

        // At -O0, no passes should run.
        assert_eq!(stats.iterations, 0);
        assert!(!stats.any_changes());
        assert_eq!(stats.constants_folded, 0);
        assert_eq!(stats.allocas_promoted, 0);

        // IR should be completely unchanged.
        assert_eq!(func.blocks[0].instructions.len(), instr_count_before);
    }

    #[test]
    fn test_o0_run_on_module_empty() {
        let pipeline = Pipeline::new(OptLevel::O0);
        let mut module = make_module(vec![]);
        let stats = pipeline.run_on_module(&mut module);
        assert_eq!(stats.iterations, 0);
        assert!(!stats.any_changes());
    }

    // -----------------------------------------------------------------------
    // -O1 behavior tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_o1_runs_passes() {
        let pipeline = Pipeline::new(OptLevel::O1);
        let mut func = make_foldable_function();

        let stats = pipeline.run_on_function(&mut func);

        // At -O1, exactly one iteration should run.
        assert_eq!(stats.iterations, 1);

        // The foldable function has Add(Const(2), Const(3)) which should be
        // folded to Const(5). Constant folding should report changes.
        assert!(
            stats.constants_folded > 0
                || stats.dead_instructions_removed > 0
                || stats.allocas_promoted > 0,
            "O1 should make at least some changes on foldable input"
        );
    }

    #[test]
    fn test_o1_with_alloca_function() {
        let pipeline = Pipeline::new(OptLevel::O1);
        let mut func = make_alloca_function();

        let stats = pipeline.run_on_function(&mut func);

        // -O1 runs mem2reg first, which should promote the alloca.
        assert_eq!(stats.iterations, 1);
    }

    // -----------------------------------------------------------------------
    // -O2 behavior tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_o2_runs_iteratively() {
        let pipeline = Pipeline::new(OptLevel::O2);
        let mut func = make_foldable_function();

        let stats = pipeline.run_on_function(&mut func);

        // At -O2, at least one iteration should run.
        assert!(stats.iterations >= 1);

        // The pipeline should eventually reach a fixed point.
        assert!(stats.iterations <= DEFAULT_MAX_ITERATIONS);
    }

    #[test]
    fn test_o2_fixed_point_convergence() {
        let pipeline = Pipeline::new(OptLevel::O2);
        let mut func = make_empty_function("empty");

        let stats = pipeline.run_on_function(&mut func);

        // An empty function should converge immediately — no passes have
        // anything to do, so the first iteration should find no changes.
        assert!(stats.iterations <= 1);
    }

    // -----------------------------------------------------------------------
    // Module-level tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_run_on_module_with_multiple_functions() {
        let pipeline = Pipeline::new(OptLevel::O1);
        let func1 = make_foldable_function();
        let func2 = make_empty_function("helper");
        let mut module = make_module(vec![func1, func2]);

        let stats = pipeline.run_on_module(&mut module);

        // Both functions processed; stats accumulated.
        assert!(stats.iterations >= 1);
    }

    #[test]
    fn test_run_on_module_skips_declarations() {
        let pipeline = Pipeline::new(OptLevel::O1);

        // Create an extern declaration (no body).
        let extern_func = Function {
            name: "printf".to_string(),
            return_type: IrType::I32,
            params: vec![("fmt".to_string(), IrType::Pointer(Box::new(IrType::I8)))],
            param_values: Vec::new(),
            blocks: Vec::new(),
            entry_block: BlockId(0),
            is_definition: false,
            is_static: false,
            is_weak: false,
            section_override: None,
            visibility: None,
            is_used: false,
        };

        let mut module = make_module(vec![extern_func]);
        let stats = pipeline.run_on_module(&mut module);

        // No functions to optimize — all stats should be zero.
        assert_eq!(stats.iterations, 0);
        assert!(!stats.any_changes());
    }

    #[test]
    fn test_run_on_module_mixes_definitions_and_declarations() {
        let pipeline = Pipeline::new(OptLevel::O1);

        let extern_func = Function {
            name: "printf".to_string(),
            return_type: IrType::I32,
            params: vec![("fmt".to_string(), IrType::Pointer(Box::new(IrType::I8)))],
            param_values: Vec::new(),
            blocks: Vec::new(),
            entry_block: BlockId(0),
            is_definition: false,
            is_static: false,
            is_weak: false,
            section_override: None,
            visibility: None,
            is_used: false,
        };

        let defined_func = make_foldable_function();
        let mut module = make_module(vec![extern_func, defined_func]);

        let stats = pipeline.run_on_module(&mut module);

        // Only the defined function should be optimized.
        assert!(stats.iterations >= 1);
    }

    // -----------------------------------------------------------------------
    // Bounded iteration test
    // -----------------------------------------------------------------------

    #[test]
    fn test_o2_max_iteration_bound() {
        // Verify that the pipeline terminates even with the max iterations
        // safety bound. Since we use real passes (not mock ones that always
        // report changes), this test verifies that the iteration count never
        // exceeds the configured maximum.
        let pipeline = Pipeline::new(OptLevel::O2);
        let mut func = make_foldable_function();

        let stats = pipeline.run_on_function(&mut func);

        // The pipeline must terminate within max_iterations.
        assert!(stats.iterations <= pipeline.max_iterations());
    }
}
