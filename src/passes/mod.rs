//! # Optimization Passes Module
//!
//! This module implements the optimization pipeline for the `bcc` C compiler,
//! supporting `-O0` through `-O2` optimization levels. Each pass is independently
//! testable and operates on the SSA-form IR defined in `crate::ir`.
//!
//! ## Pass Architecture
//!
//! All optimization passes implement the [`FunctionPass`] trait, which defines:
//! - A [`name()`](FunctionPass::name) method returning a human-readable pass name
//! - A [`run_on_function()`](FunctionPass::run_on_function) method that transforms
//!   the IR in place and returns whether any changes were made (used for
//!   fixed-point iteration at `-O2`)
//!
//! For future extensibility, the [`ModulePass`] trait provides an analogous
//! interface for passes that operate on the entire compilation module (all
//! functions and globals). Currently all optimization passes are
//! [`FunctionPass`] instances.
//!
//! ## Optimization Levels
//!
//! - **`-O0`**: No passes run. IR is emitted directly for maximum debuggability.
//! - **`-O1`**: Basic passes: `mem2reg` → `constant_fold` → `dce`
//! - **`-O2`**: Aggressive: `mem2reg`, then iterate `constant_fold` → `cse` →
//!   `simplify` → `dce` until fixed point
//!
//! ## Data Flow
//!
//! ```text
//! SSA IR (from ir::builder) → Pipeline → Optimized IR (to codegen)
//! ```
//!
//! ## Submodules
//!
//! | Submodule          | Contents                                                       |
//! |--------------------|----------------------------------------------------------------|
//! | [`constant_fold`]  | Compile-time constant arithmetic evaluation                    |
//! | [`dce`]            | Dead code and unreachable block elimination                    |
//! | [`cse`]            | Common subexpression elimination via value numbering            |
//! | [`simplify`]       | Algebraic simplification and strength reduction                |
//! | [`mem2reg`]        | Memory-to-register (alloca) promotion into SSA registers       |
//! | [`pipeline`]       | Pass pipeline configuration per `-O` level and orchestration   |

use crate::ir::{Function, Module};

// ---------------------------------------------------------------------------
// Submodule declarations
// ---------------------------------------------------------------------------

/// Constant folding pass — evaluates arithmetic, comparison, logical, bitwise,
/// and cast operations on known compile-time constants, replacing expressions
/// like `Add(Const(2), Const(3))` with `Const(5)`. Also folds conditional
/// branches and switches with constant conditions.
pub mod constant_fold;

/// Dead code elimination pass — removes unreachable basic blocks (via BFS
/// reachability from the entry block) and dead instructions (computations
/// whose result values have zero uses and no side effects).
pub mod dce;

/// Common subexpression elimination pass — uses hash-based value numbering
/// to detect and eliminate redundant computations by reusing previously
/// computed values, both within and across basic blocks (local and global CSE).
pub mod cse;

/// Algebraic simplification and strength reduction pass — applies identity
/// removal (`x + 0 → x`), zero absorption (`x * 0 → 0`), self-operation
/// rules (`x - x → 0`), strength reduction (`x * 2 → x << 1`), constant
/// propagation, boolean simplification, and cast elimination.
pub mod simplify;

/// Memory-to-register promotion pass — converts stack-allocated local
/// variables (`Alloca` + `Load` + `Store`) into SSA registers with phi nodes
/// at control-flow join points. Only promotes allocas whose address is never
/// taken. The single most impactful optimization in the pipeline.
pub mod mem2reg;

/// Pass pipeline configuration — defines which optimization passes run at
/// each `-O` level (`O0`, `O1`, `O2`) and orchestrates their execution,
/// including fixed-point iteration at `-O2`.
pub mod pipeline;

// ---------------------------------------------------------------------------
// FunctionPass trait — the core per-function optimization interface
// ---------------------------------------------------------------------------

/// Trait for optimization passes that operate on individual functions.
///
/// Each pass receives a mutable reference to a [`Function`] and transforms
/// its IR in place. The [`run_on_function`](FunctionPass::run_on_function)
/// method returns `true` if any changes were made, enabling fixed-point
/// iteration at `-O2`.
///
/// # Object Safety
///
/// This trait is object-safe and can be used as `dyn FunctionPass` for
/// dynamic dispatch in pass pipelines.
///
/// # Examples
///
/// ```ignore
/// use crate::passes::FunctionPass;
///
/// struct MyPass;
///
/// impl FunctionPass for MyPass {
///     fn name(&self) -> &str { "my_pass" }
///     fn run_on_function(&mut self, function: &mut Function) -> bool {
///         // transform function's IR...
///         false // no changes made
///     }
/// }
/// ```
pub trait FunctionPass {
    /// Returns a human-readable name for this pass (used in diagnostics and
    /// debugging output).
    fn name(&self) -> &str;

    /// Runs this optimization pass on the given function.
    ///
    /// # Arguments
    ///
    /// * `function` — The IR function to optimize. Modified in place.
    ///
    /// # Returns
    ///
    /// `true` if the IR was modified by this pass, `false` if no changes were
    /// made. The return value is critical for the `-O2` fixed-point iteration
    /// loop: if no pass reports changes in a full iteration, the loop
    /// terminates.
    fn run_on_function(&mut self, function: &mut Function) -> bool;
}

// ---------------------------------------------------------------------------
// ModulePass trait — the module-level optimization interface
// ---------------------------------------------------------------------------

/// Trait for optimization passes that operate on the entire compilation module.
///
/// Module-level passes can see all functions and globals, enabling
/// interprocedural analysis if needed in the future. Currently, all
/// optimization passes are [`FunctionPass`] instances, but this trait is
/// defined for architectural extensibility.
///
/// # Object Safety
///
/// This trait is object-safe and can be used as `dyn ModulePass` for
/// dynamic dispatch.
///
/// # Examples
///
/// ```ignore
/// use crate::passes::ModulePass;
///
/// struct MyModulePass;
///
/// impl ModulePass for MyModulePass {
///     fn name(&self) -> &str { "my_module_pass" }
///     fn run_on_module(&mut self, module: &mut Module) -> bool {
///         // transform module's IR...
///         false
///     }
/// }
/// ```
pub trait ModulePass {
    /// Returns a human-readable name for this pass.
    fn name(&self) -> &str;

    /// Runs this optimization pass on the given module.
    ///
    /// # Arguments
    ///
    /// * `module` — The IR module to optimize. Modified in place.
    ///
    /// # Returns
    ///
    /// `true` if the module was modified by this pass, `false` otherwise.
    fn run_on_module(&mut self, module: &mut Module) -> bool;
}

// ---------------------------------------------------------------------------
// Public re-exports — the passes API surface
// ---------------------------------------------------------------------------
// Downstream consumers (driver, codegen) import from `crate::passes` rather
// than reaching into submodules directly.

/// Pipeline configuration and execution.
pub use pipeline::{OptLevel, PassStats, Pipeline};

/// Individual pass types for direct instantiation.
pub use constant_fold::ConstantFoldPass;
pub use cse::CsePass;
pub use dce::DcePass;
pub use mem2reg::Mem2RegPass;
pub use simplify::SimplifyPass;

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::cfg::{BasicBlock, Terminator};
    use crate::ir::instructions::BlockId;
    use crate::ir::types::IrType;
    use crate::ir::{Function, Module};

    // -----------------------------------------------------------------------
    // FunctionPass trait tests
    // -----------------------------------------------------------------------

    /// Verifies that the `FunctionPass` trait is object-safe — it can be used
    /// behind a `dyn FunctionPass` trait object. This is a compile-time check:
    /// if `FunctionPass` had methods returning `Self` or using generic type
    /// parameters, this test would fail to compile.
    #[test]
    fn test_function_pass_is_object_safe() {
        struct TestPass;
        impl FunctionPass for TestPass {
            fn name(&self) -> &str {
                "test"
            }
            fn run_on_function(&mut self, _function: &mut Function) -> bool {
                false
            }
        }

        // Verify object safety — this compiles only if FunctionPass is object-safe.
        let pass: Box<dyn FunctionPass> = Box::new(TestPass);
        assert_eq!(pass.name(), "test");
    }

    /// Verifies that a `FunctionPass` returning `true` correctly signals that
    /// changes were made to the IR.
    #[test]
    fn test_function_pass_changed_flag() {
        struct AlwaysChanges;
        impl FunctionPass for AlwaysChanges {
            fn name(&self) -> &str {
                "always_changes"
            }
            fn run_on_function(&mut self, _function: &mut Function) -> bool {
                true
            }
        }

        struct NeverChanges;
        impl FunctionPass for NeverChanges {
            fn name(&self) -> &str {
                "never_changes"
            }
            fn run_on_function(&mut self, _function: &mut Function) -> bool {
                false
            }
        }

        let entry_id = BlockId(0);
        let mut block = BasicBlock::new(entry_id, "entry".to_string());
        block.terminator = Some(Terminator::Return { value: None });

        let mut func = Function {
            name: "test_fn".to_string(),
            return_type: IrType::Void,
            params: Vec::new(),
            blocks: vec![block],
            entry_block: entry_id,
            is_definition: true,
        };

        let mut pass_a = AlwaysChanges;
        assert!(pass_a.run_on_function(&mut func));

        let mut pass_b = NeverChanges;
        assert!(!pass_b.run_on_function(&mut func));
    }

    /// Verifies that multiple `FunctionPass` implementations can be stored
    /// in a `Vec<Box<dyn FunctionPass>>` for dynamic pipeline construction.
    #[test]
    fn test_function_pass_dynamic_dispatch() {
        struct PassA;
        impl FunctionPass for PassA {
            fn name(&self) -> &str {
                "pass_a"
            }
            fn run_on_function(&mut self, _function: &mut Function) -> bool {
                false
            }
        }

        struct PassB;
        impl FunctionPass for PassB {
            fn name(&self) -> &str {
                "pass_b"
            }
            fn run_on_function(&mut self, _function: &mut Function) -> bool {
                false
            }
        }

        let passes: Vec<Box<dyn FunctionPass>> = vec![Box::new(PassA), Box::new(PassB)];
        assert_eq!(passes.len(), 2);
        assert_eq!(passes[0].name(), "pass_a");
        assert_eq!(passes[1].name(), "pass_b");
    }

    // -----------------------------------------------------------------------
    // ModulePass trait tests
    // -----------------------------------------------------------------------

    /// Verifies that the `ModulePass` trait is object-safe.
    #[test]
    fn test_module_pass_is_object_safe() {
        struct TestModulePass;
        impl ModulePass for TestModulePass {
            fn name(&self) -> &str {
                "test_module"
            }
            fn run_on_module(&mut self, _module: &mut Module) -> bool {
                false
            }
        }

        let pass: Box<dyn ModulePass> = Box::new(TestModulePass);
        assert_eq!(pass.name(), "test_module");
    }

    /// Verifies that a `ModulePass` can iterate functions within a module.
    #[test]
    fn test_module_pass_iterates_functions() {
        struct CountFunctions {
            count: usize,
        }
        impl ModulePass for CountFunctions {
            fn name(&self) -> &str {
                "count_functions"
            }
            fn run_on_module(&mut self, module: &mut Module) -> bool {
                self.count = module.functions.len();
                false
            }
        }

        let mut module = Module::new("test_module".to_string());

        // Add a function to the module.
        let entry_id = BlockId(0);
        let mut block = BasicBlock::new(entry_id, "entry".to_string());
        block.terminator = Some(Terminator::Return { value: None });

        module.functions.push(Function {
            name: "func_a".to_string(),
            return_type: IrType::Void,
            params: Vec::new(),
            blocks: vec![block],
            entry_block: entry_id,
            is_definition: true,
        });

        let mut pass = CountFunctions { count: 0 };
        pass.run_on_module(&mut module);
        assert_eq!(pass.count, 1);
    }

    // -----------------------------------------------------------------------
    // Re-export verification tests
    // -----------------------------------------------------------------------

    /// Verifies that `ConstantFoldPass` is accessible via the re-export.
    #[test]
    fn test_constant_fold_pass_reexport() {
        let pass = ConstantFoldPass::new();
        assert_eq!(pass.name(), "constant_fold");
    }

    /// Verifies that `DcePass` is accessible via the re-export.
    #[test]
    fn test_dce_pass_reexport() {
        let pass = DcePass::new();
        assert_eq!(pass.name(), "dce");
    }

    /// Verifies that `CsePass` is accessible via the re-export.
    #[test]
    fn test_cse_pass_reexport() {
        let pass = CsePass::new();
        assert_eq!(pass.name(), "cse");
    }

    /// Verifies that `SimplifyPass` is accessible via the re-export.
    #[test]
    fn test_simplify_pass_reexport() {
        let pass = SimplifyPass::new();
        assert_eq!(pass.name(), "simplify");
    }

    /// Verifies that `Mem2RegPass` is accessible via the re-export.
    #[test]
    fn test_mem2reg_pass_reexport() {
        let pass = Mem2RegPass::new();
        assert_eq!(pass.name(), "mem2reg");
    }

    /// Verifies that `OptLevel` is accessible via the re-export and that
    /// `from_flag` parses correctly.
    #[test]
    fn test_opt_level_reexport() {
        assert_eq!(OptLevel::from_flag("-O0"), Some(OptLevel::O0));
        assert_eq!(OptLevel::from_flag("-O1"), Some(OptLevel::O1));
        assert_eq!(OptLevel::from_flag("-O2"), Some(OptLevel::O2));
        assert_eq!(OptLevel::from_flag("-O3"), None);
        assert_eq!(OptLevel::from_flag(""), None);
    }

    /// Verifies that `Pipeline` is accessible via the re-export.
    #[test]
    fn test_pipeline_reexport() {
        let pipeline = Pipeline::new(OptLevel::O0);
        assert_eq!(pipeline.opt_level(), OptLevel::O0);
    }

    /// Verifies that `PassStats` is accessible via the re-export and defaults
    /// to zero.
    #[test]
    fn test_pass_stats_reexport() {
        let stats = PassStats::default();
        assert_eq!(stats.iterations, 0);
        assert!(!stats.any_changes());
    }

    // -----------------------------------------------------------------------
    // All concrete passes implement FunctionPass
    // -----------------------------------------------------------------------

    /// Verifies that all five concrete pass types implement `FunctionPass`
    /// and can be collected into a trait object vector — confirming they all
    /// conform to the pipeline's pass interface.
    #[test]
    fn test_all_passes_implement_function_pass() {
        let passes: Vec<Box<dyn FunctionPass>> = vec![
            Box::new(ConstantFoldPass::new()),
            Box::new(DcePass::new()),
            Box::new(CsePass::new()),
            Box::new(SimplifyPass::new()),
            Box::new(Mem2RegPass::new()),
        ];

        let expected_names = ["constant_fold", "dce", "cse", "simplify", "mem2reg"];
        for (pass, expected) in passes.iter().zip(expected_names.iter()) {
            assert_eq!(pass.name(), *expected);
        }
    }
}
