//! Dead Code Elimination (DCE) optimization pass for the `bcc` compiler.
//!
//! This module implements [`DcePass`], which performs two complementary analyses
//! to remove provably useless code from the IR:
//!
//! 1. **Unreachable Block Elimination** — Breadth-first search from the function's
//!    entry block identifies all reachable blocks. Blocks not visited during the
//!    BFS are cleared in place (instructions removed, terminator set to
//!    `Unreachable`, edges severed) to preserve the `blocks[id.0]` index invariant
//!    used by other compiler phases.
//!
//! 2. **Dead Instruction and Phi Node Elimination** — A use-count map
//!    (`HashMap<Value, usize>`) tracks how many times each SSA value appears as an
//!    operand across all instructions, phi nodes, and terminators. Instructions
//!    whose result value has zero uses and that have no side effects are removed.
//!    Because removing a dead instruction can reduce the use counts of its
//!    operands, elimination is performed iteratively until a fixed point is
//!    reached (no further removals possible).
//!
//! # Correctness Invariants
//!
//! - **Side-effect preservation**: `Store` and `Call` instructions are **never**
//!   removed, even if their result value (if any) is unused, because they may
//!   write to memory or perform I/O.
//! - **CFG well-formedness**: After unreachable block removal, predecessor lists
//!   and phi-node incoming edges in surviving blocks are cleaned up so that no
//!   stale references to removed blocks remain.
//! - **Index invariant**: Blocks are cleared in place (not removed from the Vec)
//!   so that `function.blocks[block_id.0 as usize]` continues to work for
//!   downstream passes and code generation.
//! - **Terminator operands**: Values used by terminators (e.g., the condition in
//!   `CondBranch`, the value in `Return`) are counted as uses, preventing
//!   premature elimination of values consumed only by terminators.
//!
//! # Performance
//!
//! The BFS reachability pass is O(V + E) where V is the number of blocks and E
//! is the number of CFG edges. The dead instruction pass is O(I × K) where I is
//! the total instruction count and K is the number of elimination rounds (bounded
//! by the longest dependency chain of dead instructions). For typical C code, K
//! is small (single digits), making the overall cost effectively linear.
//!
//! # Integration
//!
//! Called by the pass pipeline at `-O1` (after `mem2reg` and `constant_fold`) and
//! at `-O2` (within the fixed-point iteration loop). DCE typically runs last in
//! the pass sequence to clean up dead code produced by preceding passes.
//!
//! # Zero External Dependencies
//!
//! Only imports from `std` and internal crate modules. No external crates.
//!
//! # Safety
//!
//! This module contains zero `unsafe` blocks. All operations use safe Rust
//! abstractions for graph traversal and instruction filtering.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::ir::builder::Function;
use crate::ir::cfg::{BasicBlock, PhiNode, Terminator};
use crate::ir::instructions::{BlockId, Instruction, Value};

use super::FunctionPass;

// ---------------------------------------------------------------------------
// DcePass — Dead Code Elimination pass
// ---------------------------------------------------------------------------

/// Dead Code Elimination optimization pass.
///
/// Removes unreachable basic blocks (blocks with no path from the function's
/// entry block) and dead instructions (computations whose result values are
/// never used and that have no observable side effects).
///
/// # Examples
///
/// ```ignore
/// use crate::passes::{FunctionPass, DcePass};
///
/// let mut pass = DcePass::new();
/// let changed = pass.run_on_function(&mut function);
/// if changed {
///     println!("DCE removed dead code");
/// }
/// ```
pub struct DcePass;

impl DcePass {
    /// Creates a new DCE pass instance.
    ///
    /// The pass is stateless — all analysis is performed fresh on each
    /// invocation of [`run_on_function`].
    pub fn new() -> Self {
        DcePass
    }

    // -----------------------------------------------------------------------
    // Helper: extract operand Values from a terminator
    // -----------------------------------------------------------------------

    /// Extracts all SSA operand [`Value`]s consumed by the given terminator.
    ///
    /// Terminators do not define result values, but some variants consume SSA
    /// values as operands:
    /// - `CondBranch` uses its `condition` value.
    /// - `Return` uses its optional return `value`.
    /// - `Switch` uses its discriminant `value`.
    /// - `Branch` and `Unreachable` consume no SSA values.
    ///
    /// These operand values must be included in use-count computation to prevent
    /// the dead instruction pass from incorrectly eliminating values that are
    /// only consumed by terminators.
    fn terminator_operands(term: &Terminator) -> Vec<Value> {
        match term {
            Terminator::Branch { .. } => vec![],
            Terminator::CondBranch { condition, .. } => vec![*condition],
            Terminator::Return { value: Some(v) } => vec![*v],
            Terminator::Return { value: None } => vec![],
            Terminator::Switch { value, .. } => vec![*value],
            Terminator::Unreachable => vec![],
        }
    }

    // -----------------------------------------------------------------------
    // Helper: BFS reachability from entry block
    // -----------------------------------------------------------------------

    /// Computes the set of basic block IDs reachable from the function's entry
    /// block using breadth-first search.
    ///
    /// The entry block is always included in the result (assuming the function
    /// has blocks). Any block not in the returned set is unreachable and can be
    /// safely eliminated.
    ///
    /// # Algorithm
    ///
    /// A standard BFS using a [`VecDeque`] as a FIFO worklist. Each block is
    /// visited at most once (tracked via the returned [`HashSet`]). For each
    /// visited block, all successors are enqueued.
    fn compute_reachable(function: &Function) -> HashSet<BlockId> {
        let mut reachable = HashSet::new();
        let mut worklist = VecDeque::new();

        // Guard: empty functions have no reachable blocks.
        if function.blocks.is_empty() {
            return reachable;
        }

        worklist.push_back(function.entry_block);

        while let Some(block_id) = worklist.pop_front() {
            if reachable.insert(block_id) {
                // Look up the block by its ID index in the blocks vector.
                let idx = block_id.0 as usize;
                if idx < function.blocks.len() {
                    for &succ in &function.blocks[idx].successors {
                        worklist.push_back(succ);
                    }
                }
            }
        }

        reachable
    }

    // -----------------------------------------------------------------------
    // Helper: use-count computation
    // -----------------------------------------------------------------------

    /// Builds a map counting how many times each SSA [`Value`] is used as an
    /// operand across all instructions, phi nodes, and terminators in the
    /// function.
    ///
    /// The returned map keys are SSA values that appear as operands; values
    /// not used anywhere will have no entry (treated as zero uses).
    fn compute_use_counts(function: &Function) -> HashMap<Value, usize> {
        let mut uses: HashMap<Value, usize> = HashMap::new();

        for block in &function.blocks {
            // Count uses in phi node incoming values.
            for phi in &block.phi_nodes {
                for &(val, _) in &phi.incoming {
                    *uses.entry(val).or_insert(0) += 1;
                }
            }

            // Count uses in regular instruction operands.
            for inst in &block.instructions {
                for operand in inst.operands() {
                    *uses.entry(operand).or_insert(0) += 1;
                }
            }

            // Count uses in terminator operands.
            if let Some(ref term) = block.terminator {
                for operand in Self::terminator_operands(term) {
                    *uses.entry(operand).or_insert(0) += 1;
                }
            }
        }

        uses
    }

    // -----------------------------------------------------------------------
    // Phase 1: Unreachable block elimination
    // -----------------------------------------------------------------------

    /// Removes unreachable basic blocks from the function.
    ///
    /// Unreachable blocks are those with no path from the function's entry block.
    /// They are cleared in place (instructions and phi nodes removed, terminator
    /// set to [`Terminator::Unreachable`], edges severed) to preserve the index
    /// invariant where `function.blocks[block_id.0 as usize]` corresponds to
    /// the block with that ID.
    ///
    /// After clearing unreachable blocks, the predecessor lists and phi-node
    /// incoming edges in surviving reachable blocks are cleaned up to remove
    /// stale references.
    ///
    /// # Returns
    ///
    /// `true` if any blocks were cleared (i.e., unreachable blocks existed),
    /// `false` if all blocks were already reachable.
    fn remove_unreachable_blocks(&self, function: &mut Function) -> bool {
        if function.blocks.is_empty() {
            return false;
        }

        let reachable = Self::compute_reachable(function);

        // Identify which block IDs are unreachable.
        let removed: HashSet<BlockId> = function
            .blocks
            .iter()
            .map(|b| b.id)
            .filter(|id| !reachable.contains(id))
            .collect();

        if removed.is_empty() {
            return false;
        }

        // Phase 1a: Clear unreachable blocks in place.
        // We preserve the Vec positions to maintain the index invariant.
        for block in &mut function.blocks {
            if removed.contains(&block.id) {
                block.instructions.clear();
                block.phi_nodes.clear();
                block.terminator = Some(Terminator::Unreachable);
                block.predecessors.clear();
                block.successors.clear();
            }
        }

        // Phase 1b: Clean up predecessor lists and phi nodes in reachable blocks.
        // Reachable blocks may have had unreachable predecessors in their lists.
        for block in &mut function.blocks {
            if reachable.contains(&block.id) {
                // Remove unreachable blocks from predecessor lists.
                block.predecessors.retain(|p| !removed.contains(p));

                // Remove phi node incoming edges from unreachable blocks.
                for phi in &mut block.phi_nodes {
                    phi.incoming.retain(|&(_, pred)| !removed.contains(&pred));
                }
            }
        }

        true
    }

    // -----------------------------------------------------------------------
    // Phase 2: Dead instruction and phi node elimination
    // -----------------------------------------------------------------------

    /// Removes dead instructions and dead phi nodes from the function.
    ///
    /// An instruction is considered *dead* if all three conditions hold:
    /// 1. It produces a result value (`result()` returns `Some`).
    /// 2. That result value has zero uses across all instructions, phi nodes,
    ///    and terminators.
    /// 3. The instruction has no side effects (`has_side_effects()` returns
    ///    `false`).
    ///
    /// A phi node is considered *dead* if its result value has zero uses.
    ///
    /// Removal is iterative: when a dead instruction is removed, its operands'
    /// use counts are decremented, potentially making other instructions dead.
    /// The loop continues until no further removals occur (fixed-point
    /// convergence).
    ///
    /// # Side-Effect Classification
    ///
    /// - `Store` — writes to memory, must be preserved.
    /// - `Call` — may have side effects (I/O, memory writes), must be preserved.
    /// - `Nop` — has no effect and produces no result, eligible for removal
    ///   (though it has no result, so it's kept by the `result()` check).
    /// - All other instructions — preserved only if their result is used.
    ///
    /// # Returns
    ///
    /// `true` if any instructions or phi nodes were removed, `false` if
    /// everything was already live.
    fn remove_dead_instructions(&self, function: &mut Function) -> bool {
        let mut any_removed = false;

        loop {
            // Recompute use counts from scratch each iteration.
            // This is simpler and more robust than incremental maintenance,
            // and the cost is acceptable (linear in function size).
            let mut use_counts = Self::compute_use_counts(function);
            let mut removed_in_pass = false;

            for block in &mut function.blocks {
                // ---- Dead regular instruction removal ----
                block.instructions.retain(|inst| {
                    if let Some(result) = inst.result() {
                        if use_counts.get(&result).copied().unwrap_or(0) == 0
                            && !inst.has_side_effects()
                        {
                            // Decrement use counts for this instruction's operands.
                            // This enables cascading elimination in subsequent
                            // iterations: operands that just lost their last use
                            // become dead candidates.
                            for operand in inst.operands() {
                                if let Some(count) = use_counts.get_mut(&operand) {
                                    *count = count.saturating_sub(1);
                                }
                            }
                            removed_in_pass = true;
                            return false; // Remove this instruction.
                        }
                    }
                    true // Keep this instruction.
                });

                // ---- Dead phi node removal ----
                block.phi_nodes.retain(|phi| {
                    if use_counts.get(&phi.result).copied().unwrap_or(0) == 0 {
                        // Decrement use counts for incoming values.
                        for &(val, _) in &phi.incoming {
                            if let Some(count) = use_counts.get_mut(&val) {
                                *count = count.saturating_sub(1);
                            }
                        }
                        removed_in_pass = true;
                        return false; // Remove this phi node.
                    }
                    true // Keep this phi node.
                });
            }

            if removed_in_pass {
                any_removed = true;
            } else {
                break;
            }
        }

        any_removed
    }
}

// ---------------------------------------------------------------------------
// FunctionPass trait implementation
// ---------------------------------------------------------------------------

impl FunctionPass for DcePass {
    /// Returns the human-readable name of this pass: `"dce"`.
    fn name(&self) -> &str {
        "dce"
    }

    /// Runs dead code elimination on the given function.
    ///
    /// Executes two phases in sequence:
    /// 1. Unreachable block elimination (BFS mark-sweep).
    /// 2. Dead instruction and phi node elimination (iterative use-count analysis).
    ///
    /// # Returns
    ///
    /// `true` if any blocks, instructions, or phi nodes were removed (the IR was
    /// modified), `false` if the function was already fully live.
    fn run_on_function(&mut self, function: &mut Function) -> bool {
        let mut changed = false;
        changed |= self.remove_unreachable_blocks(function);
        changed |= self.remove_dead_instructions(function);
        changed
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::Function;
    use crate::ir::cfg::{BasicBlock, PhiNode, Terminator};
    use crate::ir::instructions::{
        BlockId, Callee, Constant, Instruction, Value,
    };
    use crate::ir::types::IrType;

    // -----------------------------------------------------------------------
    // Test Helpers
    // -----------------------------------------------------------------------

    /// Creates a minimal test function with the given blocks and entry block.
    fn make_function(blocks: Vec<BasicBlock>, entry: BlockId) -> Function {
        Function {
            name: "test_fn".to_string(),
            return_type: IrType::Void,
            params: vec![],
            blocks,
            entry_block: entry,
            is_definition: true,
        }
    }

    /// Creates a basic block with the given ID, label, and terminator.
    /// Predecessors and successors must be set manually after creation.
    fn make_block(id: u32, label: &str, terminator: Terminator) -> BasicBlock {
        BasicBlock {
            id: BlockId(id),
            label: label.to_string(),
            phi_nodes: Vec::new(),
            instructions: Vec::new(),
            terminator: Some(terminator),
            predecessors: Vec::new(),
            successors: Vec::new(),
        }
    }

    /// Links two blocks: adds `to` as successor of `from` and `from` as predecessor of `to`.
    fn link_blocks(blocks: &mut [BasicBlock], from_idx: usize, to_idx: usize) {
        let to_id = blocks[to_idx].id;
        let from_id = blocks[from_idx].id;
        blocks[from_idx].successors.push(to_id);
        blocks[to_idx].predecessors.push(from_id);
    }

    // -----------------------------------------------------------------------
    // Unreachable Block Removal Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_entry_block_nothing_removed() {
        // Single block function — entry is the only block, nothing to remove.
        let blocks = vec![make_block(
            0,
            "entry",
            Terminator::Return { value: None },
        )];
        let mut func = make_function(blocks, BlockId(0));

        let mut pass = DcePass::new();
        let changed = pass.run_on_function(&mut func);

        // No unreachable blocks, no dead instructions.
        assert!(!changed);
        assert_eq!(func.blocks.len(), 1);
        assert!(func.blocks[0].terminator.is_some());
    }

    #[test]
    fn test_linear_chain_with_unreachable_block() {
        // Entry(0) → A(1) → B(2), C(3) is unreachable.
        let mut blocks = vec![
            make_block(0, "entry", Terminator::Branch { target: BlockId(1) }),
            make_block(1, "A", Terminator::Branch { target: BlockId(2) }),
            make_block(2, "B", Terminator::Return { value: None }),
            make_block(3, "C", Terminator::Return { value: None }),
        ];

        link_blocks(&mut blocks, 0, 1);
        link_blocks(&mut blocks, 1, 2);
        // C (block 3) has no predecessors — it is unreachable.

        let mut func = make_function(blocks, BlockId(0));

        let mut pass = DcePass::new();
        let changed = pass.run_on_function(&mut func);

        assert!(changed);
        // Block 3 (C) should be cleared: empty instructions, Unreachable terminator.
        assert!(func.blocks[3].instructions.is_empty());
        assert!(func.blocks[3].phi_nodes.is_empty());
        assert!(func.blocks[3].predecessors.is_empty());
        assert!(func.blocks[3].successors.is_empty());
        assert!(matches!(
            func.blocks[3].terminator,
            Some(Terminator::Unreachable)
        ));

        // Reachable blocks should be intact.
        assert!(matches!(
            func.blocks[0].terminator,
            Some(Terminator::Branch { .. })
        ));
        assert!(matches!(
            func.blocks[1].terminator,
            Some(Terminator::Branch { .. })
        ));
        assert!(matches!(
            func.blocks[2].terminator,
            Some(Terminator::Return { .. })
        ));
    }

    #[test]
    fn test_diamond_cfg_all_reachable() {
        // Entry(0) → A(1), Entry(0) → B(2), A → C(3), B → C(3).
        // All blocks reachable — nothing removed.
        let cond_val = Value(0);
        let mut blocks = vec![
            make_block(
                0,
                "entry",
                Terminator::CondBranch {
                    condition: cond_val,
                    true_block: BlockId(1),
                    false_block: BlockId(2),
                },
            ),
            make_block(1, "A", Terminator::Branch { target: BlockId(3) }),
            make_block(2, "B", Terminator::Branch { target: BlockId(3) }),
            make_block(3, "C", Terminator::Return { value: None }),
        ];

        // Add a Const instruction that produces cond_val so it's defined.
        blocks[0].instructions.push(Instruction::Const {
            result: cond_val,
            value: Constant::Bool(true),
        });

        link_blocks(&mut blocks, 0, 1);
        link_blocks(&mut blocks, 0, 2);
        link_blocks(&mut blocks, 1, 3);
        link_blocks(&mut blocks, 2, 3);

        let mut func = make_function(blocks, BlockId(0));

        let mut pass = DcePass::new();
        let changed = pass.run_on_function(&mut func);

        // The const instruction for cond_val is used by the CondBranch terminator,
        // so it should be preserved. No blocks are unreachable.
        assert!(!changed);
    }

    #[test]
    fn test_multiple_unreachable_blocks() {
        // Entry(0) → A(1), B(2) and C(3) unreachable.
        let mut blocks = vec![
            make_block(0, "entry", Terminator::Branch { target: BlockId(1) }),
            make_block(1, "A", Terminator::Return { value: None }),
            make_block(2, "B", Terminator::Return { value: None }),
            make_block(3, "C", Terminator::Return { value: None }),
        ];

        link_blocks(&mut blocks, 0, 1);
        // B and C have no predecessors from reachable blocks.

        let mut func = make_function(blocks, BlockId(0));

        let mut pass = DcePass::new();
        let changed = pass.run_on_function(&mut func);

        assert!(changed);
        // Blocks 2 and 3 should be cleared.
        for idx in [2, 3] {
            assert!(func.blocks[idx].instructions.is_empty());
            assert!(matches!(
                func.blocks[idx].terminator,
                Some(Terminator::Unreachable)
            ));
        }
        // Blocks 0 and 1 remain intact.
        assert!(matches!(
            func.blocks[0].terminator,
            Some(Terminator::Branch { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // Dead Instruction Removal Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_dead_add_removed() {
        // a = add 1, 2 where `a` is never used → `a` should be removed.
        let v1 = Value(0);
        let v2 = Value(1);
        let v_result = Value(2);

        let mut blocks = vec![{
            let mut block = make_block(0, "entry", Terminator::Return { value: None });
            block.instructions.push(Instruction::Const {
                result: v1,
                value: Constant::Integer {
                    value: 1,
                    ty: IrType::I32,
                },
            });
            block.instructions.push(Instruction::Const {
                result: v2,
                value: Constant::Integer {
                    value: 2,
                    ty: IrType::I32,
                },
            });
            block.instructions.push(Instruction::Add {
                result: v_result,
                lhs: v1,
                rhs: v2,
                ty: IrType::I32,
            });
            block
        }];

        let mut func = make_function(blocks, BlockId(0));

        let mut pass = DcePass::new();
        let changed = pass.run_on_function(&mut func);

        // All three instructions are dead (add result unused, const values only
        // used by the dead add). They should all be removed via cascading elimination.
        assert!(changed);
        assert!(func.blocks[0].instructions.is_empty());
    }

    #[test]
    fn test_live_add_preserved() {
        // a = add 1, 2; ret a → `a` is used by return, must be preserved.
        let v1 = Value(0);
        let v2 = Value(1);
        let v_result = Value(2);

        let mut blocks = vec![{
            let mut block = make_block(
                0,
                "entry",
                Terminator::Return {
                    value: Some(v_result),
                },
            );
            block.instructions.push(Instruction::Const {
                result: v1,
                value: Constant::Integer {
                    value: 1,
                    ty: IrType::I32,
                },
            });
            block.instructions.push(Instruction::Const {
                result: v2,
                value: Constant::Integer {
                    value: 2,
                    ty: IrType::I32,
                },
            });
            block.instructions.push(Instruction::Add {
                result: v_result,
                lhs: v1,
                rhs: v2,
                ty: IrType::I32,
            });
            block
        }];

        let mut func = make_function(blocks, BlockId(0));

        let mut pass = DcePass::new();
        let changed = pass.run_on_function(&mut func);

        // All instructions are live (add used by return, consts used by add).
        assert!(!changed);
        assert_eq!(func.blocks[0].instructions.len(), 3);
    }

    #[test]
    fn test_store_preserved_as_side_effect() {
        // store v, ptr → must be preserved (side effect).
        let v_val = Value(0);
        let v_ptr = Value(1);

        let mut blocks = vec![{
            let mut block = make_block(0, "entry", Terminator::Return { value: None });
            block.instructions.push(Instruction::Const {
                result: v_val,
                value: Constant::Integer {
                    value: 42,
                    ty: IrType::I32,
                },
            });
            block.instructions.push(Instruction::Const {
                result: v_ptr,
                value: Constant::Integer {
                    value: 0,
                    ty: IrType::I64,
                },
            });
            block.instructions.push(Instruction::Store {
                value: v_val,
                ptr: v_ptr,
            });
            block
        }];

        let mut func = make_function(blocks, BlockId(0));

        let mut pass = DcePass::new();
        let changed = pass.run_on_function(&mut func);

        // Store is side-effecting, and its operands (the two consts) are used by
        // the store, so everything is live.
        assert!(!changed);
        assert_eq!(func.blocks[0].instructions.len(), 3);
    }

    #[test]
    fn test_call_preserved_even_with_unused_result() {
        // result = call @printf() where result is unused → call preserved (side effect).
        let v_result = Value(0);

        let mut blocks = vec![{
            let mut block = make_block(0, "entry", Terminator::Return { value: None });
            block.instructions.push(Instruction::Call {
                result: Some(v_result),
                callee: Callee::Direct("printf".to_string()),
                args: vec![],
                return_ty: IrType::I32,
            });
            block
        }];

        let mut func = make_function(blocks, BlockId(0));

        let mut pass = DcePass::new();
        let changed = pass.run_on_function(&mut func);

        // Call has side effects — preserved even with unused result.
        assert!(!changed);
        assert_eq!(func.blocks[0].instructions.len(), 1);
    }

    #[test]
    fn test_chained_dead_code_eliminated() {
        // a = const 5; b = add a, 3; c = mul b, 2 — c is unused.
        // All three should be eliminated via cascading.
        let v_a = Value(0);
        let v_3 = Value(1);
        let v_b = Value(2);
        let v_2 = Value(3);
        let v_c = Value(4);

        let mut blocks = vec![{
            let mut block = make_block(0, "entry", Terminator::Return { value: None });
            block.instructions.push(Instruction::Const {
                result: v_a,
                value: Constant::Integer {
                    value: 5,
                    ty: IrType::I32,
                },
            });
            block.instructions.push(Instruction::Const {
                result: v_3,
                value: Constant::Integer {
                    value: 3,
                    ty: IrType::I32,
                },
            });
            block.instructions.push(Instruction::Add {
                result: v_b,
                lhs: v_a,
                rhs: v_3,
                ty: IrType::I32,
            });
            block.instructions.push(Instruction::Const {
                result: v_2,
                value: Constant::Integer {
                    value: 2,
                    ty: IrType::I32,
                },
            });
            block.instructions.push(Instruction::Mul {
                result: v_c,
                lhs: v_b,
                rhs: v_2,
                ty: IrType::I32,
            });
            block
        }];

        let mut func = make_function(blocks, BlockId(0));

        let mut pass = DcePass::new();
        let changed = pass.run_on_function(&mut func);

        // c unused → c removed → b loses its use → b removed → a and constants
        // lose their uses → all removed.
        assert!(changed);
        assert!(func.blocks[0].instructions.is_empty());
    }

    // -----------------------------------------------------------------------
    // Dead Phi Node Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_dead_phi_node_removed() {
        // Phi node whose result is unused → removed.
        let v_left = Value(0);
        let v_right = Value(1);
        let v_phi = Value(2);

        let mut blocks = vec![
            {
                let mut block = make_block(
                    0,
                    "entry",
                    Terminator::Branch { target: BlockId(2) },
                );
                block.instructions.push(Instruction::Const {
                    result: v_left,
                    value: Constant::Integer {
                        value: 10,
                        ty: IrType::I32,
                    },
                });
                block
            },
            {
                let mut block = make_block(
                    1,
                    "other",
                    Terminator::Branch { target: BlockId(2) },
                );
                block.instructions.push(Instruction::Const {
                    result: v_right,
                    value: Constant::Integer {
                        value: 20,
                        ty: IrType::I32,
                    },
                });
                block
            },
            {
                let mut block = make_block(
                    2,
                    "merge",
                    Terminator::Return { value: None },
                );
                block.phi_nodes.push(PhiNode {
                    result: v_phi,
                    ty: IrType::I32,
                    incoming: vec![
                        (v_left, BlockId(0)),
                        (v_right, BlockId(1)),
                    ],
                });
                block
            },
        ];

        link_blocks(&mut blocks, 0, 2);
        link_blocks(&mut blocks, 1, 2);

        let mut func = make_function(blocks, BlockId(0));

        let mut pass = DcePass::new();
        let changed = pass.run_on_function(&mut func);

        // Block 1 is unreachable (no path from entry through successors).
        // The phi node result is unused → phi removed.
        // The const in block 0 (v_left) loses its use → removed.
        assert!(changed);
        assert!(func.blocks[2].phi_nodes.is_empty());
    }

    #[test]
    fn test_live_phi_node_preserved() {
        // Phi node whose result is used by return → preserved.
        let v_left = Value(0);
        let v_right = Value(1);
        let v_phi = Value(2);
        let v_cond = Value(3);

        let mut blocks = vec![
            {
                let mut block = make_block(
                    0,
                    "entry",
                    Terminator::CondBranch {
                        condition: v_cond,
                        true_block: BlockId(1),
                        false_block: BlockId(2),
                    },
                );
                block.instructions.push(Instruction::Const {
                    result: v_cond,
                    value: Constant::Bool(true),
                });
                block
            },
            {
                let mut block = make_block(
                    1,
                    "left",
                    Terminator::Branch { target: BlockId(3) },
                );
                block.instructions.push(Instruction::Const {
                    result: v_left,
                    value: Constant::Integer {
                        value: 10,
                        ty: IrType::I32,
                    },
                });
                block
            },
            {
                let mut block = make_block(
                    2,
                    "right",
                    Terminator::Branch { target: BlockId(3) },
                );
                block.instructions.push(Instruction::Const {
                    result: v_right,
                    value: Constant::Integer {
                        value: 20,
                        ty: IrType::I32,
                    },
                });
                block
            },
            {
                let mut block = make_block(
                    3,
                    "merge",
                    Terminator::Return {
                        value: Some(v_phi),
                    },
                );
                block.phi_nodes.push(PhiNode {
                    result: v_phi,
                    ty: IrType::I32,
                    incoming: vec![
                        (v_left, BlockId(1)),
                        (v_right, BlockId(2)),
                    ],
                });
                block
            },
        ];

        link_blocks(&mut blocks, 0, 1);
        link_blocks(&mut blocks, 0, 2);
        link_blocks(&mut blocks, 1, 3);
        link_blocks(&mut blocks, 2, 3);

        let mut func = make_function(blocks, BlockId(0));

        let mut pass = DcePass::new();
        let changed = pass.run_on_function(&mut func);

        // Phi is used by return → live. All instructions are live.
        assert!(!changed);
        assert_eq!(func.blocks[3].phi_nodes.len(), 1);
    }

    // -----------------------------------------------------------------------
    // No-Change Test
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_live_returns_false() {
        // Function where every instruction is live → pass returns false.
        let v_a = Value(0);

        let mut blocks = vec![{
            let mut block = make_block(
                0,
                "entry",
                Terminator::Return {
                    value: Some(v_a),
                },
            );
            block.instructions.push(Instruction::Const {
                result: v_a,
                value: Constant::Integer {
                    value: 42,
                    ty: IrType::I32,
                },
            });
            block
        }];

        let mut func = make_function(blocks, BlockId(0));

        let mut pass = DcePass::new();
        let changed = pass.run_on_function(&mut func);

        assert!(!changed);
        assert_eq!(func.blocks[0].instructions.len(), 1);
    }

    // -----------------------------------------------------------------------
    // CFG Cleanup Test
    // -----------------------------------------------------------------------

    #[test]
    fn test_phi_cleanup_after_block_removal() {
        // After removing an unreachable block, phi nodes in surviving blocks
        // should not reference the removed block.
        //
        // CFG: Entry(0) → Merge(2), Dead(1) → Merge(2) [Dead is unreachable]
        // Merge has a phi with incoming from both Entry and Dead.
        let v_from_entry = Value(0);
        let v_from_dead = Value(1);
        let v_phi = Value(2);

        let mut blocks = vec![
            {
                let mut block = make_block(
                    0,
                    "entry",
                    Terminator::Branch { target: BlockId(2) },
                );
                block.instructions.push(Instruction::Const {
                    result: v_from_entry,
                    value: Constant::Integer {
                        value: 1,
                        ty: IrType::I32,
                    },
                });
                block
            },
            {
                // Dead block — unreachable from entry.
                let mut block = make_block(
                    1,
                    "dead",
                    Terminator::Branch { target: BlockId(2) },
                );
                block.instructions.push(Instruction::Const {
                    result: v_from_dead,
                    value: Constant::Integer {
                        value: 2,
                        ty: IrType::I32,
                    },
                });
                block
            },
            {
                let mut block = make_block(
                    2,
                    "merge",
                    Terminator::Return {
                        value: Some(v_phi),
                    },
                );
                block.phi_nodes.push(PhiNode {
                    result: v_phi,
                    ty: IrType::I32,
                    incoming: vec![
                        (v_from_entry, BlockId(0)),
                        (v_from_dead, BlockId(1)),
                    ],
                });
                block
            },
        ];

        link_blocks(&mut blocks, 0, 2);
        // Dead(1) → Merge(2) edge exists in the data but Dead is unreachable.
        link_blocks(&mut blocks, 1, 2);

        let mut func = make_function(blocks, BlockId(0));

        let mut pass = DcePass::new();
        let changed = pass.run_on_function(&mut func);

        assert!(changed);
        // The phi node in Merge should only have the incoming edge from Entry.
        assert_eq!(func.blocks[2].phi_nodes.len(), 1);
        assert_eq!(func.blocks[2].phi_nodes[0].incoming.len(), 1);
        assert_eq!(func.blocks[2].phi_nodes[0].incoming[0].1, BlockId(0));

        // Dead block should be cleared.
        assert!(func.blocks[1].instructions.is_empty());
        assert!(matches!(
            func.blocks[1].terminator,
            Some(Terminator::Unreachable)
        ));

        // Dead block should no longer appear in Merge's predecessors.
        assert!(!func.blocks[2].predecessors.contains(&BlockId(1)));
    }

    // -----------------------------------------------------------------------
    // Side Effect Preservation Test
    // -----------------------------------------------------------------------

    #[test]
    fn test_mixed_dead_pure_and_live_side_effects() {
        // Mix of dead pure instructions and live side-effecting instructions.
        // Only dead pure instructions should be removed.
        let v_dead = Value(0);
        let v_store_val = Value(1);
        let v_store_ptr = Value(2);

        let mut blocks = vec![{
            let mut block = make_block(0, "entry", Terminator::Return { value: None });
            // Dead pure instruction: unused add.
            block.instructions.push(Instruction::Const {
                result: v_dead,
                value: Constant::Integer {
                    value: 999,
                    ty: IrType::I32,
                },
            });
            // Live side-effecting instruction: store.
            block.instructions.push(Instruction::Const {
                result: v_store_val,
                value: Constant::Integer {
                    value: 42,
                    ty: IrType::I32,
                },
            });
            block.instructions.push(Instruction::Const {
                result: v_store_ptr,
                value: Constant::Integer {
                    value: 0,
                    ty: IrType::I64,
                },
            });
            block.instructions.push(Instruction::Store {
                value: v_store_val,
                ptr: v_store_ptr,
            });
            block
        }];

        let mut func = make_function(blocks, BlockId(0));

        let mut pass = DcePass::new();
        let changed = pass.run_on_function(&mut func);

        assert!(changed);
        // The dead const (v_dead) should be removed. Store and its operands stay.
        assert_eq!(func.blocks[0].instructions.len(), 3);
        // Verify the store is still present.
        let has_store = func.blocks[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Store { .. }));
        assert!(has_store);
    }

    // -----------------------------------------------------------------------
    // Empty Function Test
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_function() {
        let mut func = Function {
            name: "empty".to_string(),
            return_type: IrType::Void,
            params: vec![],
            blocks: vec![],
            entry_block: BlockId(0),
            is_definition: true,
        };

        let mut pass = DcePass::new();
        let changed = pass.run_on_function(&mut func);

        assert!(!changed);
    }

    // -----------------------------------------------------------------------
    // Pass Name Test
    // -----------------------------------------------------------------------

    #[test]
    fn test_pass_name() {
        let pass = DcePass::new();
        assert_eq!(pass.name(), "dce");
    }

    // -----------------------------------------------------------------------
    // Add Used by Store Is Live Test
    // -----------------------------------------------------------------------

    #[test]
    fn test_value_used_by_store_is_live() {
        // a = add 1, 2; store a, ptr → a is used by store → a preserved.
        let v1 = Value(0);
        let v2 = Value(1);
        let v_a = Value(2);
        let v_ptr = Value(3);

        let mut blocks = vec![{
            let mut block = make_block(0, "entry", Terminator::Return { value: None });
            block.instructions.push(Instruction::Const {
                result: v1,
                value: Constant::Integer {
                    value: 1,
                    ty: IrType::I32,
                },
            });
            block.instructions.push(Instruction::Const {
                result: v2,
                value: Constant::Integer {
                    value: 2,
                    ty: IrType::I32,
                },
            });
            block.instructions.push(Instruction::Add {
                result: v_a,
                lhs: v1,
                rhs: v2,
                ty: IrType::I32,
            });
            block.instructions.push(Instruction::Const {
                result: v_ptr,
                value: Constant::Integer {
                    value: 0,
                    ty: IrType::I64,
                },
            });
            block.instructions.push(Instruction::Store {
                value: v_a,
                ptr: v_ptr,
            });
            block
        }];

        let mut func = make_function(blocks, BlockId(0));

        let mut pass = DcePass::new();
        let changed = pass.run_on_function(&mut func);

        // All instructions are live: consts used by add, add used by store.
        assert!(!changed);
        assert_eq!(func.blocks[0].instructions.len(), 5);
    }

    // -----------------------------------------------------------------------
    // Value Used by CondBranch Is Live Test
    // -----------------------------------------------------------------------

    #[test]
    fn test_value_used_by_condbranch_is_live() {
        // cond = icmp ...; condbranch cond, ... → cond is live.
        let v_lhs = Value(0);
        let v_rhs = Value(1);
        let v_cond = Value(2);

        let mut blocks = vec![
            {
                let mut block = make_block(
                    0,
                    "entry",
                    Terminator::CondBranch {
                        condition: v_cond,
                        true_block: BlockId(1),
                        false_block: BlockId(2),
                    },
                );
                block.instructions.push(Instruction::Const {
                    result: v_lhs,
                    value: Constant::Integer {
                        value: 1,
                        ty: IrType::I32,
                    },
                });
                block.instructions.push(Instruction::Const {
                    result: v_rhs,
                    value: Constant::Integer {
                        value: 2,
                        ty: IrType::I32,
                    },
                });
                block.instructions.push(Instruction::ICmp {
                    result: v_cond,
                    op: crate::ir::instructions::CompareOp::Equal,
                    lhs: v_lhs,
                    rhs: v_rhs,
                    ty: IrType::I32,
                });
                block
            },
            make_block(1, "then", Terminator::Return { value: None }),
            make_block(2, "else", Terminator::Return { value: None }),
        ];

        link_blocks(&mut blocks, 0, 1);
        link_blocks(&mut blocks, 0, 2);

        let mut func = make_function(blocks, BlockId(0));

        let mut pass = DcePass::new();
        let changed = pass.run_on_function(&mut func);

        // All instructions are live: consts used by icmp, icmp used by condbranch.
        assert!(!changed);
        assert_eq!(func.blocks[0].instructions.len(), 3);
    }

    // -----------------------------------------------------------------------
    // Value Used by Switch Is Live Test
    // -----------------------------------------------------------------------

    #[test]
    fn test_value_used_by_switch_is_live() {
        let v_switch_val = Value(0);

        let mut blocks = vec![
            {
                let mut block = make_block(
                    0,
                    "entry",
                    Terminator::Switch {
                        value: v_switch_val,
                        default: BlockId(1),
                        cases: vec![(1, BlockId(2))],
                    },
                );
                block.instructions.push(Instruction::Const {
                    result: v_switch_val,
                    value: Constant::Integer {
                        value: 0,
                        ty: IrType::I32,
                    },
                });
                block
            },
            make_block(1, "default", Terminator::Return { value: None }),
            make_block(2, "case1", Terminator::Return { value: None }),
        ];

        link_blocks(&mut blocks, 0, 1);
        link_blocks(&mut blocks, 0, 2);

        let mut func = make_function(blocks, BlockId(0));

        let mut pass = DcePass::new();
        let changed = pass.run_on_function(&mut func);

        // Const is used by switch terminator → live.
        assert!(!changed);
        assert_eq!(func.blocks[0].instructions.len(), 1);
    }
}
