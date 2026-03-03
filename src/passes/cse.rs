//! Common subexpression elimination (CSE) optimization pass for the `bcc` compiler.
//!
//! This module implements [`CsePass`], which uses value numbering to detect
//! redundant computations — expressions that compute the same value from the
//! same operands — and eliminates them by reusing previously computed values.
//!
//! # Algorithms
//!
//! Two complementary algorithms are implemented:
//!
//! - **Local CSE** — Within a single basic block, a `HashMap<ExpressionKey, Value>`
//!   maps each expression to the first SSA value that computes it. Duplicate
//!   expressions have their uses replaced and are removed.
//!
//! - **Global CSE** — A dominator-tree walk with a scoped expression table
//!   enables reuse of expressions computed in dominating blocks. This is the
//!   standard *Dominator-based Value Numbering* technique (Alpern, Wegman,
//!   Zadeck 1988).
//!
//! # Correctness Invariants
//!
//! - Instructions with side effects (`Store`, `Call`) are **never** eliminated.
//! - `Load` instructions are conservatively excluded (no alias analysis).
//! - `Alloca`, `Phi`, `Const`, `Select`, `Copy`, and `Nop` are not CSE-eligible.
//! - Two expressions are considered identical only when they share the **same
//!   opcode**, **same operand `Value`s**, and **same `IrType`**.
//! - Commutative operations (`Add`, `Mul`, `And`, `Or`, `Xor`) canonicalize
//!   operand order (smaller `Value` first) so that `a + b` and `b + a` hash
//!   to the same key.
//! - Global CSE only reuses an expression if the original definition dominates
//!   the current use point, preserving SSA correctness.
//!
//! # Performance
//!
//! Hash-based value numbering is O(N) per basic block. The dominator-tree walk
//! for global CSE is O(N) for the entire function. Total cost is linear in the
//! number of IR instructions, suitable for compiling the SQLite amalgamation
//! (~230K LOC) within the <60 s budget.
//!
//! # Zero External Dependencies
//!
//! Only imports from `std` and internal crate modules. No external crates.
//!
//! # Safety
//!
//! This module contains zero `unsafe` blocks. All operations use safe Rust.

use std::collections::{HashMap, HashSet};

use crate::ir::builder::Function;
use crate::ir::cfg::{
    BasicBlock, ControlFlowGraph, DominanceTree, PhiNode, Terminator,
    compute_dominance_tree,
};
use crate::ir::instructions::{
    BlockId, CastOp, CompareOp, FloatCompareOp, Instruction, Value,
};
use crate::ir::types::IrType;

use super::FunctionPass;

// ---------------------------------------------------------------------------
// BinaryOpcode — discriminant for binary expression keys
// ---------------------------------------------------------------------------

/// Encodes the specific binary operation for expression key hashing.
///
/// Signed and unsigned division/modulo, and arithmetic vs. logical right shift,
/// are distinct opcodes because they have different semantics: `sdiv(a, b)` ≠
/// `udiv(a, b)` even when operands are bitwise identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BinaryOpcode {
    Add,
    Sub,
    Mul,
    /// Unsigned integer division.
    UDiv,
    /// Signed integer division.
    SDiv,
    /// Unsigned integer modulo.
    UMod,
    /// Signed integer modulo.
    SMod,
    And,
    Or,
    Xor,
    Shl,
    /// Logical (unsigned) right shift — zeros shifted in.
    LShr,
    /// Arithmetic (signed) right shift — sign bit replicated.
    AShr,
}

impl BinaryOpcode {
    /// Returns `true` for commutative operations where operand order is
    /// interchangeable: `a ⊕ b == b ⊕ a`.
    #[inline]
    fn is_commutative(self) -> bool {
        matches!(
            self,
            BinaryOpcode::Add
                | BinaryOpcode::Mul
                | BinaryOpcode::And
                | BinaryOpcode::Or
                | BinaryOpcode::Xor
        )
    }
}

// ---------------------------------------------------------------------------
// ExpressionKey — hashable identity for CSE-eligible computations
// ---------------------------------------------------------------------------

/// A hashable key that uniquely identifies a pure computation in the IR.
///
/// Two instructions that map to the same `ExpressionKey` compute exactly the
/// same value (given SSA uniqueness of operand `Value`s), making the second
/// one redundant and replaceable.
///
/// Only side-effect-free, deterministic instructions produce keys. Memory
/// operations, function calls, phi nodes, constants, and selects are excluded.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ExpressionKey {
    /// Binary arithmetic or bitwise operation.
    BinaryOp {
        opcode: BinaryOpcode,
        lhs: Value,
        rhs: Value,
        ty: IrType,
    },
    /// Integer comparison (`ICmp`).
    /// `ty` is the operand type being compared (result is always `i1`).
    Comparison {
        op: CompareOp,
        lhs: Value,
        rhs: Value,
        ty: IrType,
    },
    /// Floating-point comparison (`FCmp`).
    /// `ty` is the operand type being compared (result is always `i1`).
    FloatComparison {
        op: FloatCompareOp,
        lhs: Value,
        rhs: Value,
        ty: IrType,
    },
    /// Value-changing type cast (`Cast` with explicit `CastOp`).
    Cast {
        op: CastOp,
        value: Value,
        from_ty: IrType,
        to_ty: IrType,
    },
    /// Bit-level reinterpretation cast (`BitCast`).
    BitCast {
        value: Value,
        from_ty: IrType,
        to_ty: IrType,
    },
    /// Element pointer computation (`GetElementPtr`).
    GetElementPtr {
        base_ty: IrType,
        ptr: Value,
        indices: Vec<Value>,
    },
}

// ---------------------------------------------------------------------------
// Expression key extraction
// ---------------------------------------------------------------------------

/// Canonicalizes operand order for commutative binary operations.
///
/// For commutative ops the smaller `Value` is placed first so that
/// `Add(%1, %2)` and `Add(%2, %1)` produce identical keys.
/// Non-commutative ops pass through unchanged.
#[inline]
fn canonicalize(lhs: Value, rhs: Value, opcode: BinaryOpcode) -> (Value, Value) {
    if opcode.is_commutative() && lhs > rhs {
        (rhs, lhs)
    } else {
        (lhs, rhs)
    }
}

/// Attempts to extract an `ExpressionKey` from an IR instruction.
///
/// Returns `Some(key)` for pure, side-effect-free instructions eligible for
/// CSE, and `None` for all others (memory ops, calls, phi, const, select,
/// copy, nop).
fn expression_key(inst: &Instruction) -> Option<ExpressionKey> {
    match inst {
        // --- Arithmetic (commutative: Add, Mul) ---
        Instruction::Add { lhs, rhs, ty, .. } => {
            let (l, r) = canonicalize(*lhs, *rhs, BinaryOpcode::Add);
            Some(ExpressionKey::BinaryOp {
                opcode: BinaryOpcode::Add, lhs: l, rhs: r, ty: ty.clone(),
            })
        }
        Instruction::Mul { lhs, rhs, ty, .. } => {
            let (l, r) = canonicalize(*lhs, *rhs, BinaryOpcode::Mul);
            Some(ExpressionKey::BinaryOp {
                opcode: BinaryOpcode::Mul, lhs: l, rhs: r, ty: ty.clone(),
            })
        }

        // --- Arithmetic (non-commutative: Sub, Div, Mod) ---
        Instruction::Sub { lhs, rhs, ty, .. } => {
            Some(ExpressionKey::BinaryOp {
                opcode: BinaryOpcode::Sub, lhs: *lhs, rhs: *rhs, ty: ty.clone(),
            })
        }
        Instruction::Div { lhs, rhs, ty, is_signed, .. } => {
            let op = if *is_signed { BinaryOpcode::SDiv } else { BinaryOpcode::UDiv };
            Some(ExpressionKey::BinaryOp {
                opcode: op, lhs: *lhs, rhs: *rhs, ty: ty.clone(),
            })
        }
        Instruction::Mod { lhs, rhs, ty, is_signed, .. } => {
            let op = if *is_signed { BinaryOpcode::SMod } else { BinaryOpcode::UMod };
            Some(ExpressionKey::BinaryOp {
                opcode: op, lhs: *lhs, rhs: *rhs, ty: ty.clone(),
            })
        }

        // --- Bitwise (commutative: And, Or, Xor) ---
        Instruction::And { lhs, rhs, ty, .. } => {
            let (l, r) = canonicalize(*lhs, *rhs, BinaryOpcode::And);
            Some(ExpressionKey::BinaryOp {
                opcode: BinaryOpcode::And, lhs: l, rhs: r, ty: ty.clone(),
            })
        }
        Instruction::Or { lhs, rhs, ty, .. } => {
            let (l, r) = canonicalize(*lhs, *rhs, BinaryOpcode::Or);
            Some(ExpressionKey::BinaryOp {
                opcode: BinaryOpcode::Or, lhs: l, rhs: r, ty: ty.clone(),
            })
        }
        Instruction::Xor { lhs, rhs, ty, .. } => {
            let (l, r) = canonicalize(*lhs, *rhs, BinaryOpcode::Xor);
            Some(ExpressionKey::BinaryOp {
                opcode: BinaryOpcode::Xor, lhs: l, rhs: r, ty: ty.clone(),
            })
        }

        // --- Shifts (non-commutative) ---
        Instruction::Shl { lhs, rhs, ty, .. } => {
            Some(ExpressionKey::BinaryOp {
                opcode: BinaryOpcode::Shl, lhs: *lhs, rhs: *rhs, ty: ty.clone(),
            })
        }
        Instruction::Shr { lhs, rhs, ty, is_arithmetic, .. } => {
            let op = if *is_arithmetic { BinaryOpcode::AShr } else { BinaryOpcode::LShr };
            Some(ExpressionKey::BinaryOp {
                opcode: op, lhs: *lhs, rhs: *rhs, ty: ty.clone(),
            })
        }

        // --- Comparisons ---
        Instruction::ICmp { op, lhs, rhs, ty, .. } => {
            Some(ExpressionKey::Comparison { op: *op, lhs: *lhs, rhs: *rhs, ty: ty.clone() })
        }
        Instruction::FCmp { op, lhs, rhs, ty, .. } => {
            Some(ExpressionKey::FloatComparison { op: *op, lhs: *lhs, rhs: *rhs, ty: ty.clone() })
        }

        // --- Type conversions ---
        Instruction::Cast { op, value, from_ty, to_ty, .. } => {
            Some(ExpressionKey::Cast {
                op: *op, value: *value,
                from_ty: from_ty.clone(), to_ty: to_ty.clone(),
            })
        }
        Instruction::BitCast { value, from_ty, to_ty, .. } => {
            Some(ExpressionKey::BitCast {
                value: *value,
                from_ty: from_ty.clone(), to_ty: to_ty.clone(),
            })
        }

        // --- Aggregate addressing ---
        Instruction::GetElementPtr { base_ty, ptr, indices, .. } => {
            Some(ExpressionKey::GetElementPtr {
                base_ty: base_ty.clone(), ptr: *ptr, indices: indices.clone(),
            })
        }

        // --- NOT eligible for CSE ---
        // Side effects: Store, Call
        // Memory/non-deterministic: Load, Alloca
        // SSA-construction artifacts: Phi
        // Trivial / non-hashable: Const, Select, Copy, Nop
        Instruction::Store { .. }
        | Instruction::Call { .. }
        | Instruction::Load { .. }
        | Instruction::Alloca { .. }
        | Instruction::Phi { .. }
        | Instruction::Const { .. }
        | Instruction::Select { .. }
        | Instruction::Copy { .. }
        | Instruction::Nop => None,
    }
}

// ---------------------------------------------------------------------------
// ScopedExprTable — dominator-tree-scoped expression table for global CSE
// ---------------------------------------------------------------------------

/// A stack of `HashMap<ExpressionKey, Value>` scopes mirroring the dominator
/// tree hierarchy.
///
/// When entering a block during the dominator-tree walk, a new scope is pushed.
/// Expression lookups search from the innermost (current block) scope outward
/// to the root (entry block) scope, ensuring only dominating expressions are
/// found. When leaving a subtree, the scope is popped, removing expressions
/// that no longer dominate subsequent blocks.
struct ScopedExprTable {
    scopes: Vec<HashMap<ExpressionKey, Value>>,
}

impl ScopedExprTable {
    /// Creates a new empty scoped expression table.
    fn new() -> Self {
        ScopedExprTable { scopes: Vec::new() }
    }

    /// Pushes a fresh empty scope onto the scope stack.
    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    /// Pops the topmost scope from the scope stack.
    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    /// Searches for `key` from the innermost scope outward.
    ///
    /// Returns the `Value` of the first matching expression (i.e., the one in
    /// the nearest dominating block), or `None` if the expression has not been
    /// seen in any dominating scope.
    fn lookup(&self, key: &ExpressionKey) -> Option<Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(&val) = scope.get(key) {
                return Some(val);
            }
        }
        None
    }

    /// Inserts an expression key -> result value mapping into the current
    /// (innermost) scope.
    fn insert(&mut self, key: ExpressionKey, value: Value) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(key, value);
        }
    }
}

// ---------------------------------------------------------------------------
// CsePass — the public optimization pass struct
// ---------------------------------------------------------------------------

/// Common subexpression elimination optimization pass.
///
/// Implements the [`FunctionPass`] trait to integrate with the optimization
/// pipeline at `-O2`. Performs both local (intra-block) and global
/// (dominator-tree-based inter-block) CSE.
///
/// # Example
///
/// ```ignore
/// use crate::passes::cse::CsePass;
/// use crate::passes::FunctionPass;
///
/// let mut pass = CsePass::new();
/// let changed = pass.run_on_function(&mut function);
/// ```
pub struct CsePass;

impl CsePass {
    /// Creates a new CSE pass instance.
    pub fn new() -> Self {
        CsePass
    }
}

impl FunctionPass for CsePass {
    /// Returns the human-readable name for this optimization pass.
    fn name(&self) -> &str {
        "cse"
    }

    /// Runs common subexpression elimination on the given function.
    ///
    /// For single-block functions, only local CSE runs (cheap and sufficient).
    /// For multi-block functions, local CSE runs first, then global CSE via a
    /// dominator-tree walk catches cross-block redundancies.
    ///
    /// # Returns
    ///
    /// `true` if any redundant instructions were eliminated, `false` otherwise.
    fn run_on_function(&mut self, function: &mut Function) -> bool {
        // Skip non-definitions and empty functions.
        if !function.is_definition || function.blocks.is_empty() {
            return false;
        }

        let mut changed = false;

        // Phase 1: Local CSE within each basic block — catches the most common
        // case of duplicate expressions within straight-line code.
        changed |= run_local_cse(function);

        // Phase 2: Global CSE across basic blocks — catches cross-block
        // redundancies where an expression computed in a dominating block is
        // recomputed in a dominated block.
        if function.blocks.len() > 1 {
            changed |= run_global_cse(function);
        }

        changed
    }
}

// ---------------------------------------------------------------------------
// Value replacement helpers
// ---------------------------------------------------------------------------

/// Replaces all uses of `old` with `new` across the entire function.
///
/// Walks every phi node, instruction, and terminator in every block, updating
/// any operand that references `old` to instead reference `new`.
fn replace_uses_in_function(function: &mut Function, old: Value, new: Value) {
    for block in function.blocks.iter_mut() {
        // Phi node incoming values.
        for phi in block.phi_nodes.iter_mut() {
            for (val, _block_id) in phi.incoming.iter_mut() {
                if *val == old {
                    *val = new;
                }
            }
        }
        // Regular instruction operands.
        for inst in block.instructions.iter_mut() {
            inst.replace_use(old, new);
        }
        // Terminator operands (condition, return value, switch value).
        if let Some(ref mut term) = block.terminator {
            replace_use_in_terminator(term, old, new);
        }
    }
}

/// Replaces occurrences of `old` with `new` in a terminator's value operands.
///
/// Block-id operands (branch targets) are not modified; only SSA `Value`
/// operands used for conditions, return values, and switch discriminants are
/// updated.
fn replace_use_in_terminator(term: &mut Terminator, old: Value, new: Value) {
    match term {
        Terminator::CondBranch { condition, .. } => {
            if *condition == old {
                *condition = new;
            }
        }
        Terminator::Return { value } => {
            if let Some(ref mut v) = value {
                if *v == old {
                    *v = new;
                }
            }
        }
        Terminator::Switch { value, .. } => {
            if *value == old {
                *value = new;
            }
        }
        Terminator::Branch { .. } | Terminator::Unreachable => {
            // No SSA value operands in these terminators.
        }
    }
}

// ---------------------------------------------------------------------------
// Local CSE — intra-block value numbering
// ---------------------------------------------------------------------------

/// Performs local CSE on every basic block in the function.
///
/// Within each block, a `HashMap<ExpressionKey, Value>` records each new
/// expression encountered. When a duplicate key is found, the redundant
/// instruction's result is replaced throughout the function and the
/// instruction is removed.
///
/// # Returns
///
/// `true` if any eliminations were performed.
fn run_local_cse(function: &mut Function) -> bool {
    // Collect replacement pairs and instruction removal indices first
    // (to avoid aliasing issues with mutable borrows).
    let mut replacements: Vec<(Value, Value)> = Vec::new();
    let mut removals: HashMap<BlockId, Vec<usize>> = HashMap::new();

    for block in &function.blocks {
        let mut expr_table: HashMap<ExpressionKey, Value> = HashMap::new();

        for (i, inst) in block.instructions.iter().enumerate() {
            if let Some(key) = expression_key(inst) {
                if let Some(result) = inst.result() {
                    if let Some(&existing) = expr_table.get(&key) {
                        // Redundant computation — schedule replacement.
                        replacements.push((result, existing));
                        removals.entry(block.id).or_default().push(i);
                    } else {
                        // First occurrence — record it.
                        expr_table.insert(key, result);
                    }
                }
            }
        }
    }

    if replacements.is_empty() {
        return false;
    }

    // Apply value replacements across the entire function.
    for &(old, new) in &replacements {
        replace_uses_in_function(function, old, new);
    }

    // Remove redundant instructions (reverse order per block to preserve indices).
    for block in function.blocks.iter_mut() {
        if let Some(indices) = removals.get(&block.id) {
            let mut sorted: Vec<usize> = indices.clone();
            sorted.sort_unstable_by(|a, b| b.cmp(a));
            for idx in sorted {
                if idx < block.instructions.len() {
                    block.instructions.remove(idx);
                }
            }
        }
    }

    true
}

// ---------------------------------------------------------------------------
// Global CSE — dominator-tree-based inter-block value numbering
// ---------------------------------------------------------------------------

/// Constructs a lightweight `ControlFlowGraph` from the function's blocks,
/// suitable for dominance computation.
///
/// Only block ids, predecessor/successor lists, and terminators are cloned —
/// instructions and phi nodes are omitted since the dominance algorithm only
/// needs the graph structure.
fn build_cfg_for_dominance(function: &Function) -> ControlFlowGraph {
    let mut cfg = ControlFlowGraph::new(function.entry_block);
    for block in &function.blocks {
        let mut bb = BasicBlock::new(block.id, String::new());
        bb.predecessors = block.predecessors.clone();
        bb.successors = block.successors.clone();
        bb.terminator = block.terminator.clone();
        cfg.add_block(bb);
    }
    cfg
}

/// Performs global CSE using a dominator-tree walk with scoped expression tables.
///
/// The dominator tree is traversed in preorder (parent before children).
/// At each block a fresh scope is pushed; when the subtree rooted at that block
/// is fully processed the scope is popped. Expression lookups search all active
/// scopes (innermost first), ensuring only dominating expressions are matched.
///
/// # Returns
///
/// `true` if any eliminations were performed.
fn run_global_cse(function: &mut Function) -> bool {
    // Build a lightweight CFG for dominance computation.
    let cfg = build_cfg_for_dominance(function);
    let dom_tree = compute_dominance_tree(&cfg);

    // Walk the dominator tree, collecting replacements and removal indices.
    let mut table = ScopedExprTable::new();
    let mut replacements: Vec<(Value, Value)> = Vec::new();
    let mut removals: HashMap<BlockId, HashSet<usize>> = HashMap::new();

    global_cse_walk(
        function.entry_block,
        &dom_tree,
        &function.blocks,
        &mut table,
        &mut replacements,
        &mut removals,
    );

    if replacements.is_empty() {
        return false;
    }

    // Apply value replacements across the entire function.
    for &(old, new) in &replacements {
        replace_uses_in_function(function, old, new);
    }

    // Remove redundant instructions (reverse order per block).
    for block in function.blocks.iter_mut() {
        if let Some(indices) = removals.get(&block.id) {
            let mut sorted: Vec<usize> = indices.iter().copied().collect();
            sorted.sort_unstable_by(|a, b| b.cmp(a));
            for idx in sorted {
                if idx < block.instructions.len() {
                    block.instructions.remove(idx);
                }
            }
        }
    }

    true
}

/// Recursively walks the dominator tree in preorder, processing one block at a
/// time with scoped expression tables.
///
/// At each block:
/// 1. A new scope is pushed.
/// 2. Each instruction is checked against the full scoped table (all ancestors).
/// 3. Duplicates are recorded as replacement pairs; new expressions are inserted
///    into the current scope.
/// 4. Children in the dominator tree are recursively processed.
/// 5. The scope is popped, removing block-local expressions.
fn global_cse_walk(
    block_id: BlockId,
    dom_tree: &DominanceTree,
    blocks: &[BasicBlock],
    table: &mut ScopedExprTable,
    replacements: &mut Vec<(Value, Value)>,
    removals: &mut HashMap<BlockId, HashSet<usize>>,
) {
    table.push_scope();

    // Find the block in the blocks slice by matching block id.
    if let Some(block) = blocks.iter().find(|b| b.id == block_id) {
        for (i, inst) in block.instructions.iter().enumerate() {
            if let Some(key) = expression_key(inst) {
                if let Some(result) = inst.result() {
                    if let Some(existing) = table.lookup(&key) {
                        // Expression already computed in a dominating block.
                        replacements.push((result, existing));
                        removals.entry(block_id).or_default().insert(i);
                    } else {
                        // First occurrence in the dominator chain.
                        table.insert(key, result);
                    }
                }
            }
        }
    }

    // Recurse into dominated children.
    for &child in dom_tree.children(block_id) {
        global_cse_walk(child, dom_tree, blocks, table, replacements, removals);
    }

    table.pop_scope();
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::cfg::{BasicBlock, PhiNode, Terminator};
    use crate::ir::instructions::{BlockId, Callee, Instruction, Value};
    use crate::ir::types::IrType;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Creates a minimal single-block function containing the given instructions.
    fn make_single_block_function(instructions: Vec<Instruction>) -> Function {
        let entry = BlockId(0);
        let mut block = BasicBlock::new(entry, "entry".to_string());
        block.instructions = instructions;
        block.terminator = Some(Terminator::Return { value: None });
        Function {
            name: "test_fn".to_string(),
            return_type: IrType::Void,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![block],
            entry_block: entry,
            is_definition: true,
is_static: false,
is_weak: false,
        }
    }

    /// Creates a diamond-shaped CFG:
    ///
    /// ```text
    ///        entry (bb0)
    ///        /         \
    ///   left (bb1)   right (bb2)
    ///        \         /
    ///        merge (bb3)
    /// ```
    fn make_diamond_function(
        entry_insts: Vec<Instruction>,
        left_insts: Vec<Instruction>,
        right_insts: Vec<Instruction>,
        merge_insts: Vec<Instruction>,
        cond: Value,
    ) -> Function {
        let bb0 = BlockId(0);
        let bb1 = BlockId(1);
        let bb2 = BlockId(2);
        let bb3 = BlockId(3);

        let mut entry_block = BasicBlock::new(bb0, "entry".to_string());
        entry_block.instructions = entry_insts;
        entry_block.terminator = Some(Terminator::CondBranch {
            condition: cond,
            true_block: bb1,
            false_block: bb2,
        });
        entry_block.successors = vec![bb1, bb2];

        let mut left_block = BasicBlock::new(bb1, "left".to_string());
        left_block.instructions = left_insts;
        left_block.terminator = Some(Terminator::Branch { target: bb3 });
        left_block.predecessors = vec![bb0];
        left_block.successors = vec![bb3];

        let mut right_block = BasicBlock::new(bb2, "right".to_string());
        right_block.instructions = right_insts;
        right_block.terminator = Some(Terminator::Branch { target: bb3 });
        right_block.predecessors = vec![bb0];
        right_block.successors = vec![bb3];

        let mut merge_block = BasicBlock::new(bb3, "merge".to_string());
        merge_block.instructions = merge_insts;
        merge_block.terminator = Some(Terminator::Return { value: None });
        merge_block.predecessors = vec![bb1, bb2];

        Function {
            name: "diamond_fn".to_string(),
            return_type: IrType::Void,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![entry_block, left_block, right_block, merge_block],
            entry_block: bb0,
            is_definition: true,
is_static: false,
is_weak: false,
        }
    }

    // -----------------------------------------------------------------------
    // Expression key canonicalization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_commutative_add_same_key() {
        let v1 = Value(1);
        let v2 = Value(2);

        let inst_a = Instruction::Add {
            result: Value(10), lhs: v1, rhs: v2, ty: IrType::I32,
        };
        let inst_b = Instruction::Add {
            result: Value(11), lhs: v2, rhs: v1, ty: IrType::I32,
        };

        let key_a = expression_key(&inst_a).unwrap();
        let key_b = expression_key(&inst_b).unwrap();
        assert_eq!(key_a, key_b, "Add(v1,v2) and Add(v2,v1) must hash equally");
    }

    #[test]
    fn test_non_commutative_sub_different_keys() {
        let v1 = Value(1);
        let v2 = Value(2);

        let inst_a = Instruction::Sub {
            result: Value(10), lhs: v1, rhs: v2, ty: IrType::I32,
        };
        let inst_b = Instruction::Sub {
            result: Value(11), lhs: v2, rhs: v1, ty: IrType::I32,
        };

        let key_a = expression_key(&inst_a).unwrap();
        let key_b = expression_key(&inst_b).unwrap();
        assert_ne!(key_a, key_b, "Sub(v1,v2) and Sub(v2,v1) must differ");
    }

    #[test]
    fn test_commutative_mul_same_key() {
        let v1 = Value(3);
        let v2 = Value(5);

        let inst_a = Instruction::Mul {
            result: Value(10), lhs: v1, rhs: v2, ty: IrType::I64,
        };
        let inst_b = Instruction::Mul {
            result: Value(11), lhs: v2, rhs: v1, ty: IrType::I64,
        };

        assert_eq!(expression_key(&inst_a), expression_key(&inst_b));
    }

    #[test]
    fn test_different_opcode_different_key() {
        let v1 = Value(1);
        let v2 = Value(2);

        let add_inst = Instruction::Add {
            result: Value(10), lhs: v1, rhs: v2, ty: IrType::I32,
        };
        let mul_inst = Instruction::Mul {
            result: Value(11), lhs: v1, rhs: v2, ty: IrType::I32,
        };

        assert_ne!(
            expression_key(&add_inst),
            expression_key(&mul_inst),
            "Add and Mul must produce different keys"
        );
    }

    #[test]
    fn test_different_type_different_key() {
        let v1 = Value(1);
        let v2 = Value(2);

        let i32_add = Instruction::Add {
            result: Value(10), lhs: v1, rhs: v2, ty: IrType::I32,
        };
        let i64_add = Instruction::Add {
            result: Value(11), lhs: v1, rhs: v2, ty: IrType::I64,
        };

        assert_ne!(
            expression_key(&i32_add),
            expression_key(&i64_add),
            "Same op with different types must differ"
        );
    }

    // -----------------------------------------------------------------------
    // Side-effect exclusion tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_store_not_eligible() {
        let inst = Instruction::Store { value: Value(1), ptr: Value(2), store_ty: None };
        assert!(expression_key(&inst).is_none(), "Store must not be CSE-eligible");
    }

    #[test]
    fn test_call_not_eligible() {
        let inst = Instruction::Call {
            result: Some(Value(10)),
            callee: Callee::Direct("foo".to_string()),
            args: vec![Value(1)],
            return_ty: IrType::I32,
        };
        assert!(expression_key(&inst).is_none(), "Call must not be CSE-eligible");
    }

    #[test]
    fn test_load_not_eligible() {
        let inst = Instruction::Load {
            result: Value(10), ty: IrType::I32, ptr: Value(1),
        };
        assert!(expression_key(&inst).is_none(), "Load must not be CSE-eligible");
    }

    #[test]
    fn test_alloca_not_eligible() {
        let inst = Instruction::Alloca {
            result: Value(10), ty: IrType::I32, count: None,
        };
        assert!(expression_key(&inst).is_none(), "Alloca must not be CSE-eligible");
    }

    #[test]
    fn test_phi_not_eligible() {
        let inst = Instruction::Phi {
            result: Value(10), ty: IrType::I32, incoming: Vec::new(),
        };
        assert!(expression_key(&inst).is_none(), "Phi must not be CSE-eligible");
    }

    #[test]
    fn test_nop_not_eligible() {
        let inst = Instruction::Nop;
        assert!(expression_key(&inst).is_none(), "Nop must not be CSE-eligible");
    }

    // -----------------------------------------------------------------------
    // Local CSE tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_local_cse_eliminates_duplicate_add() {
        // %10 = add i32 %1, %2
        // %11 = add i32 %1, %2   <-- redundant, should be eliminated
        let instructions = vec![
            Instruction::Add {
                result: Value(10), lhs: Value(1), rhs: Value(2), ty: IrType::I32,
            },
            Instruction::Add {
                result: Value(11), lhs: Value(1), rhs: Value(2), ty: IrType::I32,
            },
        ];
        let mut func = make_single_block_function(instructions);
        let mut pass = CsePass::new();
        let changed = pass.run_on_function(&mut func);

        assert!(changed, "CSE should report a change");
        assert_eq!(
            func.blocks[0].instructions.len(), 1,
            "One of the two duplicate instructions should be removed"
        );
    }

    #[test]
    fn test_local_cse_commutative_add() {
        // %10 = add i32 %1, %2
        // %11 = add i32 %2, %1   <-- same as above (commutative)
        let instructions = vec![
            Instruction::Add {
                result: Value(10), lhs: Value(1), rhs: Value(2), ty: IrType::I32,
            },
            Instruction::Add {
                result: Value(11), lhs: Value(2), rhs: Value(1), ty: IrType::I32,
            },
        ];
        let mut func = make_single_block_function(instructions);
        let mut pass = CsePass::new();
        let changed = pass.run_on_function(&mut func);

        assert!(changed, "Commutative duplicate should be caught");
        assert_eq!(func.blocks[0].instructions.len(), 1);
    }

    #[test]
    fn test_local_cse_non_commutative_sub_preserved() {
        // %10 = sub i32 %1, %2
        // %11 = sub i32 %2, %1   <-- different (non-commutative)
        let instructions = vec![
            Instruction::Sub {
                result: Value(10), lhs: Value(1), rhs: Value(2), ty: IrType::I32,
            },
            Instruction::Sub {
                result: Value(11), lhs: Value(2), rhs: Value(1), ty: IrType::I32,
            },
        ];
        let mut func = make_single_block_function(instructions);
        let mut pass = CsePass::new();
        let changed = pass.run_on_function(&mut func);

        assert!(!changed, "Non-commutative Sub with swapped operands must be preserved");
        assert_eq!(func.blocks[0].instructions.len(), 2);
    }

    #[test]
    fn test_local_cse_stores_preserved() {
        // Two store instructions to the same address must both be preserved.
        let instructions = vec![
            Instruction::Store { value: Value(1), ptr: Value(5), store_ty: None },
            Instruction::Store { value: Value(2), ptr: Value(5), store_ty: None },
        ];
        let mut func = make_single_block_function(instructions);
        let mut pass = CsePass::new();
        let changed = pass.run_on_function(&mut func);

        assert!(!changed, "Store instructions must never be eliminated by CSE");
        assert_eq!(func.blocks[0].instructions.len(), 2);
    }

    #[test]
    fn test_local_cse_loads_preserved() {
        // Two load instructions from the same address must both be preserved
        // (conservative: no alias analysis).
        let instructions = vec![
            Instruction::Load { result: Value(10), ty: IrType::I32, ptr: Value(5) },
            Instruction::Load { result: Value(11), ty: IrType::I32, ptr: Value(5) },
        ];
        let mut func = make_single_block_function(instructions);
        let mut pass = CsePass::new();
        let changed = pass.run_on_function(&mut func);

        assert!(!changed, "Load instructions must be conservatively preserved");
        assert_eq!(func.blocks[0].instructions.len(), 2);
    }

    #[test]
    fn test_local_cse_no_change_all_unique() {
        // All unique expressions — no eliminations.
        let instructions = vec![
            Instruction::Add {
                result: Value(10), lhs: Value(1), rhs: Value(2), ty: IrType::I32,
            },
            Instruction::Sub {
                result: Value(11), lhs: Value(1), rhs: Value(2), ty: IrType::I32,
            },
            Instruction::Mul {
                result: Value(12), lhs: Value(1), rhs: Value(2), ty: IrType::I32,
            },
        ];
        let mut func = make_single_block_function(instructions);
        let mut pass = CsePass::new();
        let changed = pass.run_on_function(&mut func);

        assert!(!changed, "No duplicates means no changes");
        assert_eq!(func.blocks[0].instructions.len(), 3);
    }

    #[test]
    fn test_local_cse_value_replacement_in_subsequent_instruction() {
        // %10 = add i32 %1, %2
        // %11 = add i32 %1, %2   <-- eliminated, %11 -> %10
        // %12 = sub i32 %11, %3  <-- %11 should be replaced with %10
        let instructions = vec![
            Instruction::Add {
                result: Value(10), lhs: Value(1), rhs: Value(2), ty: IrType::I32,
            },
            Instruction::Add {
                result: Value(11), lhs: Value(1), rhs: Value(2), ty: IrType::I32,
            },
            Instruction::Sub {
                result: Value(12), lhs: Value(11), rhs: Value(3), ty: IrType::I32,
            },
        ];
        let mut func = make_single_block_function(instructions);
        let mut pass = CsePass::new();
        let changed = pass.run_on_function(&mut func);

        assert!(changed);
        assert_eq!(func.blocks[0].instructions.len(), 2);

        // The Sub should now reference %10 instead of %11.
        match &func.blocks[0].instructions[1] {
            Instruction::Sub { lhs, .. } => {
                assert_eq!(*lhs, Value(10), "Use of eliminated %11 should be replaced with %10");
            }
            other => panic!("Expected Sub, got {:?}", other),
        }
    }

    #[test]
    fn test_local_cse_calls_preserved() {
        // Two identical call instructions must both be preserved (side effects).
        let instructions = vec![
            Instruction::Call {
                result: Some(Value(10)),
                callee: Callee::Direct("foo".to_string()),
                args: vec![Value(1)],
                return_ty: IrType::I32,
            },
            Instruction::Call {
                result: Some(Value(11)),
                callee: Callee::Direct("foo".to_string()),
                args: vec![Value(1)],
                return_ty: IrType::I32,
            },
        ];
        let mut func = make_single_block_function(instructions);
        let mut pass = CsePass::new();
        let changed = pass.run_on_function(&mut func);

        assert!(!changed, "Call instructions must both be preserved");
        assert_eq!(func.blocks[0].instructions.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Global CSE tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_global_cse_entry_expr_available_in_both_branches() {
        // Expression in entry block should be available in both branches.
        //
        // entry: %10 = add i32 %1, %2
        // left:  %11 = add i32 %1, %2   <-- should be eliminated
        // right: %12 = add i32 %1, %2   <-- should be eliminated
        // merge: (empty)
        let cond = Value(0);

        let entry_insts = vec![Instruction::Add {
            result: Value(10), lhs: Value(1), rhs: Value(2), ty: IrType::I32,
        }];
        let left_insts = vec![Instruction::Add {
            result: Value(11), lhs: Value(1), rhs: Value(2), ty: IrType::I32,
        }];
        let right_insts = vec![Instruction::Add {
            result: Value(12), lhs: Value(1), rhs: Value(2), ty: IrType::I32,
        }];
        let merge_insts = vec![];

        let mut func = make_diamond_function(
            entry_insts, left_insts, right_insts, merge_insts, cond,
        );
        let mut pass = CsePass::new();
        let changed = pass.run_on_function(&mut func);

        assert!(changed, "Global CSE should eliminate dominated duplicates");
        // Entry block keeps its Add.
        assert_eq!(func.blocks[0].instructions.len(), 1);
        // Left and right branches should have the Add eliminated.
        assert_eq!(func.blocks[1].instructions.len(), 0);
        assert_eq!(func.blocks[2].instructions.len(), 0);
    }

    #[test]
    fn test_global_cse_non_dominating_branch_not_reused() {
        // Expression in left branch should NOT be reused in right branch
        // (left does not dominate right).
        //
        // entry: (empty)
        // left:  %10 = add i32 %1, %2
        // right: %11 = add i32 %1, %2  <-- must be preserved
        // merge: (empty)
        let cond = Value(0);

        let entry_insts = vec![];
        let left_insts = vec![Instruction::Add {
            result: Value(10), lhs: Value(1), rhs: Value(2), ty: IrType::I32,
        }];
        let right_insts = vec![Instruction::Add {
            result: Value(11), lhs: Value(1), rhs: Value(2), ty: IrType::I32,
        }];
        let merge_insts = vec![];

        let mut func = make_diamond_function(
            entry_insts, left_insts, right_insts, merge_insts, cond,
        );
        let mut pass = CsePass::new();
        let changed = pass.run_on_function(&mut func);

        // Neither branch dominates the other, so no global elimination.
        assert!(!changed, "Non-dominating branch expression must not be reused");
        assert_eq!(func.blocks[1].instructions.len(), 1);
        assert_eq!(func.blocks[2].instructions.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Phi node value replacement test
    // -----------------------------------------------------------------------

    #[test]
    fn test_phi_incoming_values_updated() {
        // After CSE replaces %11 with %10, phi nodes referencing %11 must be
        // updated.
        let bb0 = BlockId(0);
        let bb1 = BlockId(1);

        let mut entry = BasicBlock::new(bb0, "entry".to_string());
        entry.instructions = vec![
            Instruction::Add {
                result: Value(10), lhs: Value(1), rhs: Value(2), ty: IrType::I32,
            },
            Instruction::Add {
                result: Value(11), lhs: Value(1), rhs: Value(2), ty: IrType::I32,
            },
        ];
        entry.terminator = Some(Terminator::Branch { target: bb1 });
        entry.successors = vec![bb1];

        let mut merge = BasicBlock::new(bb1, "merge".to_string());
        merge.phi_nodes = vec![PhiNode {
            result: Value(20),
            ty: IrType::I32,
            incoming: vec![(Value(11), bb0)], // references the eliminated %11
        }];
        merge.terminator = Some(Terminator::Return { value: Some(Value(20)) });
        merge.predecessors = vec![bb0];

        let mut func = Function {
            name: "phi_test".to_string(),
            return_type: IrType::I32,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![entry, merge],
            entry_block: bb0,
            is_definition: true,
is_static: false,
is_weak: false,
        };

        let mut pass = CsePass::new();
        let changed = pass.run_on_function(&mut func);

        assert!(changed);
        // The phi incoming value should now reference %10 instead of %11.
        assert_eq!(func.blocks[1].phi_nodes[0].incoming[0].0, Value(10));
    }

    // -----------------------------------------------------------------------
    // Terminator value replacement tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_terminator_cond_branch_value_replaced() {
        // %10 = icmp eq %1, %2
        // %11 = icmp eq %1, %2   <-- eliminated, %11 -> %10
        // condbranch %11 then bb1 else bb2   <-- %11 should become %10
        let bb0 = BlockId(0);
        let bb1 = BlockId(1);
        let bb2 = BlockId(2);

        let mut entry = BasicBlock::new(bb0, "entry".to_string());
        entry.instructions = vec![
            Instruction::ICmp {
                result: Value(10), op: CompareOp::Equal,
                lhs: Value(1), rhs: Value(2), ty: IrType::I32,
            },
            Instruction::ICmp {
                result: Value(11), op: CompareOp::Equal,
                lhs: Value(1), rhs: Value(2), ty: IrType::I32,
            },
        ];
        entry.terminator = Some(Terminator::CondBranch {
            condition: Value(11),
            true_block: bb1,
            false_block: bb2,
        });
        entry.successors = vec![bb1, bb2];

        let mut left = BasicBlock::new(bb1, "left".to_string());
        left.terminator = Some(Terminator::Return { value: None });
        left.predecessors = vec![bb0];

        let mut right = BasicBlock::new(bb2, "right".to_string());
        right.terminator = Some(Terminator::Return { value: None });
        right.predecessors = vec![bb0];

        let mut func = Function {
            name: "cond_test".to_string(),
            return_type: IrType::Void,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![entry, left, right],
            entry_block: bb0,
            is_definition: true,
is_static: false,
is_weak: false,
        };

        let mut pass = CsePass::new();
        let changed = pass.run_on_function(&mut func);

        assert!(changed);
        // The condbranch condition should now reference %10.
        match &func.blocks[0].terminator {
            Some(Terminator::CondBranch { condition, .. }) => {
                assert_eq!(*condition, Value(10));
            }
            other => panic!("Expected CondBranch, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Changed flag tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_changed_flag_true_on_elimination() {
        let instructions = vec![
            Instruction::Xor {
                result: Value(10), lhs: Value(1), rhs: Value(2), ty: IrType::I32,
            },
            Instruction::Xor {
                result: Value(11), lhs: Value(2), rhs: Value(1), ty: IrType::I32,
            },
        ];
        let mut func = make_single_block_function(instructions);
        let mut pass = CsePass::new();
        assert!(pass.run_on_function(&mut func));
    }

    #[test]
    fn test_changed_flag_false_on_no_change() {
        let instructions = vec![
            Instruction::Add {
                result: Value(10), lhs: Value(1), rhs: Value(2), ty: IrType::I32,
            },
        ];
        let mut func = make_single_block_function(instructions);
        let mut pass = CsePass::new();
        assert!(!pass.run_on_function(&mut func));
    }

    #[test]
    fn test_empty_function_returns_false() {
        let mut func = Function {
            name: "empty".to_string(),
            return_type: IrType::Void,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: Vec::new(),
            entry_block: BlockId(0),
            is_definition: true,
is_static: false,
is_weak: false,
        };
        let mut pass = CsePass::new();
        assert!(!pass.run_on_function(&mut func));
    }

    #[test]
    fn test_non_definition_returns_false() {
        let mut func = Function {
            name: "extern_fn".to_string(),
            return_type: IrType::Void,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: Vec::new(),
            entry_block: BlockId(0),
            is_definition: false,
is_static: false,
is_weak: false,
        };
        let mut pass = CsePass::new();
        assert!(!pass.run_on_function(&mut func));
    }

    // -----------------------------------------------------------------------
    // Cast CSE tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_identical_casts_eliminated() {
        let instructions = vec![
            Instruction::Cast {
                result: Value(10), op: CastOp::ZExt,
                value: Value(1), from_ty: IrType::I32, to_ty: IrType::I64,
            },
            Instruction::Cast {
                result: Value(11), op: CastOp::ZExt,
                value: Value(1), from_ty: IrType::I32, to_ty: IrType::I64,
            },
        ];
        let mut func = make_single_block_function(instructions);
        let mut pass = CsePass::new();
        let changed = pass.run_on_function(&mut func);

        assert!(changed);
        assert_eq!(func.blocks[0].instructions.len(), 1);
    }

    #[test]
    fn test_different_cast_ops_preserved() {
        let instructions = vec![
            Instruction::Cast {
                result: Value(10), op: CastOp::ZExt,
                value: Value(1), from_ty: IrType::I32, to_ty: IrType::I64,
            },
            Instruction::Cast {
                result: Value(11), op: CastOp::SExt,
                value: Value(1), from_ty: IrType::I32, to_ty: IrType::I64,
            },
        ];
        let mut func = make_single_block_function(instructions);
        let mut pass = CsePass::new();
        let changed = pass.run_on_function(&mut func);

        assert!(!changed);
        assert_eq!(func.blocks[0].instructions.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Pass name test
    // -----------------------------------------------------------------------

    #[test]
    fn test_pass_name() {
        let pass = CsePass::new();
        assert_eq!(pass.name(), "cse");
    }

    // -----------------------------------------------------------------------
    // Select instruction not eligible for CSE
    // -----------------------------------------------------------------------

    #[test]
    fn test_select_not_eligible() {
        let inst = Instruction::Select {
            result: Value(10), condition: Value(1),
            true_val: Value(2), false_val: Value(3),
            ty: IrType::I32,
        };
        assert!(expression_key(&inst).is_none(), "Select must not be CSE-eligible");
    }

    // -----------------------------------------------------------------------
    // Signed vs unsigned div distinction test
    // -----------------------------------------------------------------------

    #[test]
    fn test_signed_unsigned_div_different_keys() {
        let v1 = Value(1);
        let v2 = Value(2);

        let signed_div = Instruction::Div {
            result: Value(10), lhs: v1, rhs: v2, ty: IrType::I32, is_signed: true,
        };
        let unsigned_div = Instruction::Div {
            result: Value(11), lhs: v1, rhs: v2, ty: IrType::I32, is_signed: false,
        };

        assert_ne!(
            expression_key(&signed_div),
            expression_key(&unsigned_div),
            "Signed and unsigned div must produce different keys"
        );
    }
}
