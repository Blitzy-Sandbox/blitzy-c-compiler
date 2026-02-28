//! SSA construction and destruction for the `bcc` compiler's intermediate
//! representation.
//!
//! This module implements:
//! - **Iterated Dominance Frontier Algorithm** for phi-node placement
//!   (Cytron, Ferrante, Rosen, Wegman, Zadeck — 1991)
//! - **Stack-based Variable Renaming** for establishing SSA naming conventions
//! - **Phi Node Simplification** for removing trivial phi nodes
//! - **SSA Verification** for validating well-formed SSA
//! - **SSA Destruction** via copy insertion for code generation
//! - **Critical Edge Splitting** as a prerequisite for SSA destruction
//!
//! # Algorithm Overview
//!
//! SSA construction transforms IR with multiple variable definitions across
//! branches into proper SSA form where every value is defined exactly once
//! and phi nodes merge values at control-flow join points.
//!
//! The algorithm proceeds in three major phases:
//! 1. **Identify promotable allocas** — allocas used only via load/store
//! 2. **Place phi nodes** — using iterated dominance frontiers
//! 3. **Rename variables** — using dominator tree preorder traversal
//!
//! # Performance
//!
//! The algorithm is O(N × |DF|) for phi placement and O(N) for renaming,
//! where N is the number of instructions and |DF| is the dominance frontier
//! size. This is efficient for the large functions found in real-world C
//! codebases such as the SQLite amalgamation (~230K LOC).
//!
//! # References
//!
//! - Cytron, R., et al., "Efficiently Computing Static Single Assignment
//!   Form and the Control Dependence Graph" (1991)
//! - Cooper, Harvey, Kennedy — "A Simple, Fast Dominance Algorithm" (2001)

use std::collections::{HashMap, HashSet, VecDeque};

use crate::ir::cfg::{
    BasicBlock, ControlFlowGraph, DominanceTree, PhiNode, Terminator,
    compute_dominance_tree, compute_dominance_frontiers,
};
use crate::ir::instructions::{BlockId, Instruction, Value};
use crate::ir::types::IrType;

/// Type alias for a variable in the SSA construction context.
///
/// A "variable" is identified by the [`Value`] produced by its `Alloca`
/// instruction. This value serves as the unique identifier for the memory
/// location throughout the SSA construction and destruction algorithms.
type Variable = Value;

// ===========================================================================
// SSA Construction — Public API
// ===========================================================================

/// Constructs SSA form for the given control flow graph.
///
/// This function promotes alloca/load/store patterns to SSA registers with
/// phi nodes at join points. Only "promotable" allocas are converted — those
/// whose address is never taken and are only accessed via direct load/store.
///
/// # Algorithm
///
/// 1. Identify all promotable `Alloca` instructions
/// 2. Collect blocks where each variable is defined (stored to)
/// 3. Compute dominance tree and dominance frontiers
/// 4. Place phi nodes using the iterated dominance frontier algorithm
/// 5. Rename variables using dominator tree preorder traversal
/// 6. Remove promoted alloca/load/store instructions
/// 7. Simplify trivial phi nodes
///
/// # Parameters
///
/// - `cfg` — The control flow graph to transform. Modified in-place.
///
/// # Post-conditions
///
/// After this function returns, the CFG is in valid SSA form:
/// - Every value use is dominated by its definition
/// - Phi nodes exist at all necessary join points
/// - Promoted allocas, loads, and stores have been removed
pub fn construct_ssa(cfg: &mut ControlFlowGraph) {
    if cfg.num_blocks() == 0 {
        return;
    }

    // Step 1: Find all promotable allocas and their types.
    let promotable = collect_promotable_allocas(cfg);
    if promotable.is_empty() {
        return;
    }

    let promoted_set: HashSet<Variable> = promotable.iter().map(|(v, _)| *v).collect();
    let alloca_types: HashMap<Variable, IrType> = promotable.into_iter().collect();

    // Step 2: Collect variable definition sites (blocks containing stores).
    let var_defs = collect_variable_definitions(cfg, &promoted_set);

    // Step 3: Compute dominance tree and dominance frontiers.
    let dom_tree = compute_dominance_tree(cfg);
    let dom_frontiers = compute_dominance_frontiers(cfg, &dom_tree);

    // Step 4: Place phi nodes using iterated dominance frontier algorithm.
    let phi_vars = place_phi_nodes(cfg, &var_defs, &dom_frontiers, &alloca_types);

    // Step 5: Rename variables via dominator tree preorder walk.
    let mut value_counter = find_max_value(cfg) + 1;
    rename_variables(
        cfg,
        &dom_tree,
        &phi_vars,
        &promoted_set,
        &mut value_counter,
    );

    // Step 6: Simplify trivial phi nodes.
    simplify_phi_nodes(cfg);
}

// ===========================================================================
// SSA Destruction — Public API
// ===========================================================================

/// Destroys SSA form by replacing phi nodes with copy instructions.
///
/// This is needed before register allocation in the code generation phase.
/// For each phi node `x = phi [a, pred1], [b, pred2]`, copy instructions
/// are inserted at the end of each predecessor block (before the terminator):
/// - In pred1: `x = copy a`
/// - In pred2: `x = copy b`
///
/// Critical edges are split first to ensure copies can be safely inserted
/// without affecting other control flow paths.
///
/// # Parameters
///
/// - `cfg` — The SSA-form CFG to transform out of SSA. Modified in-place.
///
/// # Post-conditions
///
/// After this function returns:
/// - All phi nodes have been removed
/// - Copy instructions implement the phi semantics
/// - Critical edges have been split where necessary
pub fn destruct_ssa(cfg: &mut ControlFlowGraph) {
    if cfg.num_blocks() == 0 {
        return;
    }

    // Split critical edges to ensure safe copy insertion.
    split_critical_edges(cfg);

    // Collect all phi node data before mutating.
    let block_ids: Vec<BlockId> = cfg.blocks().iter().map(|b| b.id).collect();
    for &block_id in &block_ids {
        let phi_nodes: Vec<PhiNode> = cfg.block(block_id).phi_nodes.clone();
        if phi_nodes.is_empty() {
            continue;
        }

        // For each phi node, insert copy instructions in predecessors.
        for phi in &phi_nodes {
            for &(value, pred_id) in &phi.incoming {
                let copy_inst = Instruction::Copy {
                    result: phi.result,
                    source: value,
                    ty: phi.ty.clone(),
                };
                // Insert the copy before the terminator in the predecessor block.
                cfg.block_mut(pred_id).instructions.push(copy_inst);
            }
        }

        // Remove all phi nodes from this block.
        cfg.block_mut(block_id).phi_nodes.clear();
    }
}

// ===========================================================================
// Internal — Value counter
// ===========================================================================

/// Finds the maximum `Value` ID used anywhere in the CFG.
///
/// Returns 0 if the CFG is empty. The caller should add 1 to get the next
/// available value ID for fresh value generation.
fn find_max_value(cfg: &ControlFlowGraph) -> u32 {
    let mut max_val: u32 = 0;

    for block in cfg.blocks() {
        // Scan phi nodes
        for phi in &block.phi_nodes {
            if phi.result.0 < u32::MAX {
                max_val = max_val.max(phi.result.0);
            }
            for &(v, _) in &phi.incoming {
                if v.0 < u32::MAX {
                    max_val = max_val.max(v.0);
                }
            }
        }

        // Scan instructions
        for inst in &block.instructions {
            if let Some(r) = inst.result() {
                if r.0 < u32::MAX {
                    max_val = max_val.max(r.0);
                }
            }
            for op in inst.operands() {
                if op.0 < u32::MAX {
                    max_val = max_val.max(op.0);
                }
            }
        }

        // Scan terminator
        if let Some(ref term) = block.terminator {
            match term {
                Terminator::CondBranch { condition, .. } => {
                    if condition.0 < u32::MAX {
                        max_val = max_val.max(condition.0);
                    }
                }
                Terminator::Return { value: Some(v) } => {
                    if v.0 < u32::MAX {
                        max_val = max_val.max(v.0);
                    }
                }
                Terminator::Switch { value, .. } => {
                    if value.0 < u32::MAX {
                        max_val = max_val.max(value.0);
                    }
                }
                _ => {}
            }
        }
    }

    max_val
}

// ===========================================================================
// Internal — Promotability analysis
// ===========================================================================

/// Collects all promotable alloca instructions from the CFG.
///
/// An alloca is considered for promotion if:
/// - It has no dynamic count (fixed-size allocation only)
/// - It passes the promotability check ([`is_promotable`])
///
/// Returns a list of `(alloca_result_value, alloca_type)` pairs.
fn collect_promotable_allocas(cfg: &ControlFlowGraph) -> Vec<(Value, IrType)> {
    let mut allocas: Vec<(Value, IrType)> = Vec::new();

    // Gather all fixed-size alloca instructions across all blocks.
    for block in cfg.blocks() {
        for inst in &block.instructions {
            if let Instruction::Alloca {
                result,
                ty,
                count: None,
            } = inst
            {
                allocas.push((*result, ty.clone()));
            }
        }
    }

    // Retain only those that are promotable.
    allocas.retain(|(val, _)| is_promotable(*val, cfg));
    allocas
}

/// Determines whether an alloca can be promoted to an SSA register.
///
/// An `Alloca` is promotable if:
/// - Its address is never taken (never passed to a Call, never stored into
///   another location, never used in `GetElementPtr`)
/// - It is only accessed via direct `Load` and `Store` instructions
/// - In `Store` instructions, it appears as the `ptr` operand, not `value`
///
/// Non-promotable allocas remain as memory operations (untouched by SSA).
fn is_promotable(alloca_val: Value, cfg: &ControlFlowGraph) -> bool {
    for block in cfg.blocks() {
        for inst in &block.instructions {
            match inst {
                // Loading from the alloca — acceptable use.
                Instruction::Load { ptr, .. } if *ptr == alloca_val => {
                    continue;
                }
                // Storing TO the alloca (ptr operand) — acceptable.
                // Storing the alloca's ADDRESS (value operand) — NOT promotable.
                Instruction::Store { value, ptr } => {
                    if *ptr == alloca_val {
                        continue;
                    }
                    if *value == alloca_val {
                        return false;
                    }
                }
                // The alloca definition itself — skip.
                Instruction::Alloca { result, .. } if *result == alloca_val => {
                    continue;
                }
                // Any other instruction that uses the alloca value means the
                // address escapes (function argument, GEP base, etc.).
                _ => {
                    if inst.uses_value(alloca_val) {
                        return false;
                    }
                }
            }
        }

        // Check terminator for alloca usage (address should never appear there).
        if let Some(ref term) = block.terminator {
            let used_in_term = match term {
                Terminator::CondBranch { condition, .. } => *condition == alloca_val,
                Terminator::Return { value: Some(v) } => *v == alloca_val,
                Terminator::Switch { value, .. } => *value == alloca_val,
                _ => false,
            };
            if used_in_term {
                return false;
            }
        }
    }

    true
}

// ===========================================================================
// Internal — Variable definition collection
// ===========================================================================

/// Collects the set of blocks where each promoted variable is defined.
///
/// A "definition" is a `Store` instruction targeting the variable's alloca.
/// Returns a map from each variable to the list of blocks containing stores.
fn collect_variable_definitions(
    cfg: &ControlFlowGraph,
    promoted_set: &HashSet<Variable>,
) -> HashMap<Variable, Vec<BlockId>> {
    let mut var_defs: HashMap<Variable, Vec<BlockId>> = HashMap::new();

    // Initialize empty definition lists for all promoted variables.
    for &var in promoted_set {
        var_defs.entry(var).or_default();
    }

    // Scan all blocks for Store instructions targeting promoted allocas.
    for block in cfg.blocks() {
        for inst in &block.instructions {
            if let Instruction::Store { ptr, .. } = inst {
                if promoted_set.contains(ptr) {
                    let defs = var_defs.entry(*ptr).or_default();
                    if !defs.contains(&block.id) {
                        defs.push(block.id);
                    }
                }
            }
        }
    }

    var_defs
}

// ===========================================================================
// Internal — Phi node placement (Iterated Dominance Frontier)
// ===========================================================================

/// Places phi nodes at join points using the iterated dominance frontier
/// algorithm (Cytron et al., 1991).
///
/// For each variable with definitions in blocks `D(v)`, the algorithm:
/// 1. Initializes a worklist with all definition blocks
/// 2. For each block in the worklist, examines its dominance frontier
/// 3. At each frontier block, inserts a phi node for the variable
/// 4. Adds the frontier block to the worklist (phi is a new definition)
///
/// Returns a mapping from `(BlockId, Variable)` to the phi node's result
/// [`Value`], enabling the renaming pass to correlate phi nodes with their
/// corresponding variables.
fn place_phi_nodes(
    cfg: &mut ControlFlowGraph,
    var_defs: &HashMap<Variable, Vec<BlockId>>,
    dom_frontiers: &HashMap<BlockId, HashSet<BlockId>>,
    alloca_types: &HashMap<Variable, IrType>,
) -> HashMap<(BlockId, Variable), Value> {
    let mut phi_vars: HashMap<(BlockId, Variable), Value> = HashMap::new();
    let mut next_val = find_max_value(cfg) + 1;

    for (&var, def_blocks) in var_defs {
        let ty = match alloca_types.get(&var) {
            Some(t) => t.clone(),
            None => continue,
        };

        // Worklist W: blocks whose dominance frontiers need examination.
        let mut worklist: VecDeque<BlockId> = VecDeque::new();
        // Tracks blocks where a phi has already been inserted for this variable.
        let mut inserted: HashSet<BlockId> = HashSet::new();
        // Tracks blocks already added to the worklist (avoids re-processing).
        let mut in_worklist: HashSet<BlockId> = HashSet::new();

        for &block_id in def_blocks {
            worklist.push_back(block_id);
            in_worklist.insert(block_id);
        }

        while let Some(block_id) = worklist.pop_front() {
            if let Some(frontier) = dom_frontiers.get(&block_id) {
                for &df_block in frontier {
                    if !inserted.contains(&df_block) {
                        // Generate a fresh Value for the phi node result.
                        let result = Value(next_val);
                        next_val += 1;

                        let phi = PhiNode {
                            result,
                            ty: ty.clone(),
                            incoming: Vec::new(),
                        };

                        cfg.block_mut(df_block).phi_nodes.push(phi);
                        phi_vars.insert((df_block, var), result);
                        inserted.insert(df_block);

                        // The phi node constitutes a new definition of the
                        // variable — add the block to the worklist so its
                        // dominance frontier is also examined.
                        if !in_worklist.contains(&df_block) {
                            worklist.push_back(df_block);
                            in_worklist.insert(df_block);
                        }
                    }
                }
            }
        }
    }

    phi_vars
}

// ===========================================================================
// Internal — Variable renaming infrastructure
// ===========================================================================

/// Classification of an instruction during the SSA renaming pass.
///
/// Each instruction in a block is classified into one of four categories
/// that determines how the renaming pass handles it.
enum RenameAction {
    /// Instruction is not related to any promoted variable — leave untouched.
    Keep,
    /// A promoted `Alloca` instruction — mark for deletion.
    DeleteAlloca,
    /// A `Load` from a promoted variable. The load result will be replaced
    /// with the current reaching definition from the variable's stack.
    ReplaceLoad {
        result: Value,
        variable: Variable,
    },
    /// A `Store` to a promoted variable. The stored value becomes the new
    /// reaching definition and is pushed onto the variable's stack.
    DeleteStore {
        stored_value: Value,
        variable: Variable,
    },
}

/// Classifies an instruction for the SSA renaming pass.
///
/// Pattern-matches the instruction to determine whether it involves a
/// promoted alloca variable and what rename action to take.
fn classify_instruction(
    inst: &Instruction,
    promoted_set: &HashSet<Variable>,
) -> RenameAction {
    match inst {
        Instruction::Alloca { result, .. } if promoted_set.contains(result) => {
            RenameAction::DeleteAlloca
        }
        Instruction::Load { result, ptr, .. } if promoted_set.contains(ptr) => {
            RenameAction::ReplaceLoad {
                result: *result,
                variable: *ptr,
            }
        }
        Instruction::Store { value, ptr } if promoted_set.contains(ptr) => {
            RenameAction::DeleteStore {
                stored_value: *value,
                variable: *ptr,
            }
        }
        _ => RenameAction::Keep,
    }
}

/// Resolves a value through the replacement chain.
///
/// Follows transitive replacements (e.g., `%2 → %5 → %8`) with a depth
/// limit of 100 to guard against cycles in degenerate cases.
fn resolve_value(val: Value, replace_map: &HashMap<Value, Value>) -> Value {
    let mut current = val;
    let mut depth = 0;
    while let Some(&replacement) = replace_map.get(&current) {
        if replacement == current || depth > 100 {
            break;
        }
        current = replacement;
        depth += 1;
    }
    current
}

// ===========================================================================
// Internal — Variable renaming pass
// ===========================================================================

/// Orchestrates the variable renaming pass over the CFG.
///
/// This function:
/// 1. Builds a reverse mapping from phi result values to variables
/// 2. Walks the dominator tree in preorder via [`rename_block`]
/// 3. Applies the accumulated value replacements globally
/// 4. Deletes promoted alloca, load, and store instructions
fn rename_variables(
    cfg: &mut ControlFlowGraph,
    dom_tree: &DominanceTree,
    phi_vars: &HashMap<(BlockId, Variable), Value>,
    promoted_set: &HashSet<Variable>,
    _value_counter: &mut u32,
) {
    // Per-variable definition stack: tracks the current reaching definition.
    let mut stacks: HashMap<Variable, Vec<Value>> = HashMap::new();
    for &var in promoted_set {
        stacks.insert(var, Vec::new());
    }

    // Accumulated replacement map: load_result → reaching_definition.
    let mut replace_map: HashMap<Value, Value> = HashMap::new();

    // Instructions to delete: block_id → sorted list of instruction indices.
    let mut to_delete: HashMap<BlockId, Vec<usize>> = HashMap::new();

    // Reverse mapping: phi_result_value → variable, so the renaming pass
    // knows which variable each phi node defines.
    let phi_to_var: HashMap<Value, Variable> = phi_vars
        .iter()
        .map(|(&(_blk, var), &phi_result)| (phi_result, var))
        .collect();

    // Kick off the recursive dominator-tree preorder walk from the entry.
    let entry = cfg.entry();
    rename_block(
        entry,
        cfg,
        dom_tree,
        &mut stacks,
        &mut replace_map,
        &mut to_delete,
        promoted_set,
        &phi_to_var,
    );

    // Apply accumulated replacements to all remaining instructions.
    apply_replacements(cfg, &replace_map);

    // Delete promoted alloca / load / store instructions.
    delete_instructions(cfg, &to_delete);
}

/// Recursive block-level SSA renaming in dominator tree preorder.
///
/// At each block, the function:
/// 1. Processes phi nodes — each defines a new SSA value for its variable
/// 2. Processes instructions — classifies and handles alloca/load/store
/// 3. Fills phi operands in successor blocks with current definitions
/// 4. Recurses into dominated children
/// 5. Restores stacks to their pre-block state
fn rename_block(
    block_id: BlockId,
    cfg: &mut ControlFlowGraph,
    dom_tree: &DominanceTree,
    stacks: &mut HashMap<Variable, Vec<Value>>,
    replace_map: &mut HashMap<Value, Value>,
    to_delete: &mut HashMap<BlockId, Vec<usize>>,
    promoted_set: &HashSet<Variable>,
    phi_to_var: &HashMap<Value, Variable>,
) {
    // Track how many definitions were pushed per variable in this block
    // so we can restore the stacks when we leave.
    let mut push_counts: HashMap<Variable, usize> = HashMap::new();

    // ------------------------------------------------------------------
    // Step 1: Process phi nodes — each phi defines a new SSA value.
    // ------------------------------------------------------------------
    let phi_defs: Vec<(Value, Option<Variable>)> = {
        let block = cfg.block(block_id);
        block
            .phi_nodes
            .iter()
            .map(|phi| {
                let var = phi_to_var.get(&phi.result).copied();
                (phi.result, var)
            })
            .collect()
    };

    for (phi_result, var_opt) in &phi_defs {
        if let Some(var) = var_opt {
            stacks.entry(*var).or_default().push(*phi_result);
            *push_counts.entry(*var).or_insert(0) += 1;
        }
    }

    // ------------------------------------------------------------------
    // Step 2: Process instructions — classify each and handle accordingly.
    // ------------------------------------------------------------------
    let instruction_actions: Vec<(usize, RenameAction)> = {
        let block = cfg.block(block_id);
        block
            .instructions
            .iter()
            .enumerate()
            .map(|(idx, inst)| (idx, classify_instruction(inst, promoted_set)))
            .collect()
    };

    for (idx, action) in instruction_actions {
        match action {
            RenameAction::Keep => {}
            RenameAction::DeleteAlloca => {
                to_delete.entry(block_id).or_default().push(idx);
            }
            RenameAction::ReplaceLoad { result, variable } => {
                // Replace the load result with the variable's current
                // reaching definition (top of its stack).
                let current = stacks
                    .get(&variable)
                    .and_then(|s| s.last())
                    .copied()
                    .unwrap_or_else(Value::undef);
                replace_map.insert(result, current);
                to_delete.entry(block_id).or_default().push(idx);
            }
            RenameAction::DeleteStore {
                stored_value,
                variable,
            } => {
                // The stored value becomes the new reaching definition.
                // Resolve it through the replacement map in case it was
                // itself a load result that has already been replaced.
                let resolved = resolve_value(stored_value, replace_map);
                stacks.entry(variable).or_default().push(resolved);
                *push_counts.entry(variable).or_insert(0) += 1;
                to_delete.entry(block_id).or_default().push(idx);
            }
        }
    }

    // ------------------------------------------------------------------
    // Step 3: Fill phi operands in successor blocks.
    // ------------------------------------------------------------------
    let successors: Vec<BlockId> = cfg.block(block_id).successors.clone();
    for &succ_id in &successors {
        let succ_block = cfg.block_mut(succ_id);
        for phi in &mut succ_block.phi_nodes {
            if let Some(&var) = phi_to_var.get(&phi.result) {
                let current = stacks
                    .get(&var)
                    .and_then(|s| s.last())
                    .copied()
                    .unwrap_or_else(Value::undef);
                phi.add_incoming(current, block_id);
            }
        }
    }

    // ------------------------------------------------------------------
    // Step 4: Recurse into dominated children.
    // ------------------------------------------------------------------
    let children: Vec<BlockId> = dom_tree.children(block_id).to_vec();
    for child_id in children {
        rename_block(
            child_id,
            cfg,
            dom_tree,
            stacks,
            replace_map,
            to_delete,
            promoted_set,
            phi_to_var,
        );
    }

    // ------------------------------------------------------------------
    // Step 5: Restore stacks to pre-block state.
    // ------------------------------------------------------------------
    for (var, count) in &push_counts {
        if let Some(stack) = stacks.get_mut(var) {
            for _ in 0..*count {
                stack.pop();
            }
        }
    }
}

// ===========================================================================
// Internal — Global value replacement
// ===========================================================================

/// Applies the accumulated replacement map to all values in the CFG.
///
/// Replaces values in:
/// - Phi node incoming values
/// - Instruction operands (via [`Instruction::replace_use`])
/// - Terminator operands (condition, return value, switch value)
fn apply_replacements(cfg: &mut ControlFlowGraph, replace_map: &HashMap<Value, Value>) {
    if replace_map.is_empty() {
        return;
    }

    for block in cfg.blocks_mut() {
        // Replace in phi node incoming values.
        for phi in &mut block.phi_nodes {
            for (val, _) in &mut phi.incoming {
                let resolved = resolve_value(*val, replace_map);
                if resolved != *val {
                    *val = resolved;
                }
            }
        }

        // Replace in instruction operands.
        for inst in &mut block.instructions {
            let operands: Vec<Value> = inst.operands();
            for op in operands {
                if replace_map.contains_key(&op) {
                    let resolved = resolve_value(op, replace_map);
                    if resolved != op {
                        inst.replace_use(op, resolved);
                    }
                }
            }
        }

        // Replace in terminator operands.
        if let Some(ref mut term) = block.terminator {
            replace_in_terminator(term, replace_map);
        }
    }
}

/// Replaces values in a terminator instruction according to the replacement map.
fn replace_in_terminator(term: &mut Terminator, replace_map: &HashMap<Value, Value>) {
    match term {
        Terminator::CondBranch { condition, .. } => {
            let resolved = resolve_value(*condition, replace_map);
            if resolved != *condition {
                *condition = resolved;
            }
        }
        Terminator::Return { value: Some(v) } => {
            let resolved = resolve_value(*v, replace_map);
            if resolved != *v {
                *v = resolved;
            }
        }
        Terminator::Switch { value, .. } => {
            let resolved = resolve_value(*value, replace_map);
            if resolved != *value {
                *value = resolved;
            }
        }
        _ => {}
    }
}

// ===========================================================================
// Internal — Instruction deletion
// ===========================================================================

/// Deletes instructions at the specified indices from their respective blocks.
///
/// Indices are sorted in descending order before removal so that earlier
/// removals do not shift the positions of later ones.
fn delete_instructions(cfg: &mut ControlFlowGraph, to_delete: &HashMap<BlockId, Vec<usize>>) {
    for (&block_id, indices) in to_delete {
        let mut sorted: Vec<usize> = indices.clone();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        sorted.dedup();

        let block = cfg.block_mut(block_id);
        for idx in sorted {
            if idx < block.instructions.len() {
                block.instructions.remove(idx);
            }
        }
    }
}

// ===========================================================================
// Internal — Phi node simplification
// ===========================================================================

/// Simplifies trivial phi nodes in a fixed-point loop.
///
/// A phi node is trivial if all its non-self-referential, non-undef incoming
/// values are identical. Trivial phi nodes are replaced with the single
/// unique value and removed.
///
/// The loop iterates because simplifying one phi may cause other phi nodes
/// to become trivial (cascading simplification).
fn simplify_phi_nodes(cfg: &mut ControlFlowGraph) {
    loop {
        let mut replacements: HashMap<Value, Value> = HashMap::new();

        for block in cfg.blocks() {
            for phi in &block.phi_nodes {
                if let Some(single_val) = trivial_phi_value(phi) {
                    replacements.insert(phi.result, single_val);
                }
            }
        }

        if replacements.is_empty() {
            break;
        }

        // Propagate the simplified values throughout the CFG.
        apply_replacements(cfg, &replacements);

        // Remove the now-trivial phi nodes.
        for block in cfg.blocks_mut() {
            block
                .phi_nodes
                .retain(|phi| !replacements.contains_key(&phi.result));
        }
    }
}

/// Determines whether a phi node is trivial.
///
/// Returns `Some(value)` if the phi has exactly one unique non-self,
/// non-undef incoming value (i.e., the phi is redundant). Returns `None`
/// if the phi is non-trivial and must be preserved.
fn trivial_phi_value(phi: &PhiNode) -> Option<Value> {
    if phi.incoming.is_empty() {
        return None;
    }

    let mut unique_val: Option<Value> = None;
    for &(val, _) in &phi.incoming {
        // Ignore self-references and undef sentinels.
        if val == phi.result || val == Value::undef() {
            continue;
        }
        match unique_val {
            None => unique_val = Some(val),
            Some(existing) if existing == val => continue,
            Some(_) => return None, // Multiple distinct values → non-trivial.
        }
    }

    unique_val
}

// ===========================================================================
// Internal — SSA verification
// ===========================================================================

/// Verifies that the CFG is in well-formed SSA.
///
/// Checks:
/// 1. Every value use is dominated by its definition
/// 2. Every phi node has exactly one operand per predecessor block
/// 3. Every phi incoming edge references an actual predecessor
///
/// Returns `Ok(())` if the SSA form is valid, or `Err` with a list of
/// violation descriptions. Useful for debugging and testing.
fn verify_ssa(cfg: &ControlFlowGraph) -> Result<(), Vec<String>> {
    let mut errors: Vec<String> = Vec::new();
    let dom_tree = compute_dominance_tree(cfg);

    // Build a definition map: Value → BlockId where it is defined.
    let mut def_block: HashMap<Value, BlockId> = HashMap::new();
    for block in cfg.blocks() {
        for phi in &block.phi_nodes {
            def_block.insert(phi.result, block.id);
        }
        for inst in &block.instructions {
            if let Some(result) = inst.result() {
                def_block.insert(result, block.id);
            }
        }
    }

    // Check 1: Every value use is dominated by its definition.
    for block in cfg.blocks() {
        for inst in &block.instructions {
            for op in inst.operands() {
                if op == Value::undef() {
                    continue;
                }
                if let Some(&def_blk) = def_block.get(&op) {
                    if !dom_tree.dominates(def_blk, block.id) {
                        errors.push(format!(
                            "Value({}) used in Block {} but defined in Block {} \
                             which does not dominate it",
                            op.0, block.id.0, def_blk.0
                        ));
                    }
                }
            }
        }
    }

    // Check 2: Phi node predecessor count and validity.
    for block in cfg.blocks() {
        let pred_count = block.predecessors.len();
        for phi in &block.phi_nodes {
            if phi.incoming.len() != pred_count {
                errors.push(format!(
                    "Block {} phi for Value({}) has {} incoming values \
                     but {} predecessors",
                    block.id.0, phi.result.0, phi.incoming.len(), pred_count
                ));
            }

            // Check 3: Every incoming edge references an actual predecessor.
            for &(_, src_block) in &phi.incoming {
                if !block.predecessors.contains(&src_block) {
                    errors.push(format!(
                        "Block {} phi for Value({}) has incoming from \
                         Block {} which is not a predecessor",
                        block.id.0, phi.result.0, src_block.0
                    ));
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// ===========================================================================
// Internal — Critical edge splitting
// ===========================================================================

/// Splits all critical edges in the CFG.
///
/// A critical edge runs from a block with multiple successors to a block
/// with multiple predecessors. Such edges must be split before SSA
/// destruction to ensure that copy instructions can be inserted on specific
/// edges without affecting other control flow paths.
///
/// For each critical edge `(A → B)`:
/// 1. A new intermediate block `C` is created with `C: br B`
/// 2. `A`'s branch target is updated from `B` to `C`
/// 3. Edge `A→B` is replaced with `A→C` and `C→B`
/// 4. Phi nodes in `B` are updated: incoming from `A` → incoming from `C`
fn split_critical_edges(cfg: &mut ControlFlowGraph) {
    // Collect all critical edges before mutating.
    let mut critical_edges: Vec<(BlockId, BlockId)> = Vec::new();

    for block in cfg.blocks() {
        if block.successors.len() > 1 {
            for &succ_id in &block.successors {
                let succ = cfg.block(succ_id);
                if succ.predecessors.len() > 1 {
                    critical_edges.push((block.id, succ_id));
                }
            }
        }
    }

    if critical_edges.is_empty() {
        return;
    }

    // Allocate block IDs for the new intermediate blocks.
    let mut next_block_id = cfg.num_blocks() as u32;

    for (from_id, to_id) in critical_edges {
        let new_block_id = BlockId(next_block_id);
        next_block_id += 1;

        // Create an intermediate block with an unconditional branch to the
        // original target.
        let mut new_block = BasicBlock::new(
            new_block_id,
            format!("split_{}_{}", from_id.0, to_id.0),
        );
        new_block.terminator = Some(Terminator::Branch { target: to_id });

        cfg.add_block(new_block);

        // Rewire edges: from→to becomes from→new, new→to.
        cfg.remove_edge(from_id, to_id);
        cfg.add_edge(from_id, new_block_id);
        cfg.add_edge(new_block_id, to_id);

        // Update the terminator of the source block to target the new block.
        replace_terminator_target(cfg.block_mut(from_id), to_id, new_block_id);

        // Update phi nodes in the target block: incoming from the source
        // block now comes from the new intermediate block.
        let to_block = cfg.block_mut(to_id);
        for phi in &mut to_block.phi_nodes {
            for (_, block_ref) in &mut phi.incoming {
                if *block_ref == from_id {
                    *block_ref = new_block_id;
                }
            }
        }
    }
}

/// Updates a terminator's branch target from `old_target` to `new_target`.
///
/// Handles all terminator variants that carry block targets:
/// `Branch`, `CondBranch`, and `Switch`.
fn replace_terminator_target(
    block: &mut BasicBlock,
    old_target: BlockId,
    new_target: BlockId,
) {
    if let Some(ref mut term) = block.terminator {
        match term {
            Terminator::Branch { target } => {
                if *target == old_target {
                    *target = new_target;
                }
            }
            Terminator::CondBranch {
                true_block,
                false_block,
                ..
            } => {
                if *true_block == old_target {
                    *true_block = new_target;
                }
                if *false_block == old_target {
                    *false_block = new_target;
                }
            }
            Terminator::Switch {
                default, cases, ..
            } => {
                if *default == old_target {
                    *default = new_target;
                }
                for (_, target) in cases {
                    if *target == old_target {
                        *target = new_target;
                    }
                }
            }
            _ => {}
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::instructions::Callee;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn val(id: u32) -> Value {
        Value(id)
    }

    fn bid(id: u32) -> BlockId {
        BlockId(id)
    }

    /// Builds a diamond CFG:
    ///
    /// ```text
    ///   entry(0): alloca %1, store %10→%1, cond_br → then(1), else(2)
    ///   then(1):  store %42→%1, br → merge(3)
    ///   else(2):  store %99→%1, br → merge(3)
    ///   merge(3): load %3←%1, ret %3
    /// ```
    fn make_diamond_cfg() -> ControlFlowGraph {
        let mut cfg = ControlFlowGraph::new(bid(0));

        let mut entry = BasicBlock::new(bid(0), "entry".to_string());
        entry.instructions.push(Instruction::Alloca {
            result: val(1),
            ty: IrType::I32,
            count: None,
        });
        entry.instructions.push(Instruction::Store {
            value: val(10),
            ptr: val(1),
        });
        entry.terminator = Some(Terminator::CondBranch {
            condition: val(20),
            true_block: bid(1),
            false_block: bid(2),
        });
        cfg.add_block(entry);

        let mut then_block = BasicBlock::new(bid(1), "then".to_string());
        then_block
            .instructions
            .push(Instruction::Store { value: val(42), ptr: val(1) });
        then_block.terminator = Some(Terminator::Branch { target: bid(3) });
        cfg.add_block(then_block);

        let mut else_block = BasicBlock::new(bid(2), "else".to_string());
        else_block
            .instructions
            .push(Instruction::Store { value: val(99), ptr: val(1) });
        else_block.terminator = Some(Terminator::Branch { target: bid(3) });
        cfg.add_block(else_block);

        let mut merge = BasicBlock::new(bid(3), "merge".to_string());
        merge.instructions.push(Instruction::Load {
            result: val(3),
            ty: IrType::I32,
            ptr: val(1),
        });
        merge.terminator = Some(Terminator::Return {
            value: Some(val(3)),
        });
        cfg.add_block(merge);

        cfg.add_edge(bid(0), bid(1));
        cfg.add_edge(bid(0), bid(2));
        cfg.add_edge(bid(1), bid(3));
        cfg.add_edge(bid(2), bid(3));

        cfg
    }

    // -----------------------------------------------------------------------
    // SSA construction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_cfg() {
        let mut cfg = ControlFlowGraph::new(bid(0));
        construct_ssa(&mut cfg);
        destruct_ssa(&mut cfg);
    }

    #[test]
    fn test_straight_line_ssa() {
        let mut cfg = ControlFlowGraph::new(bid(0));

        let mut entry = BasicBlock::new(bid(0), "entry".to_string());
        entry.instructions.push(Instruction::Alloca {
            result: val(1),
            ty: IrType::I32,
            count: None,
        });
        entry.instructions.push(Instruction::Store {
            value: val(10),
            ptr: val(1),
        });
        entry.instructions.push(Instruction::Load {
            result: val(2),
            ty: IrType::I32,
            ptr: val(1),
        });
        entry.terminator = Some(Terminator::Return {
            value: Some(val(2)),
        });
        cfg.add_block(entry);

        construct_ssa(&mut cfg);

        // No phi nodes — single block, no branches.
        assert!(cfg.block(bid(0)).phi_nodes.is_empty());
        // All promoted instructions removed.
        assert!(
            cfg.block(bid(0)).instructions.is_empty(),
            "Promoted alloca/store/load should be removed"
        );
        // Return should use the stored value directly.
        match &cfg.block(bid(0)).terminator {
            Some(Terminator::Return { value: Some(v) }) => {
                assert_eq!(*v, val(10), "Return should use the stored constant");
            }
            _ => panic!("Expected return terminator"),
        }
    }

    #[test]
    fn test_diamond_ssa() {
        let mut cfg = make_diamond_cfg();
        construct_ssa(&mut cfg);

        // Merge block should have exactly one phi.
        let merge = cfg.block(bid(3));
        assert_eq!(merge.phi_nodes.len(), 1, "Expected 1 phi at merge block");

        let phi = &merge.phi_nodes[0];
        assert_eq!(phi.incoming.len(), 2, "Phi should have 2 incoming edges");

        // Verify incoming values by predecessor.
        let incoming: HashMap<BlockId, Value> =
            phi.incoming.iter().map(|&(v, b)| (b, v)).collect();
        assert_eq!(incoming[&bid(1)], val(42), "Then branch should provide 42");
        assert_eq!(incoming[&bid(2)], val(99), "Else branch should provide 99");

        // Return should use the phi result.
        match &merge.terminator {
            Some(Terminator::Return { value: Some(v) }) => {
                assert_eq!(*v, phi.result, "Return should use phi result");
            }
            _ => panic!("Expected return terminator"),
        }

        // No promoted alloca/load/store should remain anywhere.
        for blk in cfg.blocks() {
            for inst in &blk.instructions {
                match inst {
                    Instruction::Alloca { result, .. } if *result == val(1) => {
                        panic!("Promoted alloca still present in block {}", blk.id.0);
                    }
                    Instruction::Load { ptr, .. } if *ptr == val(1) => {
                        panic!("Promoted load still present in block {}", blk.id.0);
                    }
                    Instruction::Store { ptr, .. } if *ptr == val(1) => {
                        panic!("Promoted store still present in block {}", blk.id.0);
                    }
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn test_loop_phi_placement() {
        let mut cfg = ControlFlowGraph::new(bid(0));

        let mut entry = BasicBlock::new(bid(0), "entry".to_string());
        entry.instructions.push(Instruction::Alloca {
            result: val(1),
            ty: IrType::I32,
            count: None,
        });
        entry.instructions.push(Instruction::Store {
            value: val(10),
            ptr: val(1),
        });
        entry.terminator = Some(Terminator::Branch { target: bid(1) });
        cfg.add_block(entry);

        let mut header = BasicBlock::new(bid(1), "header".to_string());
        header.instructions.push(Instruction::Load {
            result: val(2),
            ty: IrType::I32,
            ptr: val(1),
        });
        header.terminator = Some(Terminator::CondBranch {
            condition: val(20),
            true_block: bid(2),
            false_block: bid(3),
        });
        cfg.add_block(header);

        let mut body = BasicBlock::new(bid(2), "body".to_string());
        body.instructions.push(Instruction::Store {
            value: val(30),
            ptr: val(1),
        });
        body.terminator = Some(Terminator::Branch { target: bid(1) });
        cfg.add_block(body);

        let mut exit_block = BasicBlock::new(bid(3), "exit".to_string());
        exit_block.instructions.push(Instruction::Load {
            result: val(3),
            ty: IrType::I32,
            ptr: val(1),
        });
        exit_block.terminator = Some(Terminator::Return {
            value: Some(val(3)),
        });
        cfg.add_block(exit_block);

        cfg.add_edge(bid(0), bid(1));
        cfg.add_edge(bid(1), bid(2));
        cfg.add_edge(bid(1), bid(3));
        cfg.add_edge(bid(2), bid(1));

        construct_ssa(&mut cfg);

        // Loop header should have a phi node.
        let hdr = cfg.block(bid(1));
        assert_eq!(hdr.phi_nodes.len(), 1, "Expected phi at loop header");

        let phi = &hdr.phi_nodes[0];
        assert_eq!(phi.incoming.len(), 2);

        let incoming: HashMap<BlockId, Value> =
            phi.incoming.iter().map(|&(v, b)| (b, v)).collect();
        assert_eq!(incoming[&bid(0)], val(10), "Entry should provide initial value");
        assert_eq!(incoming[&bid(2)], val(30), "Loop body should provide updated value");
    }

    // -----------------------------------------------------------------------
    // Promotability tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_promotability_simple() {
        let mut cfg = ControlFlowGraph::new(bid(0));
        let mut entry = BasicBlock::new(bid(0), "entry".to_string());
        entry.instructions.push(Instruction::Alloca {
            result: val(1),
            ty: IrType::I32,
            count: None,
        });
        entry.instructions.push(Instruction::Store {
            value: val(10),
            ptr: val(1),
        });
        entry.instructions.push(Instruction::Load {
            result: val(2),
            ty: IrType::I32,
            ptr: val(1),
        });
        entry.terminator = Some(Terminator::Return {
            value: Some(val(2)),
        });
        cfg.add_block(entry);

        assert!(is_promotable(val(1), &cfg), "Simple alloca should be promotable");
    }

    #[test]
    fn test_promotability_address_taken() {
        let mut cfg = ControlFlowGraph::new(bid(0));
        let mut entry = BasicBlock::new(bid(0), "entry".to_string());
        entry.instructions.push(Instruction::Alloca {
            result: val(1),
            ty: IrType::I32,
            count: None,
        });
        entry.instructions.push(Instruction::Call {
            result: Some(val(5)),
            callee: Callee::Indirect(val(100)),
            args: vec![val(1)],
            return_ty: IrType::I32,
        });
        entry.terminator = Some(Terminator::Return {
            value: Some(val(5)),
        });
        cfg.add_block(entry);

        assert!(
            !is_promotable(val(1), &cfg),
            "Address-taken alloca should not be promotable"
        );
    }

    #[test]
    fn test_promotability_address_stored() {
        let mut cfg = ControlFlowGraph::new(bid(0));
        let mut entry = BasicBlock::new(bid(0), "entry".to_string());
        entry.instructions.push(Instruction::Alloca {
            result: val(1),
            ty: IrType::I32,
            count: None,
        });
        // Store the alloca ADDRESS as a value (escapes).
        entry.instructions.push(Instruction::Store {
            value: val(1),
            ptr: val(50),
        });
        entry.terminator = Some(Terminator::Return { value: None });
        cfg.add_block(entry);

        assert!(
            !is_promotable(val(1), &cfg),
            "Alloca whose address is stored should not be promotable"
        );
    }

    // -----------------------------------------------------------------------
    // Phi simplification and helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_trivial_phi_all_same() {
        let phi = PhiNode {
            result: val(5),
            ty: IrType::I32,
            incoming: vec![(val(10), bid(1)), (val(10), bid(2))],
        };
        assert_eq!(trivial_phi_value(&phi), Some(val(10)));
    }

    #[test]
    fn test_trivial_phi_different() {
        let phi = PhiNode {
            result: val(5),
            ty: IrType::I32,
            incoming: vec![(val(10), bid(1)), (val(20), bid(2))],
        };
        assert_eq!(trivial_phi_value(&phi), None);
    }

    #[test]
    fn test_trivial_phi_self_reference() {
        let phi = PhiNode {
            result: val(5),
            ty: IrType::I32,
            incoming: vec![(val(5), bid(1)), (val(10), bid(2))],
        };
        assert_eq!(trivial_phi_value(&phi), Some(val(10)));
    }

    #[test]
    fn test_resolve_value_chain() {
        let mut map = HashMap::new();
        map.insert(val(1), val(2));
        map.insert(val(2), val(3));
        map.insert(val(3), val(4));

        assert_eq!(resolve_value(val(1), &map), val(4));
        assert_eq!(resolve_value(val(5), &map), val(5));
        assert_eq!(resolve_value(val(4), &map), val(4));
    }

    // -----------------------------------------------------------------------
    // SSA destruction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_destruct_ssa_basic() {
        let mut cfg = ControlFlowGraph::new(bid(0));

        let mut pred1 = BasicBlock::new(bid(0), "pred1".to_string());
        pred1.terminator = Some(Terminator::Branch { target: bid(2) });
        cfg.add_block(pred1);

        let mut pred2 = BasicBlock::new(bid(1), "pred2".to_string());
        pred2.terminator = Some(Terminator::Branch { target: bid(2) });
        cfg.add_block(pred2);

        let mut merge = BasicBlock::new(bid(2), "merge".to_string());
        merge.phi_nodes.push(PhiNode {
            result: val(5),
            ty: IrType::I32,
            incoming: vec![(val(10), bid(0)), (val(20), bid(1))],
        });
        merge.terminator = Some(Terminator::Return {
            value: Some(val(5)),
        });
        cfg.add_block(merge);

        cfg.add_edge(bid(0), bid(2));
        cfg.add_edge(bid(1), bid(2));

        destruct_ssa(&mut cfg);

        // Phi should be removed.
        assert!(cfg.block(bid(2)).phi_nodes.is_empty());

        // pred1 gets Copy { result: 5, source: 10 }.
        let p1 = &cfg.block(bid(0)).instructions;
        assert_eq!(p1.len(), 1);
        match &p1[0] {
            Instruction::Copy { result, source, .. } => {
                assert_eq!(*result, val(5));
                assert_eq!(*source, val(10));
            }
            other => panic!("Expected Copy, got {:?}", other),
        }

        // pred2 gets Copy { result: 5, source: 20 }.
        let p2 = &cfg.block(bid(1)).instructions;
        assert_eq!(p2.len(), 1);
        match &p2[0] {
            Instruction::Copy { result, source, .. } => {
                assert_eq!(*result, val(5));
                assert_eq!(*source, val(20));
            }
            other => panic!("Expected Copy, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Critical edge splitting tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_critical_edge_split() {
        let mut cfg = ControlFlowGraph::new(bid(0));

        let mut entry = BasicBlock::new(bid(0), "entry".to_string());
        entry.terminator = Some(Terminator::CondBranch {
            condition: val(1),
            true_block: bid(1),
            false_block: bid(2),
        });
        cfg.add_block(entry);

        let mut a = BasicBlock::new(bid(1), "a".to_string());
        a.terminator = Some(Terminator::Branch { target: bid(2) });
        cfg.add_block(a);

        let mut join = BasicBlock::new(bid(2), "join".to_string());
        join.terminator = Some(Terminator::Return { value: None });
        cfg.add_block(join);

        cfg.add_edge(bid(0), bid(1));
        cfg.add_edge(bid(0), bid(2));
        cfg.add_edge(bid(1), bid(2));

        let before = cfg.num_blocks();
        split_critical_edges(&mut cfg);
        assert_eq!(cfg.num_blocks(), before + 1, "One split block expected");

        match &cfg.block(bid(0)).terminator {
            Some(Terminator::CondBranch {
                true_block,
                false_block,
                ..
            }) => {
                assert_eq!(*true_block, bid(1));
                assert_ne!(*false_block, bid(2), "Critical edge should be split");
                let split_id = *false_block;
                match &cfg.block(split_id).terminator {
                    Some(Terminator::Branch { target }) => {
                        assert_eq!(*target, bid(2), "Split block should jump to join");
                    }
                    _ => panic!("Split block should have unconditional branch"),
                }
            }
            _ => panic!("Expected CondBranch"),
        }
    }

    #[test]
    fn test_diamond_no_critical_edges() {
        let mut cfg = make_diamond_cfg();
        let before = cfg.num_blocks();
        split_critical_edges(&mut cfg);
        assert_eq!(
            cfg.num_blocks(),
            before,
            "Diamond CFG should have no critical edges"
        );
    }

    // -----------------------------------------------------------------------
    // SSA verification tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_ssa_after_construction() {
        let mut cfg = make_diamond_cfg();
        construct_ssa(&mut cfg);
        let result = verify_ssa(&cfg);
        assert!(result.is_ok(), "SSA should be valid: {:?}", result.err());
    }

    #[test]
    fn test_no_promotion_for_non_promotable() {
        let mut cfg = ControlFlowGraph::new(bid(0));
        let mut entry = BasicBlock::new(bid(0), "entry".to_string());
        entry.instructions.push(Instruction::Alloca {
            result: val(1),
            ty: IrType::I32,
            count: None,
        });
        entry.instructions.push(Instruction::Store {
            value: val(1),
            ptr: val(50),
        });
        entry.terminator = Some(Terminator::Return { value: None });
        cfg.add_block(entry);

        let count_before = cfg.block(bid(0)).instructions.len();
        construct_ssa(&mut cfg);
        let count_after = cfg.block(bid(0)).instructions.len();
        assert_eq!(
            count_before, count_after,
            "Non-promotable alloca should remain untouched"
        );
    }
}
