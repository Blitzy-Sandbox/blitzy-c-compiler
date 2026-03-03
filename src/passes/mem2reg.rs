//! Memory-to-Register Promotion (mem2reg) Optimization Pass
//!
//! This module implements the mem2reg pass — the single most impactful optimization
//! in the `bcc` compiler's pipeline. It converts memory-based IR (where local variables
//! are represented as `Alloca` + `Load` + `Store` sequences) into clean SSA form
//! (where local variables are SSA registers with phi nodes at join points).
//!
//! # Algorithm
//!
//! The pass uses the classical Cytron et al. (1991) algorithm:
//!
//! 1. **Promotability analysis**: Identify allocas whose address is never taken
//!    (only used by `Load` and `Store` as the pointer operand).
//! 2. **Phi node placement**: Use the iterated dominance frontier algorithm to
//!    determine where phi nodes are needed for each promotable alloca.
//! 3. **Variable renaming**: Walk the dominator tree in preorder, maintaining a
//!    stack of current definitions. Loads are replaced with the current value,
//!    stores push new values onto the stack, and phi node incoming edges are filled.
//! 4. **Cleanup**: Remove the promoted `Alloca`, `Load`, and `Store` instructions.
//! 5. **Trivial phi simplification**: Eliminate phi nodes where all incoming values
//!    are identical or self-referencing.
//!
//! # Correctness Invariants
//!
//! - Only promotes provably safe allocas (address never taken).
//! - After promotion, the resulting IR is valid SSA form (every use dominated by
//!   its definition).
//! - Phi nodes are placed at the minimal set of join points required by the
//!   iterated dominance frontier.
//!
//! # Performance
//!
//! The pass runs in O(N × α(N)) time where N is the number of IR instructions,
//! due to dominance computation and phi placement being near-linear for reducible
//! CFGs. Must handle large functions from the SQLite amalgamation efficiently.
//!
//! # References
//!
//! - Cytron, R., et al., "Efficiently Computing Static Single Assignment Form and
//!   the Control Dependence Graph" (1991)
//! - Cooper, Harvey, Kennedy, "A Simple, Fast Dominance Algorithm" (2001)
//!
//! # Zero External Dependencies
//!
//! Only imports from `std` and internal crate modules. No external crates.
//!
//! # Safety
//!
//! This module contains zero `unsafe` blocks. All operations use safe Rust.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::ir::cfg::{compute_dominance_frontiers, compute_dominance_tree};
use crate::ir::instructions::{BlockId, Instruction, Value};
use crate::ir::types::IrType;
use crate::ir::{ControlFlowGraph, DominanceTree, Function, PhiNode, Terminator};

use super::FunctionPass;

// ---------------------------------------------------------------------------
// AllocaInfo — per-alloca definition/use tracking
// ---------------------------------------------------------------------------

/// Tracks definition (store) and use (load) information for a single promotable
/// alloca instruction.
///
/// Populated by [`collect_alloca_info`] and consumed by [`place_phi_nodes`] and
/// [`rename_variables`] to drive the SSA construction algorithm.
struct AllocaInfo {
    /// The IR type of the promoted variable (the element type of the alloca,
    /// not the pointer type).
    ty: IrType,

    /// The set of blocks that contain `Store` instructions targeting this alloca.
    /// These are the "definition blocks" for the iterated dominance frontier
    /// algorithm — each store constitutes a new definition of the variable.
    def_blocks: HashSet<BlockId>,

    /// The set of blocks that contain `Load` instructions from this alloca.
    /// Used to determine which blocks require the promoted value.
    use_blocks: HashSet<BlockId>,
}

// ---------------------------------------------------------------------------
// Mem2RegPass — the public pass struct
// ---------------------------------------------------------------------------

/// Memory-to-register promotion optimization pass.
///
/// Converts stack-allocated local variables (`Alloca` + `Load` + `Store`) into
/// SSA registers with phi nodes at control-flow join points. This is the core
/// pass for producing clean SSA form and is the most impactful single
/// optimization in the pipeline.
///
/// # Usage
///
/// ```ignore
/// let mut pass = Mem2RegPass::new();
/// let changed = pass.run_on_function(&mut function);
/// ```
///
/// # Integration
///
/// Called by `pipeline.rs` at both `-O1` and `-O2` as the first pass.
pub struct Mem2RegPass;

impl Mem2RegPass {
    /// Creates a new `Mem2RegPass` instance.
    pub fn new() -> Self {
        Mem2RegPass
    }
}

impl FunctionPass for Mem2RegPass {
    /// Returns the human-readable name of this pass.
    fn name(&self) -> &str {
        "mem2reg"
    }

    /// Runs the mem2reg pass on the given function.
    ///
    /// # Algorithm
    ///
    /// 1. Finds all promotable allocas in the entry block.
    /// 2. Computes dominance tree and dominance frontiers.
    /// 3. For each promotable alloca: places phi nodes, renames variables.
    /// 4. Removes promoted alloca/load/store instructions.
    /// 5. Simplifies trivial phi nodes.
    ///
    /// # Returns
    ///
    /// `true` if any allocas were promoted (IR was modified), `false` otherwise.
    fn run_on_function(&mut self, function: &mut Function) -> bool {
        // Skip non-definitions (extern declarations have no blocks).
        if function.blocks.is_empty() || !function.is_definition {
            return false;
        }

        // Step 1: Find promotable allocas in the entry block.
        let promotable = find_promotable_allocas(function);
        if promotable.is_empty() {
            return false;
        }

        // Step 2: Build a CFG snapshot and compute dominance information.
        let cfg = build_cfg(function);
        let dom_tree = compute_dominance_tree(&cfg);
        let dom_frontiers = compute_dominance_frontiers(&cfg, &dom_tree);

        // Step 3: Find the next available Value ID for fresh phi node results.
        // Use wrapping_add to avoid overflow when undef values (u32::MAX) are
        // present in the function's IR — find_max_value_id filters them, but
        // defensive wrapping prevents panic in edge cases.
        let mut next_val = find_max_value_id(function).wrapping_add(1);

        // Step 4: Collect the set of promoted alloca values for cleanup.
        let promoted_set: HashSet<Value> =
            promotable.iter().map(|(v, _)| *v).collect();

        // Step 5: Global replacement map — load results → replacement values.
        let mut replacements: HashMap<Value, Value> = HashMap::new();

        // Step 6: Process each promotable alloca through the SSA construction
        // pipeline: collect info → place phi nodes → rename variables.
        for (alloca_val, alloca_ty) in &promotable {
            let info = collect_alloca_info(*alloca_val, alloca_ty.clone(), function);
            let phi_values =
                place_phi_nodes(&info, &dom_frontiers, function, &mut next_val);
            rename_variables(
                *alloca_val,
                &phi_values,
                &dom_tree,
                function,
                &mut replacements,
            );
        }

        // Step 7: Resolve transitive replacement chains.
        resolve_transitive_replacements(&mut replacements);

        // Step 8: Apply all accumulated value replacements across the IR.
        apply_replacements(function, &replacements);

        // Step 9: Remove promoted alloca, load, and store instructions.
        cleanup_promoted(function, &promoted_set, &replacements);

        // Step 10: Simplify trivial phi nodes produced by the promotion.
        simplify_trivial_phis(function);

        true
    }
}

// ===========================================================================
// Phase 1 — Promotability Analysis
// ===========================================================================

/// Scans the entry block for `Alloca` instructions and returns those that are
/// promotable (address never taken, only used by `Load` and `Store`).
///
/// # Promotability Criteria
///
/// An alloca is promotable if and only if:
/// 1. It has no dynamic count (scalar allocation only, `count == None`).
/// 2. Every use is either a `Load` (with the alloca as `ptr`) or a `Store`
///    (with the alloca as `ptr`, NOT as the stored `value`).
/// 3. The alloca's address never escapes via `Call`, `GetElementPtr`, `Cast`,
///    `BitCast`, phi node incoming values, or terminator operands.
fn find_promotable_allocas(function: &Function) -> Vec<(Value, IrType)> {
    let mut promotable = Vec::new();

    let entry_idx = function.entry_block.0 as usize;
    if entry_idx >= function.blocks.len() {
        return promotable;
    }

    // Build a map from Value → IrType for all instructions that produce typed
    // results. This is used below to verify type consistency between stored
    // values and the alloca's element type.
    let value_type_map = build_value_type_map(function);

    // Only scan the entry block for allocas. In well-formed IR from the builder,
    // all allocas reside in the entry block.
    for inst in &function.blocks[entry_idx].instructions {
        if let Instruction::Alloca {
            result,
            ty,
            count,
        } = inst
        {
            // Only promote scalar allocas (no dynamic array count).
            if count.is_some() {
                continue;
            }

            if is_alloca_promotable(*result, function) {
                // Additional safety check: verify that all stores to this alloca
                // write values whose types are compatible with the alloca's element
                // type. If a store writes an I32 value to an F64 alloca, promotion
                // would create a direct use of the I32 value where an F64 is
                // expected, causing register class mismatches at code generation.
                // This happens when the IR builder incorrectly resolves struct
                // member types — the alloca/store/load memory cycle acts as an
                // implicit bitcast that promotion eliminates unsafely.
                if stores_type_consistent(*result, ty, function, &value_type_map) {
                    promotable.push((*result, ty.clone()));
                }
            }
        }
    }

    promotable
}

/// Builds a mapping from Value → IrType for all defined values in the function.
///
/// Covers instruction results, phi node results, and function parameters.
/// Used by [`stores_type_consistent`] to check type compatibility of stored values.
fn build_value_type_map(function: &Function) -> HashMap<Value, IrType> {
    let mut map = HashMap::new();

    // Map parameter values to their declared types.
    for (i, pv) in function.param_values.iter().enumerate() {
        if let Some((_, ty)) = function.params.get(i) {
            map.insert(*pv, ty.clone());
        }
    }

    for block in &function.blocks {
        // Phi node results
        for phi in &block.phi_nodes {
            map.insert(phi.result, phi.ty.clone());
        }

        // Instruction results
        for inst in &block.instructions {
            if let Some(result) = inst.result() {
                if let Some(ty) = inst.result_type() {
                    map.insert(result, ty.clone());
                }
            }
        }
    }

    map
}

/// Checks that all stores to the given alloca write values whose register class
/// (integer vs float) matches the alloca's element type.
///
/// This prevents mem2reg from promoting allocas that act as implicit bitcasts
/// between integer and floating-point types. Such promotions create direct
/// references between incompatible register classes, causing panics in the
/// machine code encoder (e.g., MOVSD receiving a GPR operand).
fn stores_type_consistent(
    alloca_val: Value,
    alloca_ty: &IrType,
    function: &Function,
    value_type_map: &HashMap<Value, IrType>,
) -> bool {
    let alloca_is_float = alloca_ty.is_float();

    for block in &function.blocks {
        for inst in &block.instructions {
            if let Instruction::Store { value, ptr, .. } = inst {
                if *ptr == alloca_val {
                    // Check if the stored value's type has the same register class
                    // as the alloca's element type. If we can't determine the stored
                    // value's type, conservatively refuse promotion.
                    if let Some(stored_ty) = value_type_map.get(value) {
                        if stored_ty.is_float() != alloca_is_float {
                            // Type class mismatch: integer stored into float alloca
                            // or vice versa. Don't promote.
                            return false;
                        }
                    }
                    // If the stored value's type is unknown (e.g., from a Const with
                    // no IrType field), allow promotion — the alloca type is authoritative.
                }
            }
        }
    }

    true
}

/// Determines whether a specific alloca value is promotable by scanning all
/// instructions across all blocks for non-promotable uses.
///
/// An alloca is NOT promotable if any of the following is true:
/// - Its value appears as an argument to a `Call` (address escapes).
/// - Its value appears as an operand of `GetElementPtr` (address arithmetic).
/// - Its value is the stored `value` (not `ptr`) in a `Store` (address escapes).
/// - Its value appears in a `Cast` or `BitCast` (pointer conversion).
/// - Its value appears as a phi node incoming value.
/// - Its value appears in a terminator operand.
fn is_alloca_promotable(alloca_val: Value, function: &Function) -> bool {
    for block in &function.blocks {
        // Check phi nodes — if the alloca's address flows into a phi, it escapes.
        for phi in &block.phi_nodes {
            for (val, _) in &phi.incoming {
                if *val == alloca_val {
                    return false;
                }
            }
        }

        // Check each instruction for uses of the alloca value.
        for inst in &block.instructions {
            if !inst.uses_value(alloca_val) {
                continue;
            }

            match inst {
                // Loading from the alloca is always a valid promotable use.
                Instruction::Load { ptr, .. } if *ptr == alloca_val => {}

                // Storing TO the alloca (alloca_val is the pointer target) is valid,
                // but storing the alloca's ADDRESS as a value is NOT valid.
                Instruction::Store { ptr, value, .. }
                    if *ptr == alloca_val && *value != alloca_val => {}

                // Every other use is non-promotable (Call args, GEP, Cast, BitCast,
                // storing the address as a value, etc.).
                _ => {
                    return false;
                }
            }
        }

        // Check terminator operands — the alloca address must not appear there.
        if let Some(ref term) = block.terminator {
            let uses_alloca = match term {
                Terminator::CondBranch { condition, .. } => *condition == alloca_val,
                Terminator::Return { value: Some(v) } => *v == alloca_val,
                Terminator::Switch { value, .. } => *value == alloca_val,
                _ => false,
            };
            if uses_alloca {
                return false;
            }
        }
    }

    true
}

// ===========================================================================
// Phase 2 — Information Collection
// ===========================================================================

/// Collects definition (store) and use (load) information for a single promotable
/// alloca, scanning all blocks in the function.
///
/// The returned [`AllocaInfo`] drives phi node placement via the iterated dominance
/// frontier algorithm: `def_blocks` are the initial worklist entries, and `use_blocks`
/// indicate where the promoted value is read.
fn collect_alloca_info(
    alloca_val: Value,
    alloca_ty: IrType,
    function: &Function,
) -> AllocaInfo {
    let mut info = AllocaInfo {
        ty: alloca_ty,
        def_blocks: HashSet::new(),
        use_blocks: HashSet::new(),
    };

    for block in &function.blocks {
        for inst in &block.instructions {
            match inst {
                Instruction::Store { ptr, .. } if *ptr == alloca_val => {
                    info.def_blocks.insert(block.id);
                }
                Instruction::Load { ptr, .. } if *ptr == alloca_val => {
                    info.use_blocks.insert(block.id);
                }
                _ => {}
            }
        }
    }

    info
}

// ===========================================================================
// Phase 3 — Phi Node Placement (Iterated Dominance Frontier)
// ===========================================================================

/// Places phi nodes at the iterated dominance frontier of the alloca's definition
/// blocks, using the Cytron et al. (1991) algorithm.
///
/// # Algorithm
///
/// 1. Initialize a worklist `W` with all definition blocks (blocks containing
///    stores to the alloca).
/// 2. Maintain a set `F` of blocks where phi nodes have been placed.
/// 3. Pop a block `X` from the worklist. For each block `Y` in DF(X):
///    - If no phi has been placed at `Y`: insert one, add `Y` to `F`.
///    - If `Y` was not already a definition block, add it to the worklist
///      (the phi constitutes a new definition).
/// 4. Repeat until the worklist is empty.
///
/// # Returns
///
/// A map from `BlockId` to the `Value` produced by the phi node placed at that
/// block. The phi nodes' incoming lists are initially empty; they are filled
/// during the rename phase.
fn place_phi_nodes(
    info: &AllocaInfo,
    dom_frontiers: &HashMap<BlockId, HashSet<BlockId>>,
    function: &mut Function,
    next_val: &mut u32,
) -> HashMap<BlockId, Value> {
    let mut phi_blocks: HashMap<BlockId, Value> = HashMap::new();
    let mut has_phi: HashSet<BlockId> = HashSet::new();

    // Initialize the worklist with all definition blocks.
    let mut worklist: VecDeque<BlockId> =
        VecDeque::from(info.def_blocks.iter().copied().collect::<Vec<_>>());

    while let Some(x) = worklist.pop_front() {
        // Iterate over the dominance frontier of X.
        if let Some(df) = dom_frontiers.get(&x) {
            for &y in df {
                if has_phi.contains(&y) {
                    continue;
                }

                // Allocate a fresh Value for the phi node's result.
                let phi_result = Value(*next_val);
                *next_val = next_val.wrapping_add(1);

                // Create the phi node with empty incoming list.
                let phi_node = PhiNode {
                    result: phi_result,
                    ty: info.ty.clone(),
                    incoming: Vec::new(),
                };

                // Insert the phi node at the target block.
                let block_idx = y.0 as usize;
                if block_idx < function.blocks.len() {
                    function.blocks[block_idx].phi_nodes.push(phi_node);
                }

                phi_blocks.insert(y, phi_result);
                has_phi.insert(y);

                // A phi node is itself a definition. If Y was not already a
                // definition block, add it to the worklist for further frontier
                // propagation.
                if !info.def_blocks.contains(&y) {
                    worklist.push_back(y);
                }
            }
        }
    }

    phi_blocks
}

// ===========================================================================
// Phase 4 — Variable Renaming (Dominator Tree Walk)
// ===========================================================================

/// Renames variables for a single alloca by walking the dominator tree in preorder.
///
/// Maintains a stack of current SSA values representing the most recent definition
/// of the promoted variable along the current dominator tree path. Loads are
/// replaced (via the `replacements` map), stores push new values onto the stack,
/// and successor phi nodes receive incoming edges.
fn rename_variables(
    alloca_val: Value,
    phi_values: &HashMap<BlockId, Value>,
    dom_tree: &DominanceTree,
    function: &mut Function,
    replacements: &mut HashMap<Value, Value>,
) {
    let entry = function.entry_block;
    let mut value_stack: Vec<Value> = Vec::new();

    rename_block(
        entry,
        alloca_val,
        phi_values,
        dom_tree,
        function,
        &mut value_stack,
        replacements,
    );
}

/// Recursive workhorse for variable renaming within a single block and its
/// dominated children.
///
/// # Stack Protocol
///
/// On entry, `value_stack` reflects the reaching definition from the dominator
/// path above this block. The function:
/// 1. Pushes the phi result (if any) for this block.
/// 2. Scans instructions: loads record replacements, stores push new values.
/// 3. Fills successor phi nodes with the current reaching definition.
/// 4. Recurses into dominated children.
/// 5. Restores the stack to its entry height (backtrack).
fn rename_block(
    block_id: BlockId,
    alloca_val: Value,
    phi_values: &HashMap<BlockId, Value>,
    dom_tree: &DominanceTree,
    function: &mut Function,
    value_stack: &mut Vec<Value>,
    replacements: &mut HashMap<Value, Value>,
) {
    let stack_height = value_stack.len();
    let block_idx = block_id.0 as usize;

    if block_idx >= function.blocks.len() {
        return;
    }

    // --- Step 1: If this block has a phi node for this alloca, push its result
    //     as the new reaching definition. ---
    if let Some(&phi_val) = phi_values.get(&block_id) {
        value_stack.push(phi_val);
    }

    // --- Step 2: Process instructions in order (read-only scan). ---
    // We collect local definitions (stores) and load replacements without
    // mutably borrowing function.blocks during iteration.
    let mut local_defs: Vec<Value> = Vec::new();
    let mut local_replacements: Vec<(Value, Value)> = Vec::new();

    {
        let instructions = &function.blocks[block_idx].instructions;
        for inst in instructions {
            match inst {
                Instruction::Load { result, ptr, .. } if *ptr == alloca_val => {
                    // The current reaching definition replaces this load.
                    let current = local_defs
                        .last()
                        .copied()
                        .or_else(|| value_stack.last().copied())
                        .unwrap_or(Value::undef());
                    local_replacements.push((*result, current));
                }
                Instruction::Store { value, ptr, .. } if *ptr == alloca_val => {
                    // A new definition of the variable.
                    local_defs.push(*value);
                }
                _ => {}
            }
        }
    }

    // Record all replacements.
    for (from, to) in local_replacements {
        replacements.insert(from, to);
    }

    // Push all local definitions onto the value stack so successors see the
    // latest definition.
    for &def in &local_defs {
        value_stack.push(def);
    }

    // --- Step 3: Fill successor phi nodes. ---
    // The current reaching definition is the top of the value stack.
    let current_val = value_stack
        .last()
        .copied()
        .unwrap_or(Value::undef());

    // Extract successors from the terminator (the canonical source of truth).
    // block.successors is not populated by the IR builder, so we must read
    // the targets from the terminator directly — same approach as build_cfg.
    let successors: Vec<BlockId> = match &function.blocks[block_idx].terminator {
        Some(crate::ir::Terminator::Branch { target }) => vec![*target],
        Some(crate::ir::Terminator::CondBranch { true_block, false_block, .. }) => {
            vec![*true_block, *false_block]
        }
        Some(crate::ir::Terminator::Switch { default, cases, .. }) => {
            let mut targets = vec![*default];
            for &(_, t) in cases { targets.push(t); }
            targets
        }
        _ => Vec::new(),
    };

    for succ_id in successors {
        if let Some(&_phi_result) = phi_values.get(&succ_id) {
            let succ_idx = succ_id.0 as usize;
            if succ_idx < function.blocks.len() {
                for phi in &mut function.blocks[succ_idx].phi_nodes {
                    if phi_values.get(&succ_id) == Some(&phi.result) {
                        phi.add_incoming(current_val, block_id);
                        break;
                    }
                }
            }
        }
    }

    // --- Step 4: Recurse into dominated children. ---
    let children: Vec<BlockId> = dom_tree.children(block_id).to_vec();
    for child in children {
        rename_block(
            child,
            alloca_val,
            phi_values,
            dom_tree,
            function,
            value_stack,
            replacements,
        );
    }

    // --- Step 5: Restore the value stack to its entry height. ---
    value_stack.truncate(stack_height);
}

// ===========================================================================
// Phase 5 — Replacement Application and Cleanup
// ===========================================================================

/// Resolves transitive chains in the replacement map.
///
/// If `%a → %b` and `%b → %c` exist in the map, this function updates
/// `%a → %c` so that a single lookup suffices during replacement application.
fn resolve_transitive_replacements(replacements: &mut HashMap<Value, Value>) {
    let mut changed = true;
    while changed {
        changed = false;
        let keys: Vec<Value> = replacements.keys().copied().collect();
        for key in keys {
            let val = replacements[&key];
            if let Some(&deeper) = replacements.get(&val) {
                if deeper != val && deeper != key {
                    replacements.insert(key, deeper);
                    changed = true;
                }
            }
        }
    }
}

/// Applies all accumulated value replacements to every instruction, phi node,
/// and terminator in the function.
///
/// This is the final step that rewrites all consumers of load results to use
/// the SSA values produced by phi nodes or directly by store values.
fn apply_replacements(function: &mut Function, replacements: &HashMap<Value, Value>) {
    if replacements.is_empty() {
        return;
    }

    for block in &mut function.blocks {
        // Replace in phi node incoming values.
        for phi in &mut block.phi_nodes {
            for (val, _) in &mut phi.incoming {
                if let Some(&replacement) = replacements.get(val) {
                    *val = replacement;
                }
            }
        }

        // Replace in instruction operands.
        for inst in &mut block.instructions {
            for operand in inst.operands_mut() {
                if let Some(&replacement) = replacements.get(operand) {
                    *operand = replacement;
                }
            }
        }

        // Replace in terminator operands.
        if let Some(ref mut term) = block.terminator {
            match term {
                Terminator::CondBranch { condition, .. } => {
                    if let Some(&replacement) = replacements.get(condition) {
                        *condition = replacement;
                    }
                }
                Terminator::Return { value: Some(v) } => {
                    if let Some(&replacement) = replacements.get(v) {
                        *v = replacement;
                    }
                }
                Terminator::Switch { value, .. } => {
                    if let Some(&replacement) = replacements.get(value) {
                        *value = replacement;
                    }
                }
                _ => {}
            }
        }
    }
}

/// Removes all `Alloca`, `Load`, and `Store` instructions associated with
/// promoted allocas.
///
/// - Allocas: removed if their result is in `promoted_set`.
/// - Loads: removed if their result is in `replacements` (meaning they loaded
///   from a promoted alloca and were replaced during renaming).
/// - Stores: removed if their `ptr` operand is in `promoted_set`.
fn cleanup_promoted(
    function: &mut Function,
    promoted_set: &HashSet<Value>,
    replacements: &HashMap<Value, Value>,
) {
    for block in &mut function.blocks {
        block.instructions.retain(|inst| {
            match inst {
                Instruction::Alloca { result, .. } => {
                    !promoted_set.contains(result)
                }
                Instruction::Load { result, .. } => {
                    !replacements.contains_key(result)
                }
                Instruction::Store { ptr, .. } => {
                    !promoted_set.contains(ptr)
                }
                _ => true,
            }
        });
    }
}

// ===========================================================================
// Phase 6 — Trivial Phi Simplification
// ===========================================================================

/// Iteratively simplifies trivial phi nodes until no more simplification is
/// possible.
///
/// A phi node is trivial if:
/// - All incoming values are identical → replace with that single value.
/// - All incoming values are either the phi itself or one unique value →
///   replace with that unique value.
/// - All incoming values are the phi itself → replace with `undef`.
///
/// Simplification may cascade: removing one trivial phi may make another
/// trivial (if the removed phi was the sole non-self incoming value).
fn simplify_trivial_phis(function: &mut Function) {
    let mut outer_changed = true;
    while outer_changed {
        outer_changed = false;
        let mut phi_replacements: HashMap<Value, Value> = HashMap::new();

        // Scan all phi nodes for trivial ones.
        for block in &function.blocks {
            for phi in &block.phi_nodes {
                if let Some(replacement) = trivial_phi_value(phi) {
                    phi_replacements.insert(phi.result, replacement);
                }
            }
        }

        if !phi_replacements.is_empty() {
            outer_changed = true;

            // Resolve transitive replacements within the phi map.
            resolve_transitive_replacements(&mut phi_replacements);

            // Apply replacements throughout the IR.
            apply_replacements(function, &phi_replacements);

            // Remove the trivial phi nodes.
            for block in &mut function.blocks {
                block
                    .phi_nodes
                    .retain(|phi| !phi_replacements.contains_key(&phi.result));
            }
        }
    }
}

/// Determines whether a phi node is trivial and returns the replacement value
/// if so.
///
/// A phi is trivial if, ignoring self-references and undef values, there is at
/// most one unique incoming value.
fn trivial_phi_value(phi: &PhiNode) -> Option<Value> {
    if phi.incoming.is_empty() {
        // No incoming edges — undefined reaching definition.
        return Some(Value::undef());
    }

    let mut unique_val: Option<Value> = None;

    for &(val, _) in &phi.incoming {
        // Ignore self-references (phi referring to itself in a loop).
        if val == phi.result {
            continue;
        }
        // Ignore undef values — they are dominated by any real definition.
        if val == Value::undef() {
            continue;
        }

        match unique_val {
            None => unique_val = Some(val),
            Some(existing) if existing == val => {
                // Same value — still trivial.
            }
            Some(_) => {
                // Two distinct non-self values — not trivial.
                return None;
            }
        }
    }

    match unique_val {
        Some(val) => Some(val),
        None => Some(Value::undef()), // All self-references or undef.
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Constructs a [`ControlFlowGraph`] from the function's block list for
/// dominance computation.
///
/// The blocks are cloned into the CFG to avoid lifetime conflicts with the
/// mutable function reference needed during phi node insertion and renaming.
fn build_cfg(function: &Function) -> ControlFlowGraph {
    let mut cfg = ControlFlowGraph::new(function.entry_block);
    for block in &function.blocks {
        cfg.add_block(block.clone());
    }
    // The IR builder does not populate block.successors/predecessors, so we
    // must extract edges from block terminators. Without these edges, the
    // dominance tree computation would treat non-entry blocks as unreachable,
    // preventing mem2reg from renaming variables across blocks.
    for block in &function.blocks {
        let from = block.id;
        if let Some(ref term) = block.terminator {
            match term {
                crate::ir::Terminator::Branch { target } => {
                    cfg.add_edge(from, *target);
                }
                crate::ir::Terminator::CondBranch { true_block, false_block, .. } => {
                    cfg.add_edge(from, *true_block);
                    cfg.add_edge(from, *false_block);
                }
                crate::ir::Terminator::Switch { default, cases, .. } => {
                    cfg.add_edge(from, *default);
                    for &(_, target) in cases {
                        cfg.add_edge(from, target);
                    }
                }
                _ => {} // Return, Unreachable — no successors
            }
        }
    }
    cfg
}

/// Finds the maximum `Value` ID currently in use across all instructions,
/// phi nodes, and terminators in the function.
///
/// Used to allocate fresh `Value` IDs for phi nodes without colliding with
/// existing values.
fn find_max_value_id(function: &Function) -> u32 {
    let mut max_val: u32 = 0;
    let undef_id = Value::undef().0;

    for block in &function.blocks {
        // Scan phi nodes — filter undef (u32::MAX) from both results and incoming.
        for phi in &block.phi_nodes {
            if phi.result.0 != undef_id {
                max_val = max_val.max(phi.result.0);
            }
            for &(v, _) in &phi.incoming {
                if v.0 != undef_id {
                    max_val = max_val.max(v.0);
                }
            }
        }

        // Scan instructions.
        for inst in &block.instructions {
            if let Some(result) = inst.result() {
                if result.0 != undef_id {
                    max_val = max_val.max(result.0);
                }
            }
            for op in inst.operands() {
                if op.0 != undef_id {
                    max_val = max_val.max(op.0);
                }
            }
        }

        // Scan terminator operands.
        if let Some(ref term) = block.terminator {
            match term {
                Terminator::CondBranch { condition, .. } => {
                    if *condition != Value::undef() {
                        max_val = max_val.max(condition.0);
                    }
                }
                Terminator::Return { value: Some(v) } => {
                    if *v != Value::undef() {
                        max_val = max_val.max(v.0);
                    }
                }
                Terminator::Switch { value, .. } => {
                    if *value != Value::undef() {
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
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::cfg::Terminator;
    use crate::ir::instructions::{Callee, Constant, Instruction};
    use crate::ir::BasicBlock;

    // -----------------------------------------------------------------------
    // Test Helpers
    // -----------------------------------------------------------------------

    /// Sets up predecessor/successor edges on a Function's blocks based on
    /// their terminators.
    fn setup_edges(function: &mut Function) {
        for block in &mut function.blocks {
            block.predecessors.clear();
            block.successors.clear();
        }

        let mut edges: Vec<(BlockId, BlockId)> = Vec::new();
        for block in &function.blocks {
            if let Some(ref term) = block.terminator {
                for succ in term.successors() {
                    edges.push((block.id, succ));
                }
            }
        }

        for (from, to) in edges {
            let from_idx = from.0 as usize;
            let to_idx = to.0 as usize;
            if from_idx < function.blocks.len() && to_idx < function.blocks.len() {
                if !function.blocks[from_idx].successors.contains(&to) {
                    function.blocks[from_idx].successors.push(to);
                }
                if !function.blocks[to_idx].predecessors.contains(&from) {
                    function.blocks[to_idx].predecessors.push(from);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Promotability Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_simple_alloca_is_promotable() {
        let entry = BlockId(0);
        let mut block = BasicBlock::new(entry, "entry".to_string());
        block.instructions = vec![
            Instruction::Alloca {
                result: Value(0),
                ty: IrType::I32,
                count: None,
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer {
                    value: 42,
                    ty: IrType::I32,
                },
            },
            Instruction::Store { value: Value(1), ptr: Value(0), store_ty: None },
            Instruction::Load {
                result: Value(2),
                ty: IrType::I32,
                ptr: Value(0),
            },
        ];
        block.terminator = Some(Terminator::Return {
            value: Some(Value(2)),
        });

        let function = Function {
            name: "test".to_string(),
            return_type: IrType::I32,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![block],
            entry_block: entry,
            is_definition: true,
is_static: false,
is_weak: false,
        };

        let promotable = find_promotable_allocas(&function);
        assert_eq!(promotable.len(), 1);
        assert_eq!(promotable[0].0, Value(0));
    }

    #[test]
    fn test_alloca_passed_to_call_not_promotable() {
        let entry = BlockId(0);
        let mut block = BasicBlock::new(entry, "entry".to_string());
        block.instructions = vec![
            Instruction::Alloca {
                result: Value(0),
                ty: IrType::I32,
                count: None,
            },
            Instruction::Call {
                result: None,
                callee: Callee::Direct("foo".to_string()),
                args: vec![Value(0)],
                return_ty: IrType::Void,
            },
        ];
        block.terminator = Some(Terminator::Return { value: None });

        let function = Function {
            name: "test".to_string(),
            return_type: IrType::Void,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![block],
            entry_block: entry,
            is_definition: true,
is_static: false,
is_weak: false,
        };

        let promotable = find_promotable_allocas(&function);
        assert!(promotable.is_empty());
    }

    #[test]
    fn test_alloca_used_in_gep_not_promotable() {
        let entry = BlockId(0);
        let mut block = BasicBlock::new(entry, "entry".to_string());
        block.instructions = vec![
            Instruction::Alloca {
                result: Value(0),
                ty: IrType::I32,
                count: None,
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer {
                    value: 0,
                    ty: IrType::I32,
                },
            },
            Instruction::GetElementPtr {
                result: Value(2),
                base_ty: IrType::I32,
                ptr: Value(0),
                indices: vec![Value(1)],
                in_bounds: true,
            },
        ];
        block.terminator = Some(Terminator::Return { value: None });

        let function = Function {
            name: "test".to_string(),
            return_type: IrType::Void,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![block],
            entry_block: entry,
            is_definition: true,
is_static: false,
is_weak: false,
        };

        let promotable = find_promotable_allocas(&function);
        assert!(promotable.is_empty());
    }

    #[test]
    fn test_alloca_address_stored_not_promotable() {
        let entry = BlockId(0);
        let mut block = BasicBlock::new(entry, "entry".to_string());
        block.instructions = vec![
            Instruction::Alloca {
                result: Value(0),
                ty: IrType::I32,
                count: None,
            },
            Instruction::Alloca {
                result: Value(1),
                ty: IrType::Pointer(Box::new(IrType::I32)),
                count: None,
            },
            // Storing the ADDRESS of alloca %0 into alloca %1.
            Instruction::Store { value: Value(0), ptr: Value(1), store_ty: None },
        ];
        block.terminator = Some(Terminator::Return { value: None });

        let function = Function {
            name: "test".to_string(),
            return_type: IrType::Void,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![block],
            entry_block: entry,
            is_definition: true,
is_static: false,
is_weak: false,
        };

        let promotable = find_promotable_allocas(&function);
        // %0 is NOT promotable (address escapes via store value).
        assert!(!promotable.iter().any(|(v, _)| *v == Value(0)));
    }

    #[test]
    fn test_dynamic_alloca_not_promotable() {
        let entry = BlockId(0);
        let mut block = BasicBlock::new(entry, "entry".to_string());
        block.instructions = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer {
                    value: 10,
                    ty: IrType::I32,
                },
            },
            Instruction::Alloca {
                result: Value(1),
                ty: IrType::I32,
                count: Some(Value(0)),
            },
        ];
        block.terminator = Some(Terminator::Return { value: None });

        let function = Function {
            name: "test".to_string(),
            return_type: IrType::Void,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![block],
            entry_block: entry,
            is_definition: true,
is_static: false,
is_weak: false,
        };

        let promotable = find_promotable_allocas(&function);
        assert!(promotable.is_empty());
    }

    #[test]
    fn test_mixed_promotable_and_non_promotable() {
        let entry = BlockId(0);
        let mut block = BasicBlock::new(entry, "entry".to_string());
        block.instructions = vec![
            Instruction::Alloca {
                result: Value(0),
                ty: IrType::I32,
                count: None,
            },
            Instruction::Alloca {
                result: Value(1),
                ty: IrType::I32,
                count: None,
            },
            Instruction::Alloca {
                result: Value(2),
                ty: IrType::I64,
                count: None,
            },
            Instruction::Const {
                result: Value(3),
                value: Constant::Integer {
                    value: 10,
                    ty: IrType::I32,
                },
            },
            Instruction::Store { value: Value(3), ptr: Value(0), store_ty: None },
            Instruction::Store { value: Value(3), ptr: Value(2), store_ty: None },
            Instruction::Call {
                result: None,
                callee: Callee::Direct("bar".to_string()),
                args: vec![Value(1)],
                return_ty: IrType::Void,
            },
        ];
        block.terminator = Some(Terminator::Return { value: None });

        let function = Function {
            name: "test".to_string(),
            return_type: IrType::Void,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![block],
            entry_block: entry,
            is_definition: true,
is_static: false,
is_weak: false,
        };

        let promotable = find_promotable_allocas(&function);
        assert_eq!(promotable.len(), 2);
        let vals: HashSet<Value> = promotable.iter().map(|(v, _)| *v).collect();
        assert!(vals.contains(&Value(0)));
        assert!(vals.contains(&Value(2)));
        assert!(!vals.contains(&Value(1)));
    }

    // -----------------------------------------------------------------------
    // Single-Block Promotion Test
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_block_promotion() {
        let entry = BlockId(0);
        let mut block = BasicBlock::new(entry, "entry".to_string());
        block.instructions = vec![
            Instruction::Alloca {
                result: Value(0),
                ty: IrType::I32,
                count: None,
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer {
                    value: 42,
                    ty: IrType::I32,
                },
            },
            Instruction::Store { value: Value(1), ptr: Value(0), store_ty: None },
            Instruction::Load {
                result: Value(2),
                ty: IrType::I32,
                ptr: Value(0),
            },
        ];
        block.terminator = Some(Terminator::Return {
            value: Some(Value(2)),
        });

        let mut function = Function {
            name: "test".to_string(),
            return_type: IrType::I32,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![block],
            entry_block: entry,
            is_definition: true,
is_static: false,
is_weak: false,
        };
        setup_edges(&mut function);

        let mut pass = Mem2RegPass::new();
        let changed = pass.run_on_function(&mut function);
        assert!(changed);

        // After promotion: no alloca, no load, no store.
        let has_alloca = function.blocks[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Alloca { .. }));
        assert!(!has_alloca, "Alloca should be removed");

        let has_load = function.blocks[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Load { .. }));
        assert!(!has_load, "Load should be removed");

        let has_store = function.blocks[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Store { .. }));
        assert!(!has_store, "Store should be removed");

        // The return should now use Value(1) (the constant 42) instead of Value(2).
        match &function.blocks[0].terminator {
            Some(Terminator::Return { value: Some(v) }) => {
                assert_eq!(*v, Value(1), "Return should use the constant directly");
            }
            _ => panic!("Expected return terminator"),
        }
    }

    // -----------------------------------------------------------------------
    // Diamond CFG Promotion Test
    // -----------------------------------------------------------------------

    #[test]
    fn test_diamond_cfg_promotion() {
        let mut entry_blk = BasicBlock::new(BlockId(0), "entry".to_string());
        let mut then_blk = BasicBlock::new(BlockId(1), "if.then".to_string());
        let mut else_blk = BasicBlock::new(BlockId(2), "if.else".to_string());
        let mut merge_blk = BasicBlock::new(BlockId(3), "merge".to_string());

        entry_blk.instructions = vec![
            Instruction::Alloca {
                result: Value(0),
                ty: IrType::I32,
                count: None,
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer {
                    value: 1,
                    ty: IrType::I1,
                },
            },
        ];
        entry_blk.terminator = Some(Terminator::CondBranch {
            condition: Value(1),
            true_block: BlockId(1),
            false_block: BlockId(2),
        });

        then_blk.instructions = vec![
            Instruction::Const {
                result: Value(2),
                value: Constant::Integer {
                    value: 10,
                    ty: IrType::I32,
                },
            },
            Instruction::Store { value: Value(2), ptr: Value(0), store_ty: None },
        ];
        then_blk.terminator = Some(Terminator::Branch {
            target: BlockId(3),
        });

        else_blk.instructions = vec![
            Instruction::Const {
                result: Value(3),
                value: Constant::Integer {
                    value: 20,
                    ty: IrType::I32,
                },
            },
            Instruction::Store { value: Value(3), ptr: Value(0), store_ty: None },
        ];
        else_blk.terminator = Some(Terminator::Branch {
            target: BlockId(3),
        });

        merge_blk.instructions = vec![Instruction::Load {
            result: Value(4),
            ty: IrType::I32,
            ptr: Value(0),
        }];
        merge_blk.terminator = Some(Terminator::Return {
            value: Some(Value(4)),
        });

        let mut function = Function {
            name: "diamond".to_string(),
            return_type: IrType::I32,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![entry_blk, then_blk, else_blk, merge_blk],
            entry_block: BlockId(0),
            is_definition: true,
is_static: false,
is_weak: false,
        };
        setup_edges(&mut function);

        let mut pass = Mem2RegPass::new();
        let changed = pass.run_on_function(&mut function);
        assert!(changed);

        // After promotion, the merge block should have a phi node.
        let merge = &function.blocks[3];
        assert!(
            !merge.phi_nodes.is_empty(),
            "Merge block should have a phi node"
        );

        let phi = &merge.phi_nodes[0];
        assert_eq!(phi.incoming.len(), 2, "Phi should have 2 incoming values");

        let incoming_blocks: HashSet<BlockId> =
            phi.incoming.iter().map(|(_, b)| *b).collect();
        assert!(incoming_blocks.contains(&BlockId(1)));
        assert!(incoming_blocks.contains(&BlockId(2)));
    }

    // -----------------------------------------------------------------------
    // No-Change Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_allocas_returns_false() {
        let entry = BlockId(0);
        let mut block = BasicBlock::new(entry, "entry".to_string());
        block.instructions = vec![Instruction::Const {
            result: Value(0),
            value: Constant::Integer {
                value: 0,
                ty: IrType::I32,
            },
        }];
        block.terminator = Some(Terminator::Return {
            value: Some(Value(0)),
        });

        let mut function = Function {
            name: "test".to_string(),
            return_type: IrType::I32,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![block],
            entry_block: entry,
            is_definition: true,
is_static: false,
is_weak: false,
        };
        setup_edges(&mut function);

        let mut pass = Mem2RegPass::new();
        assert!(!pass.run_on_function(&mut function));
    }

    #[test]
    fn test_non_promotable_only_returns_false() {
        let entry = BlockId(0);
        let mut block = BasicBlock::new(entry, "entry".to_string());
        block.instructions = vec![
            Instruction::Alloca {
                result: Value(0),
                ty: IrType::I32,
                count: None,
            },
            Instruction::Call {
                result: None,
                callee: Callee::Direct("sink".to_string()),
                args: vec![Value(0)],
                return_ty: IrType::Void,
            },
        ];
        block.terminator = Some(Terminator::Return { value: None });

        let mut function = Function {
            name: "test".to_string(),
            return_type: IrType::Void,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![block],
            entry_block: entry,
            is_definition: true,
is_static: false,
is_weak: false,
        };
        setup_edges(&mut function);

        let mut pass = Mem2RegPass::new();
        assert!(!pass.run_on_function(&mut function));
    }

    #[test]
    fn test_empty_function_returns_false() {
        let mut function = Function {
            name: "empty".to_string(),
            return_type: IrType::Void,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: Vec::new(),
            entry_block: BlockId(0),
            is_definition: false,
is_static: false,
is_weak: false,
        };

        let mut pass = Mem2RegPass::new();
        assert!(!pass.run_on_function(&mut function));
    }

    // -----------------------------------------------------------------------
    // Trivial Phi Simplification Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_trivial_phi_all_same_incoming() {
        let phi = PhiNode {
            result: Value(10),
            ty: IrType::I32,
            incoming: vec![
                (Value(5), BlockId(0)),
                (Value(5), BlockId(1)),
                (Value(5), BlockId(2)),
            ],
        };
        let result = trivial_phi_value(&phi);
        assert_eq!(result, Some(Value(5)));
    }

    #[test]
    fn test_trivial_phi_self_reference_with_one_value() {
        let phi = PhiNode {
            result: Value(10),
            ty: IrType::I32,
            incoming: vec![
                (Value(10), BlockId(0)),
                (Value(5), BlockId(1)),
                (Value(10), BlockId(2)),
            ],
        };
        let result = trivial_phi_value(&phi);
        assert_eq!(result, Some(Value(5)));
    }

    #[test]
    fn test_non_trivial_phi() {
        let phi = PhiNode {
            result: Value(10),
            ty: IrType::I32,
            incoming: vec![(Value(5), BlockId(0)), (Value(7), BlockId(1))],
        };
        let result = trivial_phi_value(&phi);
        assert!(result.is_none());
    }

    #[test]
    fn test_trivial_phi_empty_incoming() {
        let phi = PhiNode {
            result: Value(10),
            ty: IrType::I32,
            incoming: vec![],
        };
        let result = trivial_phi_value(&phi);
        assert_eq!(result, Some(Value::undef()));
    }

    #[test]
    fn test_trivial_phi_only_self_refs() {
        let phi = PhiNode {
            result: Value(10),
            ty: IrType::I32,
            incoming: vec![(Value(10), BlockId(0)), (Value(10), BlockId(1))],
        };
        let result = trivial_phi_value(&phi);
        assert_eq!(result, Some(Value::undef()));
    }

    // -----------------------------------------------------------------------
    // Multiple Allocas Test
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_allocas_partial_promotion() {
        let entry = BlockId(0);
        let mut block = BasicBlock::new(entry, "entry".to_string());
        block.instructions = vec![
            Instruction::Alloca {
                result: Value(0),
                ty: IrType::I32,
                count: None,
            },
            Instruction::Alloca {
                result: Value(1),
                ty: IrType::I32,
                count: None,
            },
            Instruction::Alloca {
                result: Value(2),
                ty: IrType::I64,
                count: None,
            },
            Instruction::Const {
                result: Value(3),
                value: Constant::Integer {
                    value: 1,
                    ty: IrType::I32,
                },
            },
            Instruction::Const {
                result: Value(4),
                value: Constant::Integer {
                    value: 2,
                    ty: IrType::I64,
                },
            },
            Instruction::Store { value: Value(3), ptr: Value(0), store_ty: None },
            Instruction::Store { value: Value(4), ptr: Value(2), store_ty: None },
            Instruction::Call {
                result: None,
                callee: Callee::Direct("sink".to_string()),
                args: vec![Value(1)],
                return_ty: IrType::Void,
            },
            Instruction::Load {
                result: Value(5),
                ty: IrType::I32,
                ptr: Value(0),
            },
            Instruction::Load {
                result: Value(6),
                ty: IrType::I64,
                ptr: Value(2),
            },
        ];
        block.terminator = Some(Terminator::Return {
            value: Some(Value(5)),
        });

        let mut function = Function {
            name: "multi".to_string(),
            return_type: IrType::I32,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![block],
            entry_block: entry,
            is_definition: true,
is_static: false,
is_weak: false,
        };
        setup_edges(&mut function);

        let mut pass = Mem2RegPass::new();
        let changed = pass.run_on_function(&mut function);
        assert!(changed);

        let has_alloca_0 = function.blocks[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Alloca { result, .. } if *result == Value(0)));
        assert!(!has_alloca_0, "Alloca %0 should be removed");

        let has_alloca_1 = function.blocks[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Alloca { result, .. } if *result == Value(1)));
        assert!(has_alloca_1, "Alloca %1 should remain (non-promotable)");

        let has_alloca_2 = function.blocks[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Alloca { result, .. } if *result == Value(2)));
        assert!(!has_alloca_2, "Alloca %2 should be removed");

        match &function.blocks[0].terminator {
            Some(Terminator::Return { value: Some(v) }) => {
                assert_eq!(*v, Value(3));
            }
            _ => panic!("Expected return terminator"),
        }
    }

    // -----------------------------------------------------------------------
    // Pass Name Test
    // -----------------------------------------------------------------------

    #[test]
    fn test_pass_name() {
        let pass = Mem2RegPass::new();
        assert_eq!(pass.name(), "mem2reg");
    }

    // -----------------------------------------------------------------------
    // Collect Alloca Info Test
    // -----------------------------------------------------------------------

    #[test]
    fn test_collect_alloca_info() {
        let mut entry_blk = BasicBlock::new(BlockId(0), "entry".to_string());
        let mut blk1 = BasicBlock::new(BlockId(1), "blk1".to_string());
        let mut blk2 = BasicBlock::new(BlockId(2), "blk2".to_string());

        entry_blk.instructions = vec![
            Instruction::Alloca {
                result: Value(0),
                ty: IrType::I32,
                count: None,
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer {
                    value: 0,
                    ty: IrType::I32,
                },
            },
            Instruction::Store { value: Value(1), ptr: Value(0), store_ty: None },
        ];
        entry_blk.terminator = Some(Terminator::Branch {
            target: BlockId(1),
        });

        blk1.instructions = vec![Instruction::Load {
            result: Value(2),
            ty: IrType::I32,
            ptr: Value(0),
        }];
        blk1.terminator = Some(Terminator::Branch {
            target: BlockId(2),
        });

        blk2.instructions = vec![
            Instruction::Const {
                result: Value(3),
                value: Constant::Integer {
                    value: 99,
                    ty: IrType::I32,
                },
            },
            Instruction::Store { value: Value(3), ptr: Value(0), store_ty: None },
        ];
        blk2.terminator = Some(Terminator::Return { value: None });

        let function = Function {
            name: "test".to_string(),
            return_type: IrType::Void,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![entry_blk, blk1, blk2],
            entry_block: BlockId(0),
            is_definition: true,
is_static: false,
is_weak: false,
        };

        let info = collect_alloca_info(Value(0), IrType::I32, &function);
        assert!(info.def_blocks.contains(&BlockId(0)));
        assert!(info.def_blocks.contains(&BlockId(2)));
        assert_eq!(info.def_blocks.len(), 2);

        assert!(info.use_blocks.contains(&BlockId(1)));
        assert_eq!(info.use_blocks.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Resolve Transitive Replacements Test
    // -----------------------------------------------------------------------

    #[test]
    fn test_resolve_transitive_replacements() {
        let mut map = HashMap::new();
        map.insert(Value(10), Value(20));
        map.insert(Value(20), Value(30));
        map.insert(Value(30), Value(40));

        resolve_transitive_replacements(&mut map);

        assert_eq!(map[&Value(10)], Value(40));
        assert_eq!(map[&Value(20)], Value(40));
        assert_eq!(map[&Value(30)], Value(40));
    }

    #[test]
    fn test_resolve_no_transitives() {
        let mut map = HashMap::new();
        map.insert(Value(10), Value(20));
        map.insert(Value(30), Value(40));

        resolve_transitive_replacements(&mut map);

        assert_eq!(map[&Value(10)], Value(20));
        assert_eq!(map[&Value(30)], Value(40));
    }

    // -----------------------------------------------------------------------
    // Find Max Value ID Test
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_max_value_id() {
        let mut block = BasicBlock::new(BlockId(0), "entry".to_string());
        block.instructions = vec![
            Instruction::Alloca {
                result: Value(5),
                ty: IrType::I32,
                count: None,
            },
            Instruction::Const {
                result: Value(10),
                value: Constant::Integer {
                    value: 42,
                    ty: IrType::I32,
                },
            },
            Instruction::Store { value: Value(10), ptr: Value(5), store_ty: None },
            Instruction::Load {
                result: Value(15),
                ty: IrType::I32,
                ptr: Value(5),
            },
        ];
        block.terminator = Some(Terminator::Return {
            value: Some(Value(15)),
        });

        let function = Function {
            name: "test".to_string(),
            return_type: IrType::I32,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![block],
            entry_block: BlockId(0),
            is_definition: true,
is_static: false,
is_weak: false,
        };

        let max_id = find_max_value_id(&function);
        assert_eq!(max_id, 15);
    }

    // -----------------------------------------------------------------------
    // Build CFG Test
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_cfg() {
        let mut b0 = BasicBlock::new(BlockId(0), "b0".to_string());
        let mut b1 = BasicBlock::new(BlockId(1), "b1".to_string());

        b0.successors = vec![BlockId(1)];
        b0.terminator = Some(Terminator::Branch {
            target: BlockId(1),
        });
        b1.predecessors = vec![BlockId(0)];
        b1.terminator = Some(Terminator::Return { value: None });

        let function = Function {
            name: "test".to_string(),
            return_type: IrType::Void,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![b0, b1],
            entry_block: BlockId(0),
            is_definition: true,
is_static: false,
is_weak: false,
        };

        let cfg = build_cfg(&function);
        assert_eq!(cfg.entry(), BlockId(0));
        assert_eq!(cfg.blocks().len(), 2);
    }
}
