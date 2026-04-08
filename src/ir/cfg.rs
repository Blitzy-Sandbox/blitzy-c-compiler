//! Control flow graph (CFG) module for the `bcc` compiler's intermediate representation.
//!
//! This module defines:
//! - [`BasicBlock`] — a sequence of IR instructions ending with a [`Terminator`]
//! - [`ControlFlowGraph`] — the set of basic blocks comprising a function
//! - [`DominanceTree`] — immediate dominator relationships between blocks
//! - [`PhiNode`] — SSA phi nodes merging values at control-flow join points
//! - [`Loop`] — natural loop structures detected via back-edge analysis
//!
//! # Algorithms
//!
//! - **Dominance tree**: Cooper, Harvey, Kennedy — "A Simple, Fast Dominance Algorithm" (2001)
//! - **Dominance frontiers**: Cytron et al. — "Efficiently Computing Static Single Assignment Form" (1991)
//! - **Loop detection**: Standard back-edge identification via the dominator tree
//!
//! # Performance
//!
//! Dominance computation is on the hot path for every function compiled. The
//! iterative algorithm converges in O(N) iterations for reducible CFGs (which
//! virtually all C programs produce), ensuring fast compilation even for large
//! functions such as those found in the SQLite amalgamation.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::ir::instructions::{BlockId, Instruction, Value};
use crate::ir::types::IrType;

// ---------------------------------------------------------------------------
// Terminator — block-ending control flow instructions
// ---------------------------------------------------------------------------

/// A terminator instruction that ends a basic block and describes control flow
/// to successor blocks.
///
/// Every well-formed basic block must end with exactly one `Terminator`.
/// Terminators are distinct from regular [`Instruction`] variants — they are
/// stored separately in [`BasicBlock::terminator`] rather than in the
/// instruction list.
#[derive(Debug, Clone, PartialEq)]
pub enum Terminator {
    /// Unconditional branch: `br label %target`
    ///
    /// Transfers control to exactly one successor block.
    Branch {
        /// The target basic block to branch to.
        target: BlockId,
    },

    /// Conditional branch: `br i1 %cond, label %then, label %else`
    ///
    /// Evaluates a boolean condition and branches to one of two successor blocks.
    CondBranch {
        /// The boolean (i1) condition value to evaluate.
        condition: Value,
        /// Block to branch to when `condition` is true (nonzero).
        true_block: BlockId,
        /// Block to branch to when `condition` is false (zero).
        false_block: BlockId,
    },

    /// Return from function: `ret type value` or `ret void`
    ///
    /// Terminates the current function, optionally returning a value to the caller.
    /// Has no successor blocks.
    Return {
        /// The return value, or `None` for void functions.
        value: Option<Value>,
    },

    /// Multi-way branch (switch): `switch i32 %val, label %default [cases...]`
    ///
    /// Evaluates an integer value and branches to the matching case label, or to
    /// the default label if no case matches.
    Switch {
        /// The integer value to switch on.
        value: Value,
        /// The default target block when no case matches.
        default: BlockId,
        /// (constant_value, target_block) pairs for each case arm.
        cases: Vec<(i64, BlockId)>,
    },

    /// Unreachable terminator (placed after `noreturn` calls, etc.).
    ///
    /// Signals that control flow cannot reach past this point. Has no successors.
    Unreachable,
}

impl Terminator {
    /// Returns the list of successor block IDs for this terminator.
    ///
    /// - `Branch` → one successor
    /// - `CondBranch` → two successors (may be the same block)
    /// - `Return` → no successors
    /// - `Switch` → default + one per case
    /// - `Unreachable` → no successors
    pub fn successors(&self) -> Vec<BlockId> {
        match self {
            Terminator::Branch { target } => vec![*target],
            Terminator::CondBranch {
                true_block,
                false_block,
                ..
            } => {
                if *true_block == *false_block {
                    vec![*true_block]
                } else {
                    vec![*true_block, *false_block]
                }
            }
            Terminator::Return { .. } => vec![],
            Terminator::Switch { default, cases, .. } => {
                let mut succs = vec![*default];
                for (_, target) in cases {
                    if !succs.contains(target) {
                        succs.push(*target);
                    }
                }
                succs
            }
            Terminator::Unreachable => vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// PhiNode — SSA phi function
// ---------------------------------------------------------------------------

/// An SSA phi node placed at the beginning of a basic block to merge values
/// from different predecessor blocks at a control-flow join point.
///
/// Each phi node defines a new SSA value (`result`) whose runtime value depends
/// on which predecessor block transferred control. The `incoming` list maps
/// each predecessor's `BlockId` to the `Value` that flows along that edge.
///
/// # Example (textual IR)
///
/// ```text
/// %5 = phi i32 [%2, bb1], [%4, bb2]
/// ```
///
/// When control arrives from `bb1`, `%5` takes the value of `%2`; when from
/// `bb2`, it takes the value of `%4`.
#[derive(Debug, Clone)]
pub struct PhiNode {
    /// The SSA value defined (produced) by this phi node.
    pub result: Value,

    /// The IR type of the result value.
    pub ty: IrType,

    /// Incoming edges: each entry is `(value, predecessor_block_id)`.
    ///
    /// There should be exactly one entry for each predecessor of the block
    /// that contains this phi node.
    pub incoming: Vec<(Value, BlockId)>,
}

impl PhiNode {
    /// Adds an incoming edge from the given predecessor block with the given value.
    ///
    /// # Parameters
    ///
    /// - `value` — The SSA value flowing along the edge from `block`.
    /// - `block` — The predecessor block this value comes from.
    pub fn add_incoming(&mut self, value: Value, block: BlockId) {
        self.incoming.push((value, block));
    }

    /// Looks up the incoming value for a specific predecessor block.
    ///
    /// Returns `Some(value)` if this phi node has an incoming edge from `block`,
    /// or `None` if no such edge exists.
    pub fn get_value_for_block(&self, block: BlockId) -> Option<Value> {
        self.incoming
            .iter()
            .find(|(_, b)| *b == block)
            .map(|(v, _)| *v)
    }
}

// ---------------------------------------------------------------------------
// BasicBlock — a linear sequence of instructions
// ---------------------------------------------------------------------------

/// A basic block in the control flow graph.
///
/// A basic block is a maximal sequence of instructions with:
/// - A single entry point (the first instruction)
/// - A single exit point (the terminator)
/// - No internal branches — once execution enters the block it proceeds
///   sequentially through all instructions to the terminator.
///
/// Phi nodes precede regular instructions and represent value merges at
/// control-flow join points in SSA form.
#[derive(Debug, Clone)]
pub struct BasicBlock {
    /// Unique identifier for this block within its containing function.
    pub id: BlockId,

    /// Human-readable label for debugging output (e.g., `"entry"`, `"if.then"`,
    /// `"while.cond"`).
    pub label: String,

    /// Phi nodes at the beginning of this block (SSA form).
    ///
    /// Phi nodes must precede all regular instructions. They are separate from
    /// the `instructions` list to simplify iteration and SSA algorithms.
    pub phi_nodes: Vec<PhiNode>,

    /// Regular instructions in this block, excluding phi nodes and the terminator.
    ///
    /// These execute sequentially in order.
    pub instructions: Vec<Instruction>,

    /// The terminator instruction that ends this block and defines control flow
    /// to successor blocks.
    ///
    /// `None` indicates an incomplete block (during construction). A well-formed
    /// CFG requires every block to have a terminator.
    pub terminator: Option<Terminator>,

    /// Block IDs of predecessors (blocks that branch to this block).
    pub predecessors: Vec<BlockId>,

    /// Block IDs of successors (blocks this block branches to).
    pub successors: Vec<BlockId>,
}

impl BasicBlock {
    /// Creates a new empty basic block with the given ID and label.
    pub fn new(id: BlockId, label: String) -> Self {
        BasicBlock {
            id,
            label,
            phi_nodes: Vec::new(),
            instructions: Vec::new(),
            terminator: None,
            predecessors: Vec::new(),
            successors: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// ControlFlowGraph — the set of basic blocks for a function
// ---------------------------------------------------------------------------

/// The control flow graph (CFG) for a single function.
///
/// The CFG owns all basic blocks and provides methods for querying and
/// manipulating the graph structure, including edge management, reachability
/// analysis, and well-formedness checks.
///
/// Blocks are stored in a `Vec` indexed by their `BlockId`. The entry block
/// is designated at construction time and is typically `BlockId(0)`.
#[derive(Debug, Clone)]
pub struct ControlFlowGraph {
    /// All basic blocks in the function, indexed by `BlockId(n)` → `blocks[n]`.
    blocks: Vec<BasicBlock>,

    /// The entry block ID (the first block executed when the function is called).
    entry: BlockId,
}

impl ControlFlowGraph {
    /// Creates a new empty CFG with the specified entry block ID.
    ///
    /// The entry block itself must still be added via [`add_block`].
    pub fn new(entry: BlockId) -> Self {
        ControlFlowGraph {
            blocks: Vec::new(),
            entry,
        }
    }

    /// Adds a basic block to the CFG and returns its `BlockId`.
    ///
    /// The block's `id` field should be consistent with its position in the
    /// internal vector, but this method uses the block's own `id` as-is.
    /// The block is appended at the end of the internal storage.
    pub fn add_block(&mut self, block: BasicBlock) -> BlockId {
        let id = block.id;
        // Ensure the blocks vector is large enough to index by block.id
        let idx = id.0 as usize;
        if idx >= self.blocks.len() {
            // Extend with placeholder blocks up to the required index
            while self.blocks.len() <= idx {
                let placeholder_id = BlockId(self.blocks.len() as u32);
                self.blocks.push(BasicBlock::new(
                    placeholder_id,
                    format!("__placeholder_{}", placeholder_id.0),
                ));
            }
        }
        self.blocks[idx] = block;
        id
    }

    /// Returns an immutable reference to the block with the given ID.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not correspond to a valid block in the CFG.
    pub fn block(&self, id: BlockId) -> &BasicBlock {
        &self.blocks[id.0 as usize]
    }

    /// Returns a mutable reference to the block with the given ID.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not correspond to a valid block in the CFG.
    pub fn block_mut(&mut self, id: BlockId) -> &mut BasicBlock {
        &mut self.blocks[id.0 as usize]
    }

    /// Returns the entry block ID.
    pub fn entry(&self) -> BlockId {
        self.entry
    }

    /// Returns a slice of all basic blocks in the CFG.
    pub fn blocks(&self) -> &[BasicBlock] {
        &self.blocks
    }

    /// Returns a mutable slice of all basic blocks in the CFG.
    pub fn blocks_mut(&mut self) -> &mut [BasicBlock] {
        &mut self.blocks
    }

    /// Returns the number of basic blocks in the CFG.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Returns the successor block IDs for the block with the given ID.
    pub fn successors(&self, id: BlockId) -> &[BlockId] {
        &self.blocks[id.0 as usize].successors
    }

    /// Returns the predecessor block IDs for the block with the given ID.
    pub fn predecessors(&self, id: BlockId) -> &[BlockId] {
        &self.blocks[id.0 as usize].predecessors
    }

    /// Adds a directed edge from block `from` to block `to`.
    ///
    /// Updates both the successor list of `from` and the predecessor list of `to`.
    /// Avoids adding duplicate edges.
    pub fn add_edge(&mut self, from: BlockId, to: BlockId) {
        let from_idx = from.0 as usize;
        let to_idx = to.0 as usize;

        // Add `to` as a successor of `from` if not already present
        if !self.blocks[from_idx].successors.contains(&to) {
            self.blocks[from_idx].successors.push(to);
        }

        // Add `from` as a predecessor of `to` if not already present
        if !self.blocks[to_idx].predecessors.contains(&from) {
            self.blocks[to_idx].predecessors.push(from);
        }
    }

    /// Removes the directed edge from block `from` to block `to`.
    ///
    /// Updates both the successor list of `from` and the predecessor list of `to`.
    /// Does nothing if the edge does not exist.
    pub fn remove_edge(&mut self, from: BlockId, to: BlockId) {
        let from_idx = from.0 as usize;
        let to_idx = to.0 as usize;

        self.blocks[from_idx].successors.retain(|s| *s != to);
        self.blocks[to_idx].predecessors.retain(|p| *p != from);
    }

    /// Scans all terminators and rebuilds predecessor/successor lists from scratch.
    ///
    /// This is useful after modifying terminators directly without going through
    /// the edge management API.
    pub fn compute_edges(&mut self) {
        // Clear all existing edges
        for block in &mut self.blocks {
            block.predecessors.clear();
            block.successors.clear();
        }

        // Collect edges from terminators
        let mut edges: Vec<(BlockId, BlockId)> = Vec::new();
        for block in &self.blocks {
            if let Some(ref term) = block.terminator {
                for succ in term.successors() {
                    edges.push((block.id, succ));
                }
            }
        }

        // Apply edges
        for (from, to) in edges {
            self.add_edge(from, to);
        }
    }

    /// Returns the set of all block IDs reachable from the entry block via BFS.
    ///
    /// This is used for dead block elimination — any block not in the returned
    /// set is unreachable and can be safely removed.
    pub fn reachable_blocks(&self) -> HashSet<BlockId> {
        let mut visited = HashSet::new();
        let mut worklist = VecDeque::new();

        worklist.push_back(self.entry);
        visited.insert(self.entry);

        while let Some(block_id) = worklist.pop_front() {
            for &succ in &self.blocks[block_id.0 as usize].successors {
                if visited.insert(succ) {
                    worklist.push_back(succ);
                }
            }
        }

        visited
    }

    /// Returns a list of block IDs that are NOT reachable from the entry block.
    ///
    /// These blocks can be removed during dead code elimination.
    pub fn unreachable_blocks(&self) -> Vec<BlockId> {
        let reachable = self.reachable_blocks();
        self.blocks
            .iter()
            .map(|b| b.id)
            .filter(|id| !reachable.contains(id))
            .collect()
    }

    /// Checks whether the CFG is well-formed.
    ///
    /// A well-formed CFG satisfies:
    /// 1. Every block has a terminator.
    /// 2. All successor/predecessor block IDs reference valid blocks.
    /// 3. Successor and predecessor lists are mutually consistent.
    /// 4. The entry block exists.
    pub fn is_well_formed(&self) -> bool {
        // Entry block must exist
        if (self.entry.0 as usize) >= self.blocks.len() {
            return false;
        }

        let num = self.blocks.len();

        for block in &self.blocks {
            // Every block must have a terminator
            if block.terminator.is_none() {
                return false;
            }

            // All successor IDs must be valid
            for succ in &block.successors {
                if (succ.0 as usize) >= num {
                    return false;
                }
            }

            // All predecessor IDs must be valid
            for pred in &block.predecessors {
                if (pred.0 as usize) >= num {
                    return false;
                }
            }

            // Consistency: if B is a successor of A, then A must be a predecessor of B
            for &succ in &block.successors {
                if !self.blocks[succ.0 as usize]
                    .predecessors
                    .contains(&block.id)
                {
                    return false;
                }
            }

            // Consistency: if B is a predecessor of A, then A must be a successor of B
            for &pred in &block.predecessors {
                if !self.blocks[pred.0 as usize].successors.contains(&block.id) {
                    return false;
                }
            }
        }

        true
    }
}

// ---------------------------------------------------------------------------
// Reverse postorder traversal
// ---------------------------------------------------------------------------

/// Computes a reverse postorder (RPO) traversal of the CFG starting from
/// the entry block.
///
/// Reverse postorder visits a block before any of its successors in the
/// dominator tree (for reducible CFGs), making it the ideal traversal order
/// for dataflow algorithms such as dominance computation and SSA construction.
///
/// # Algorithm
///
/// 1. Perform a depth-first search from the entry block.
/// 2. Append each block to the result list *after* all its successors have
///    been visited (postorder).
/// 3. Reverse the list to obtain reverse postorder.
///
/// Blocks unreachable from the entry are not included in the result.
pub fn reverse_postorder(cfg: &ControlFlowGraph) -> Vec<BlockId> {
    let mut visited = HashSet::new();
    let mut postorder = Vec::new();

    fn dfs(
        block_id: BlockId,
        cfg: &ControlFlowGraph,
        visited: &mut HashSet<BlockId>,
        postorder: &mut Vec<BlockId>,
    ) {
        if !visited.insert(block_id) {
            return;
        }
        for &succ in &cfg.blocks[block_id.0 as usize].successors {
            dfs(succ, cfg, visited, postorder);
        }
        postorder.push(block_id);
    }

    dfs(cfg.entry(), cfg, &mut visited, &mut postorder);
    postorder.reverse();
    postorder
}

// ---------------------------------------------------------------------------
// DominanceTree — immediate dominator relationships
// ---------------------------------------------------------------------------

/// The dominator tree for a control flow graph.
///
/// A block `A` *dominates* block `B` if every path from the entry block to `B`
/// must pass through `A`. The *immediate dominator* `idom(B)` is the closest
/// strict dominator of `B`.
///
/// The dominator tree is a tree rooted at the entry block where the parent
/// of each node is its immediate dominator.
///
/// # Construction
///
/// Use [`compute_dominance_tree`] to build a `DominanceTree` from a
/// [`ControlFlowGraph`].
#[derive(Debug, Clone)]
pub struct DominanceTree {
    /// Maps each block to its immediate dominator.
    ///
    /// The entry block maps to itself (it has no strict dominator).
    idom: HashMap<BlockId, BlockId>,

    /// Maps each block to the list of blocks it immediately dominates
    /// (its children in the dominator tree).
    children: HashMap<BlockId, Vec<BlockId>>,
}

impl DominanceTree {
    /// Returns the immediate dominator of `block`, or `None` if the block
    /// is not in the dominator tree.
    ///
    /// For the entry block, returns `Some(entry)` (self-dominating).
    pub fn idom(&self, block: BlockId) -> Option<BlockId> {
        self.idom.get(&block).copied()
    }

    /// Returns `true` if block `a` dominates block `b`.
    ///
    /// A block dominates itself. The entry block dominates all reachable blocks.
    ///
    /// # Algorithm
    ///
    /// Walks up the dominator tree from `b` toward the entry until either `a`
    /// is found (returns `true`) or the entry is reached without finding `a`
    /// (returns `false`).
    pub fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        if a == b {
            return true;
        }
        let mut current = b;
        loop {
            match self.idom.get(&current) {
                Some(&idom) => {
                    if idom == a {
                        return true;
                    }
                    if idom == current {
                        // Reached the entry block (idom of entry is entry itself)
                        return false;
                    }
                    current = idom;
                }
                None => return false,
            }
        }
    }

    /// Returns the list of blocks immediately dominated by `block`
    /// (its children in the dominator tree).
    ///
    /// Returns an empty slice if `block` has no children.
    pub fn children(&self, block: BlockId) -> &[BlockId] {
        match self.children.get(&block) {
            Some(kids) => kids.as_slice(),
            None => &[],
        }
    }

    /// Returns a preorder traversal of the dominator tree starting from `entry`.
    ///
    /// Preorder visits a node before any of its children, which is useful for
    /// SSA renaming passes that process definitions before uses.
    pub fn preorder(&self, entry: BlockId) -> Vec<BlockId> {
        let mut result = Vec::new();
        let mut stack = vec![entry];

        while let Some(block) = stack.pop() {
            result.push(block);
            // Push children in reverse order so they are visited left-to-right
            if let Some(kids) = self.children.get(&block) {
                for &child in kids.iter().rev() {
                    stack.push(child);
                }
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// compute_dominance_tree — Cooper, Harvey, Kennedy iterative algorithm
// ---------------------------------------------------------------------------

/// Computes the dominator tree for the given CFG using the iterative algorithm
/// by Cooper, Harvey, and Kennedy (2001).
///
/// This algorithm:
/// 1. Initializes `idom[entry] = entry`; all others are undefined.
/// 2. Computes reverse postorder for iteration ordering.
/// 3. Iterates until convergence, updating `idom` for each block based on
///    the intersection of its predecessors' dominators.
///
/// The algorithm converges in O(N) iterations for reducible CFGs, making it
/// efficient for the vast majority of C programs.
///
/// # Parameters
///
/// - `cfg` — The control flow graph to analyze.
///
/// # Returns
///
/// A [`DominanceTree`] containing immediate dominator information and the
/// children map.
pub fn compute_dominance_tree(cfg: &ControlFlowGraph) -> DominanceTree {
    let rpo = reverse_postorder(cfg);

    if rpo.is_empty() {
        return DominanceTree {
            idom: HashMap::new(),
            children: HashMap::new(),
        };
    }

    // Map each block to its reverse-postorder index for the intersect function.
    let mut rpo_number: HashMap<BlockId, usize> = HashMap::new();
    for (idx, &block) in rpo.iter().enumerate() {
        rpo_number.insert(block, idx);
    }

    let entry = cfg.entry();
    let mut idom: HashMap<BlockId, BlockId> = HashMap::new();
    idom.insert(entry, entry);

    let mut changed = true;
    while changed {
        changed = false;
        for &b in &rpo {
            if b == entry {
                continue;
            }

            // Find the first predecessor that already has an idom assignment
            let preds = &cfg.block(b).predecessors;
            let mut new_idom: Option<BlockId> = None;

            for &p in preds {
                if idom.contains_key(&p) {
                    match new_idom {
                        None => {
                            new_idom = Some(p);
                        }
                        Some(current) => {
                            new_idom = Some(intersect(current, p, &idom, &rpo_number));
                        }
                    }
                }
            }

            if let Some(ni) = new_idom {
                let old = idom.get(&b).copied();
                if old != Some(ni) {
                    idom.insert(b, ni);
                    changed = true;
                }
            }
        }
    }

    // Build the children map from the idom map
    let mut children: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    for (&block, &dom) in idom.iter() {
        if block != dom {
            children.entry(dom).or_insert_with(Vec::new).push(block);
        }
    }

    // Sort children for deterministic traversal order
    for kids in children.values_mut() {
        kids.sort();
    }

    DominanceTree { idom, children }
}

/// The intersect helper for the Cooper-Harvey-Kennedy dominance algorithm.
///
/// Given two blocks `b1` and `b2` that both have known immediate dominators,
/// returns the nearest common dominator by walking up the dominator tree from
/// both sides until the paths converge.
///
/// The walk is guided by reverse-postorder indices: the block with the higher
/// index (further from entry) is moved to its immediate dominator first.
fn intersect(
    b1: BlockId,
    b2: BlockId,
    idom: &HashMap<BlockId, BlockId>,
    rpo_number: &HashMap<BlockId, usize>,
) -> BlockId {
    let mut finger1 = b1;
    let mut finger2 = b2;

    while finger1 != finger2 {
        while rpo_number.get(&finger1).copied().unwrap_or(0)
            > rpo_number.get(&finger2).copied().unwrap_or(0)
        {
            finger1 = idom[&finger1];
        }
        while rpo_number.get(&finger2).copied().unwrap_or(0)
            > rpo_number.get(&finger1).copied().unwrap_or(0)
        {
            finger2 = idom[&finger2];
        }
    }

    finger1
}

// ---------------------------------------------------------------------------
// compute_dominance_frontiers — Cytron et al. algorithm
// ---------------------------------------------------------------------------

/// Computes the dominance frontier for every block in the CFG.
///
/// The dominance frontier DF(B) of a block B is the set of blocks Y such that
/// B dominates a predecessor of Y but does not strictly dominate Y. Dominance
/// frontiers identify the exact locations where phi nodes must be inserted
/// during SSA construction.
///
/// # Algorithm
///
/// For each join point Y (a block with ≥2 predecessors):
///   For each predecessor P of Y:
///     Walk up the dominator tree from P, adding Y to the dominance frontier
///     of each block visited, until reaching idom(Y).
///
/// This is the standard algorithm from Cytron, Ferrante, Rosen, Wegman, and
/// Zadeck (1991).
///
/// # Parameters
///
/// - `cfg` — The control flow graph.
/// - `dom_tree` — The precomputed dominator tree for `cfg`.
///
/// # Returns
///
/// A map from each `BlockId` to its dominance frontier (a set of `BlockId`s).
pub fn compute_dominance_frontiers(
    cfg: &ControlFlowGraph,
    dom_tree: &DominanceTree,
) -> HashMap<BlockId, HashSet<BlockId>> {
    let mut frontiers: HashMap<BlockId, HashSet<BlockId>> = HashMap::new();

    // Initialize empty frontier sets for all blocks
    for block in cfg.blocks() {
        frontiers.insert(block.id, HashSet::new());
    }

    // For each block Y with ≥2 predecessors (a join point),
    // walk up from each predecessor to idom(Y), adding Y to each DF set.
    for block in cfg.blocks() {
        let y = block.id;
        if block.predecessors.len() < 2 {
            continue;
        }

        let idom_y = match dom_tree.idom(y) {
            Some(d) => d,
            None => continue,
        };

        for &pred in &block.predecessors {
            let mut runner = pred;
            while runner != idom_y {
                frontiers
                    .entry(runner)
                    .or_insert_with(HashSet::new)
                    .insert(y);
                match dom_tree.idom(runner) {
                    Some(d) if d != runner => {
                        runner = d;
                    }
                    _ => break,
                }
            }
        }
    }

    frontiers
}

// ---------------------------------------------------------------------------
// Loop — natural loop structure
// ---------------------------------------------------------------------------

/// A natural loop detected in the control flow graph.
///
/// A natural loop is defined by a back edge (A → B) where B dominates A.
/// The loop header is B, and the loop body consists of all blocks that can
/// reach A without passing through B, plus B itself.
#[derive(Debug, Clone)]
pub struct Loop {
    /// The loop header block (the entry point / back-edge target).
    pub header: BlockId,

    /// All blocks in the loop body, including the header.
    pub blocks: HashSet<BlockId>,

    /// Back edges that define this loop: `(source, header)` pairs.
    pub back_edges: Vec<(BlockId, BlockId)>,

    /// Nested inner loops contained within this loop.
    pub inner_loops: Vec<Loop>,

    /// Nesting depth of this loop (1 for outermost, 2 for first nesting, etc.).
    pub depth: usize,
}

// ---------------------------------------------------------------------------
// detect_loops — natural loop detection via back-edge identification
// ---------------------------------------------------------------------------

/// Detects all natural loops in the CFG using back-edge analysis.
///
/// # Algorithm
///
/// 1. Find all back edges: an edge (A → B) is a back edge if B dominates A.
/// 2. For each unique header (back-edge target), compute the natural loop body
///    by working backward from each back-edge source through predecessors.
/// 3. Build a loop nesting tree based on containment relationships.
/// 4. Assign nesting depths.
///
/// # Parameters
///
/// - `cfg` — The control flow graph.
/// - `dom_tree` — The precomputed dominator tree.
///
/// # Returns
///
/// A list of detected loops. Outermost loops are at the top level; inner loops
/// are nested in the `inner_loops` field.
pub fn detect_loops(cfg: &ControlFlowGraph, dom_tree: &DominanceTree) -> Vec<Loop> {
    // Step 1: Find all back edges
    let mut back_edges: Vec<(BlockId, BlockId)> = Vec::new();
    for block in cfg.blocks() {
        for &succ in &block.successors {
            if dom_tree.dominates(succ, block.id) {
                back_edges.push((block.id, succ));
            }
        }
    }

    if back_edges.is_empty() {
        return Vec::new();
    }

    // Step 2: Group back edges by header and compute natural loop bodies
    let mut header_map: HashMap<BlockId, Vec<(BlockId, BlockId)>> = HashMap::new();
    for &(src, header) in &back_edges {
        header_map
            .entry(header)
            .or_insert_with(Vec::new)
            .push((src, header));
    }

    let mut loops: Vec<Loop> = Vec::new();

    for (&header, edges) in &header_map {
        let mut body = HashSet::new();
        body.insert(header);

        // For each back edge source, walk backward through predecessors to
        // find all blocks in the natural loop body.
        let mut worklist = VecDeque::new();
        for &(src, _) in edges {
            if src != header && body.insert(src) {
                worklist.push_back(src);
            }
        }

        while let Some(block_id) = worklist.pop_front() {
            for &pred in &cfg.block(block_id).predecessors {
                if body.insert(pred) {
                    worklist.push_back(pred);
                }
            }
        }

        loops.push(Loop {
            header,
            blocks: body,
            back_edges: edges.clone(),
            inner_loops: Vec::new(),
            depth: 1,
        });
    }

    // Step 3: Build nesting relationships
    // Sort loops by size (smallest first) to process inner loops before outer
    loops.sort_by_key(|l| l.blocks.len());

    // Build nesting: a loop L1 is nested in L2 if L1.header ∈ L2.blocks
    // and L1 != L2.
    let loop_count = loops.len();
    let mut parent_of: Vec<Option<usize>> = vec![None; loop_count];

    for i in 0..loop_count {
        for j in (i + 1)..loop_count {
            // loops[j] is larger; check if loops[i] is nested in loops[j]
            if loops[i].header != loops[j].header && loops[j].blocks.contains(&loops[i].header) {
                // loops[i] is an inner loop of the smallest containing loop
                match parent_of[i] {
                    None => {
                        parent_of[i] = Some(j);
                    }
                    Some(current_parent) => {
                        // Pick the smaller (tighter) parent
                        if loops[j].blocks.len() < loops[current_parent].blocks.len() {
                            parent_of[i] = Some(j);
                        }
                    }
                }
            }
        }
    }

    // Assign depths and collect inner loops
    // First, compute depths via parent chain
    let mut depths: Vec<usize> = vec![1; loop_count];
    for i in 0..loop_count {
        let mut depth = 1;
        let mut p = parent_of[i];
        while let Some(pi) = p {
            depth += 1;
            p = parent_of[pi];
        }
        depths[i] = depth;
    }

    for i in 0..loop_count {
        loops[i].depth = depths[i];
    }

    // Build the nesting tree: collect loops without parents as top-level,
    // and attach children to their parent loops.
    // We need to carefully reconstruct the hierarchy since we can't move
    // elements out of the vec while iterating.
    let mut top_level_indices: Vec<usize> = Vec::new();
    let mut children_map: HashMap<usize, Vec<usize>> = HashMap::new();

    for i in 0..loop_count {
        match parent_of[i] {
            None => top_level_indices.push(i),
            Some(p) => {
                children_map.entry(p).or_insert_with(Vec::new).push(i);
            }
        }
    }

    // Recursively build the tree structure
    fn build_loop_tree(
        idx: usize,
        loops: &[Loop],
        children_map: &HashMap<usize, Vec<usize>>,
    ) -> Loop {
        let mut result = loops[idx].clone();
        result.inner_loops.clear();

        if let Some(child_indices) = children_map.get(&idx) {
            for &ci in child_indices {
                result
                    .inner_loops
                    .push(build_loop_tree(ci, loops, children_map));
            }
        }
        result
    }

    let mut result = Vec::new();
    for &ti in &top_level_indices {
        result.push(build_loop_tree(ti, &loops, &children_map));
    }

    result
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::instructions::{BlockId, Value};
    use crate::ir::types::IrType;

    /// Helper: create a basic block with given ID and label, no instructions.
    fn make_block(id: u32, label: &str) -> BasicBlock {
        BasicBlock::new(BlockId(id), label.to_string())
    }

    /// Helper: build a simple linear chain CFG: A → B → C → D (return).
    fn build_linear_cfg() -> ControlFlowGraph {
        let mut cfg = ControlFlowGraph::new(BlockId(0));

        let mut a = make_block(0, "A");
        a.terminator = Some(Terminator::Branch { target: BlockId(1) });
        cfg.add_block(a);

        let mut b = make_block(1, "B");
        b.terminator = Some(Terminator::Branch { target: BlockId(2) });
        cfg.add_block(b);

        let mut c = make_block(2, "C");
        c.terminator = Some(Terminator::Branch { target: BlockId(3) });
        cfg.add_block(c);

        let mut d = make_block(3, "D");
        d.terminator = Some(Terminator::Return { value: None });
        cfg.add_block(d);

        cfg.compute_edges();
        cfg
    }

    /// Helper: build a diamond CFG:
    ///
    /// ```text
    ///       entry(0)
    ///      /       \
    ///   left(1)  right(2)
    ///      \       /
    ///      merge(3)
    /// ```
    fn build_diamond_cfg() -> ControlFlowGraph {
        let mut cfg = ControlFlowGraph::new(BlockId(0));

        let mut entry = make_block(0, "entry");
        entry.terminator = Some(Terminator::CondBranch {
            condition: Value(0),
            true_block: BlockId(1),
            false_block: BlockId(2),
        });
        cfg.add_block(entry);

        let mut left = make_block(1, "left");
        left.terminator = Some(Terminator::Branch { target: BlockId(3) });
        cfg.add_block(left);

        let mut right = make_block(2, "right");
        right.terminator = Some(Terminator::Branch { target: BlockId(3) });
        cfg.add_block(right);

        let mut merge = make_block(3, "merge");
        merge.terminator = Some(Terminator::Return { value: None });
        cfg.add_block(merge);

        cfg.compute_edges();
        cfg
    }

    /// Helper: build a simple loop CFG:
    ///
    /// ```text
    ///   entry(0) → header(1) → body(2) → header(1)  [back edge]
    ///                  |
    ///                exit(3)
    /// ```
    fn build_loop_cfg() -> ControlFlowGraph {
        let mut cfg = ControlFlowGraph::new(BlockId(0));

        let mut entry = make_block(0, "entry");
        entry.terminator = Some(Terminator::Branch { target: BlockId(1) });
        cfg.add_block(entry);

        let mut header = make_block(1, "header");
        header.terminator = Some(Terminator::CondBranch {
            condition: Value(0),
            true_block: BlockId(2),
            false_block: BlockId(3),
        });
        cfg.add_block(header);

        let mut body = make_block(2, "body");
        body.terminator = Some(Terminator::Branch { target: BlockId(1) });
        cfg.add_block(body);

        let mut exit = make_block(3, "exit");
        exit.terminator = Some(Terminator::Return { value: None });
        cfg.add_block(exit);

        cfg.compute_edges();
        cfg
    }

    // =======================================================================
    // Basic CFG construction tests
    // =======================================================================

    #[test]
    fn test_single_block_cfg() {
        let mut cfg = ControlFlowGraph::new(BlockId(0));
        let mut blk = make_block(0, "entry");
        blk.terminator = Some(Terminator::Return { value: None });
        cfg.add_block(blk);
        cfg.compute_edges();

        assert_eq!(cfg.num_blocks(), 1);
        assert_eq!(cfg.entry(), BlockId(0));
        assert!(cfg.successors(BlockId(0)).is_empty());
        assert!(cfg.predecessors(BlockId(0)).is_empty());
        assert!(cfg.is_well_formed());
    }

    #[test]
    fn test_linear_chain_edges() {
        let cfg = build_linear_cfg();
        assert_eq!(cfg.num_blocks(), 4);

        // A → B
        assert_eq!(cfg.successors(BlockId(0)), &[BlockId(1)]);
        assert!(cfg.predecessors(BlockId(0)).is_empty());

        // B → C, predecessor A
        assert_eq!(cfg.successors(BlockId(1)), &[BlockId(2)]);
        assert_eq!(cfg.predecessors(BlockId(1)), &[BlockId(0)]);

        // C → D, predecessor B
        assert_eq!(cfg.successors(BlockId(2)), &[BlockId(3)]);
        assert_eq!(cfg.predecessors(BlockId(2)), &[BlockId(1)]);

        // D returns, predecessor C
        assert!(cfg.successors(BlockId(3)).is_empty());
        assert_eq!(cfg.predecessors(BlockId(3)), &[BlockId(2)]);

        assert!(cfg.is_well_formed());
    }

    #[test]
    fn test_diamond_edges() {
        let cfg = build_diamond_cfg();

        // Entry branches to left and right
        assert_eq!(cfg.successors(BlockId(0)).len(), 2);
        assert!(cfg.successors(BlockId(0)).contains(&BlockId(1)));
        assert!(cfg.successors(BlockId(0)).contains(&BlockId(2)));

        // Left and right both go to merge
        assert_eq!(cfg.successors(BlockId(1)), &[BlockId(3)]);
        assert_eq!(cfg.successors(BlockId(2)), &[BlockId(3)]);

        // Merge has two predecessors
        let merge_preds = cfg.predecessors(BlockId(3));
        assert_eq!(merge_preds.len(), 2);
        assert!(merge_preds.contains(&BlockId(1)));
        assert!(merge_preds.contains(&BlockId(2)));

        assert!(cfg.is_well_formed());
    }

    // =======================================================================
    // Edge management tests
    // =======================================================================

    #[test]
    fn test_add_edge_no_duplicates() {
        let mut cfg = ControlFlowGraph::new(BlockId(0));
        let mut a = make_block(0, "A");
        a.terminator = Some(Terminator::Return { value: None });
        cfg.add_block(a);

        let mut b = make_block(1, "B");
        b.terminator = Some(Terminator::Return { value: None });
        cfg.add_block(b);

        cfg.add_edge(BlockId(0), BlockId(1));
        cfg.add_edge(BlockId(0), BlockId(1)); // duplicate

        assert_eq!(cfg.successors(BlockId(0)), &[BlockId(1)]);
        assert_eq!(cfg.predecessors(BlockId(1)), &[BlockId(0)]);
    }

    #[test]
    fn test_remove_edge() {
        let mut cfg = ControlFlowGraph::new(BlockId(0));
        let mut a = make_block(0, "A");
        a.terminator = Some(Terminator::Return { value: None });
        cfg.add_block(a);

        let mut b = make_block(1, "B");
        b.terminator = Some(Terminator::Return { value: None });
        cfg.add_block(b);

        cfg.add_edge(BlockId(0), BlockId(1));
        assert_eq!(cfg.successors(BlockId(0)).len(), 1);
        assert_eq!(cfg.predecessors(BlockId(1)).len(), 1);

        cfg.remove_edge(BlockId(0), BlockId(1));
        assert!(cfg.successors(BlockId(0)).is_empty());
        assert!(cfg.predecessors(BlockId(1)).is_empty());
    }

    // =======================================================================
    // Terminator tests
    // =======================================================================

    #[test]
    fn test_branch_successors() {
        let term = Terminator::Branch { target: BlockId(5) };
        assert_eq!(term.successors(), vec![BlockId(5)]);
    }

    #[test]
    fn test_condbranch_successors() {
        let term = Terminator::CondBranch {
            condition: Value(0),
            true_block: BlockId(1),
            false_block: BlockId(2),
        };
        assert_eq!(term.successors(), vec![BlockId(1), BlockId(2)]);
    }

    #[test]
    fn test_condbranch_same_target() {
        let term = Terminator::CondBranch {
            condition: Value(0),
            true_block: BlockId(1),
            false_block: BlockId(1),
        };
        assert_eq!(term.successors(), vec![BlockId(1)]);
    }

    #[test]
    fn test_return_no_successors() {
        let term = Terminator::Return {
            value: Some(Value(0)),
        };
        assert!(term.successors().is_empty());
    }

    #[test]
    fn test_switch_successors() {
        let term = Terminator::Switch {
            value: Value(0),
            default: BlockId(10),
            cases: vec![(1, BlockId(11)), (2, BlockId(12)), (3, BlockId(10))],
        };
        let succs = term.successors();
        assert_eq!(succs.len(), 3); // default(10), 11, 12 — 10 is deduplicated
        assert!(succs.contains(&BlockId(10)));
        assert!(succs.contains(&BlockId(11)));
        assert!(succs.contains(&BlockId(12)));
    }

    #[test]
    fn test_unreachable_no_successors() {
        let term = Terminator::Unreachable;
        assert!(term.successors().is_empty());
    }

    // =======================================================================
    // PhiNode tests
    // =======================================================================

    #[test]
    fn test_phi_node_add_incoming() {
        let mut phi = PhiNode {
            result: Value(5),
            ty: IrType::I32,
            incoming: Vec::new(),
        };

        phi.add_incoming(Value(1), BlockId(0));
        phi.add_incoming(Value(2), BlockId(1));

        assert_eq!(phi.incoming.len(), 2);
        assert_eq!(phi.get_value_for_block(BlockId(0)), Some(Value(1)));
        assert_eq!(phi.get_value_for_block(BlockId(1)), Some(Value(2)));
        assert_eq!(phi.get_value_for_block(BlockId(99)), None);
    }

    // =======================================================================
    // Reverse postorder tests
    // =======================================================================

    #[test]
    fn test_rpo_linear_chain() {
        let cfg = build_linear_cfg();
        let rpo = reverse_postorder(&cfg);
        assert_eq!(rpo, vec![BlockId(0), BlockId(1), BlockId(2), BlockId(3)]);
    }

    #[test]
    fn test_rpo_diamond() {
        let cfg = build_diamond_cfg();
        let rpo = reverse_postorder(&cfg);

        // Entry must come first
        assert_eq!(rpo[0], BlockId(0));
        // Merge must come last (after both branches)
        assert_eq!(*rpo.last().unwrap(), BlockId(3));
        // Both left and right must appear between entry and merge
        assert!(rpo.contains(&BlockId(1)));
        assert!(rpo.contains(&BlockId(2)));
        assert_eq!(rpo.len(), 4);
    }

    #[test]
    fn test_rpo_loop() {
        let cfg = build_loop_cfg();
        let rpo = reverse_postorder(&cfg);

        // Entry must come first
        assert_eq!(rpo[0], BlockId(0));
        // Header must come before body
        let header_pos = rpo.iter().position(|&b| b == BlockId(1)).unwrap();
        let body_pos = rpo.iter().position(|&b| b == BlockId(2)).unwrap();
        assert!(header_pos < body_pos);
    }

    // =======================================================================
    // Dominance tree tests
    // =======================================================================

    #[test]
    fn test_domtree_linear_chain() {
        let cfg = build_linear_cfg();
        let dom = compute_dominance_tree(&cfg);

        assert_eq!(dom.idom(BlockId(0)), Some(BlockId(0)));
        assert_eq!(dom.idom(BlockId(1)), Some(BlockId(0)));
        assert_eq!(dom.idom(BlockId(2)), Some(BlockId(1)));
        assert_eq!(dom.idom(BlockId(3)), Some(BlockId(2)));
    }

    #[test]
    fn test_domtree_diamond() {
        let cfg = build_diamond_cfg();
        let dom = compute_dominance_tree(&cfg);

        assert_eq!(dom.idom(BlockId(0)), Some(BlockId(0)));
        assert_eq!(dom.idom(BlockId(1)), Some(BlockId(0)));
        assert_eq!(dom.idom(BlockId(2)), Some(BlockId(0)));
        // Merge is dominated by entry (common dominator of left and right)
        assert_eq!(dom.idom(BlockId(3)), Some(BlockId(0)));
    }

    #[test]
    fn test_domtree_loop() {
        let cfg = build_loop_cfg();
        let dom = compute_dominance_tree(&cfg);

        assert_eq!(dom.idom(BlockId(1)), Some(BlockId(0)));
        assert_eq!(dom.idom(BlockId(2)), Some(BlockId(1)));
        assert_eq!(dom.idom(BlockId(3)), Some(BlockId(1)));
    }

    #[test]
    fn test_dominates_queries() {
        let cfg = build_diamond_cfg();
        let dom = compute_dominance_tree(&cfg);

        // Entry dominates everything
        assert!(dom.dominates(BlockId(0), BlockId(0)));
        assert!(dom.dominates(BlockId(0), BlockId(1)));
        assert!(dom.dominates(BlockId(0), BlockId(2)));
        assert!(dom.dominates(BlockId(0), BlockId(3)));

        // Left does not dominate merge (right also leads to merge)
        assert!(!dom.dominates(BlockId(1), BlockId(3)));
        assert!(!dom.dominates(BlockId(2), BlockId(3)));

        // Nothing dominates entry (except itself)
        assert!(!dom.dominates(BlockId(1), BlockId(0)));
        assert!(!dom.dominates(BlockId(3), BlockId(0)));

        // Self-dominance
        assert!(dom.dominates(BlockId(1), BlockId(1)));
        assert!(dom.dominates(BlockId(3), BlockId(3)));
    }

    #[test]
    fn test_domtree_children() {
        let cfg = build_linear_cfg();
        let dom = compute_dominance_tree(&cfg);

        assert_eq!(dom.children(BlockId(0)), &[BlockId(1)]);
        assert_eq!(dom.children(BlockId(1)), &[BlockId(2)]);
        assert_eq!(dom.children(BlockId(2)), &[BlockId(3)]);
        assert!(dom.children(BlockId(3)).is_empty());
    }

    #[test]
    fn test_domtree_preorder() {
        let cfg = build_linear_cfg();
        let dom = compute_dominance_tree(&cfg);
        let pre = dom.preorder(BlockId(0));
        assert_eq!(pre, vec![BlockId(0), BlockId(1), BlockId(2), BlockId(3)]);
    }

    // =======================================================================
    // Dominance frontier tests
    // =======================================================================

    #[test]
    fn test_domfrontier_linear_chain() {
        let cfg = build_linear_cfg();
        let dom = compute_dominance_tree(&cfg);
        let df = compute_dominance_frontiers(&cfg, &dom);

        // In a linear chain, all dominance frontiers are empty
        for i in 0..4 {
            assert!(
                df.get(&BlockId(i)).map_or(true, |s| s.is_empty()),
                "DF(bb{}) should be empty",
                i
            );
        }
    }

    #[test]
    fn test_domfrontier_diamond() {
        let cfg = build_diamond_cfg();
        let dom = compute_dominance_tree(&cfg);
        let df = compute_dominance_frontiers(&cfg, &dom);

        // DF(left) = {merge}, DF(right) = {merge}
        assert!(df[&BlockId(1)].contains(&BlockId(3)));
        assert!(df[&BlockId(2)].contains(&BlockId(3)));

        // DF(entry) = {}, DF(merge) = {}
        assert!(df[&BlockId(0)].is_empty());
        assert!(df[&BlockId(3)].is_empty());
    }

    #[test]
    fn test_domfrontier_loop() {
        let cfg = build_loop_cfg();
        let dom = compute_dominance_tree(&cfg);
        let df = compute_dominance_frontiers(&cfg, &dom);

        // DF(body) should contain header (the loop back-edge target)
        assert!(df[&BlockId(2)].contains(&BlockId(1)));

        // DF(header) should contain header (it's a join point from entry and body)
        assert!(df[&BlockId(1)].contains(&BlockId(1)));
    }

    // =======================================================================
    // Loop detection tests
    // =======================================================================

    #[test]
    fn test_detect_no_loops() {
        let cfg = build_linear_cfg();
        let dom = compute_dominance_tree(&cfg);
        let loops = detect_loops(&cfg, &dom);
        assert!(loops.is_empty());
    }

    #[test]
    fn test_detect_simple_loop() {
        let cfg = build_loop_cfg();
        let dom = compute_dominance_tree(&cfg);
        let loops = detect_loops(&cfg, &dom);

        assert_eq!(loops.len(), 1);
        let the_loop = &loops[0];
        assert_eq!(the_loop.header, BlockId(1));
        assert!(the_loop.blocks.contains(&BlockId(1)));
        assert!(the_loop.blocks.contains(&BlockId(2)));
        assert!(!the_loop.blocks.contains(&BlockId(0)));
        assert!(!the_loop.blocks.contains(&BlockId(3)));
        assert_eq!(the_loop.depth, 1);
    }

    #[test]
    fn test_detect_nested_loops() {
        // Build nested loop:
        //   entry(0) → outer_header(1) → inner_header(2) → inner_body(3) → inner_header(2)
        //                                    |                                [inner back edge]
        //                                    v
        //                               outer_body(4) → outer_header(1)  [outer back edge]
        //               outer_header(1) → exit(5)
        let mut cfg = ControlFlowGraph::new(BlockId(0));

        let mut entry = make_block(0, "entry");
        entry.terminator = Some(Terminator::Branch { target: BlockId(1) });
        cfg.add_block(entry);

        let mut oh = make_block(1, "outer_header");
        oh.terminator = Some(Terminator::CondBranch {
            condition: Value(0),
            true_block: BlockId(2),
            false_block: BlockId(5),
        });
        cfg.add_block(oh);

        let mut ih = make_block(2, "inner_header");
        ih.terminator = Some(Terminator::CondBranch {
            condition: Value(1),
            true_block: BlockId(3),
            false_block: BlockId(4),
        });
        cfg.add_block(ih);

        let mut ib = make_block(3, "inner_body");
        ib.terminator = Some(Terminator::Branch { target: BlockId(2) });
        cfg.add_block(ib);

        let mut ob = make_block(4, "outer_body");
        ob.terminator = Some(Terminator::Branch { target: BlockId(1) });
        cfg.add_block(ob);

        let mut exit = make_block(5, "exit");
        exit.terminator = Some(Terminator::Return { value: None });
        cfg.add_block(exit);

        cfg.compute_edges();

        let dom = compute_dominance_tree(&cfg);
        let loops = detect_loops(&cfg, &dom);

        // There should be at least 1 top-level loop (the outer loop)
        // Inner loop may appear as nested or as separate depending on
        // the nesting logic. Let's verify basics.
        let all_loops = collect_all_loops(&loops);
        assert!(
            all_loops.len() >= 2,
            "Expected at least 2 loops (inner + outer)"
        );

        // Find the inner and outer loops
        let inner = all_loops
            .iter()
            .find(|l| l.header == BlockId(2))
            .expect("Inner loop with header=2");
        let outer = all_loops
            .iter()
            .find(|l| l.header == BlockId(1))
            .expect("Outer loop with header=1");

        assert!(inner.blocks.contains(&BlockId(2)));
        assert!(inner.blocks.contains(&BlockId(3)));
        assert!(outer.blocks.contains(&BlockId(1)));
        assert!(outer.blocks.contains(&BlockId(2)));
        assert!(outer.blocks.contains(&BlockId(3)));
        assert!(outer.blocks.contains(&BlockId(4)));

        // Inner loop should have greater depth than outer
        assert!(inner.depth > outer.depth);
    }

    /// Recursively collect all loops (top-level and nested) into a flat list.
    fn collect_all_loops(loops: &[Loop]) -> Vec<&Loop> {
        let mut result = Vec::new();
        for l in loops {
            result.push(l);
            result.extend(collect_all_loops(&l.inner_loops));
        }
        result
    }

    #[test]
    fn test_detect_loop_with_multiple_back_edges() {
        // header(0) ← body_a(1) and header(0) ← body_b(2)
        //
        //   entry(3) → header(0) → body_a(1) → header(0) [back edge A]
        //                    |
        //                    → body_b(2) → header(0) [back edge B]
        //                    |
        //                    → exit(4)
        let mut cfg = ControlFlowGraph::new(BlockId(3));

        let mut entry = make_block(3, "entry");
        entry.terminator = Some(Terminator::Branch { target: BlockId(0) });
        cfg.add_block(entry);

        let mut header = make_block(0, "header");
        header.terminator = Some(Terminator::Switch {
            value: Value(0),
            default: BlockId(4),
            cases: vec![(1, BlockId(1)), (2, BlockId(2))],
        });
        cfg.add_block(header);

        let mut body_a = make_block(1, "body_a");
        body_a.terminator = Some(Terminator::Branch { target: BlockId(0) });
        cfg.add_block(body_a);

        let mut body_b = make_block(2, "body_b");
        body_b.terminator = Some(Terminator::Branch { target: BlockId(0) });
        cfg.add_block(body_b);

        let mut exit = make_block(4, "exit");
        exit.terminator = Some(Terminator::Return { value: None });
        cfg.add_block(exit);

        cfg.compute_edges();

        let dom = compute_dominance_tree(&cfg);
        let loops = detect_loops(&cfg, &dom);

        // Should detect one loop (header=0 with two back edges)
        assert_eq!(loops.len(), 1);
        let the_loop = &loops[0];
        assert_eq!(the_loop.header, BlockId(0));
        assert!(the_loop.blocks.contains(&BlockId(0)));
        assert!(the_loop.blocks.contains(&BlockId(1)));
        assert!(the_loop.blocks.contains(&BlockId(2)));
        assert_eq!(the_loop.back_edges.len(), 2);
    }

    // =======================================================================
    // Reachability tests
    // =======================================================================

    #[test]
    fn test_all_blocks_reachable() {
        let cfg = build_diamond_cfg();
        let reachable = cfg.reachable_blocks();
        assert_eq!(reachable.len(), 4);
        for i in 0..4 {
            assert!(reachable.contains(&BlockId(i)));
        }
        assert!(cfg.unreachable_blocks().is_empty());
    }

    #[test]
    fn test_unreachable_block_detected() {
        let mut cfg = ControlFlowGraph::new(BlockId(0));

        let mut a = make_block(0, "A");
        a.terminator = Some(Terminator::Return { value: None });
        cfg.add_block(a);

        let mut b = make_block(1, "B");
        b.terminator = Some(Terminator::Return { value: None });
        cfg.add_block(b);

        // B is unreachable (no edge from A to B)
        let reachable = cfg.reachable_blocks();
        assert_eq!(reachable.len(), 1);
        assert!(reachable.contains(&BlockId(0)));

        let unreachable = cfg.unreachable_blocks();
        assert_eq!(unreachable, vec![BlockId(1)]);
    }

    // =======================================================================
    // Well-formedness tests
    // =======================================================================

    #[test]
    fn test_well_formed_cfg() {
        let cfg = build_diamond_cfg();
        assert!(cfg.is_well_formed());
    }

    #[test]
    fn test_not_well_formed_missing_terminator() {
        let mut cfg = ControlFlowGraph::new(BlockId(0));
        let blk = make_block(0, "entry");
        // No terminator set
        cfg.add_block(blk);
        assert!(!cfg.is_well_formed());
    }

    #[test]
    fn test_well_formed_inconsistent_edges() {
        let mut cfg = ControlFlowGraph::new(BlockId(0));

        let mut a = make_block(0, "A");
        a.terminator = Some(Terminator::Return { value: None });
        // Manually add a successor without corresponding predecessor
        a.successors.push(BlockId(1));
        cfg.add_block(a);

        let mut b = make_block(1, "B");
        b.terminator = Some(Terminator::Return { value: None });
        cfg.add_block(b);

        assert!(!cfg.is_well_formed());
    }

    // =======================================================================
    // Members accessed validation tests
    // =======================================================================

    #[test]
    fn test_value_undef_usage() {
        // Verify Value::undef() works correctly in phi nodes
        let phi = PhiNode {
            result: Value(10),
            ty: IrType::I32,
            incoming: vec![(Value::undef(), BlockId(0))],
        };
        assert_eq!(phi.result.0, 10);
        assert_eq!(phi.incoming[0].0, Value::undef());
        assert_eq!(phi.incoming[0].1 .0, 0); // BlockId.0 access
    }

    #[test]
    fn test_blockid_inner_value() {
        // Verify BlockId.0 is accessible
        let bid = BlockId(42);
        assert_eq!(bid.0, 42);
    }

    #[test]
    fn test_value_inner_value() {
        // Verify Value.0 is accessible
        let val = Value(99);
        assert_eq!(val.0, 99);
    }
}
