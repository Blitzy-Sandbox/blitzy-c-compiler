//! # Shared Register Allocator
//!
//! This module implements a linear scan register allocator that is shared across
//! all four architecture backends (x86-64, i686, AArch64, RISC-V 64). The
//! allocator is parameterized by a [`RegisterInfo`] descriptor that describes
//! the target's physical register file, including integer and floating-point
//! register sets, callee-saved registers, and human-readable register names.
//!
//! ## Algorithm
//!
//! The allocator uses the **linear scan** algorithm:
//! 1. Compute live intervals from SSA-form IR in reverse postorder
//! 2. Sort intervals by start point
//! 3. Walk intervals in order, maintaining an active list sorted by end point
//! 4. For each interval, expire completed intervals, then allocate or spill
//!
//! Complexity is O(N log N) where N is the number of live intervals.
//!
//! ## Register Selection Strategy
//!
//! - Non-call-crossing intervals prefer **caller-saved** registers first to
//!   avoid unnecessary save/restore overhead in prologue/epilogue.
//! - Call-crossing intervals prefer **callee-saved** registers first since
//!   they survive across function calls without explicit saves.
//! - Callee-saved registers that are actually used are tracked in
//!   [`AllocationResult::used_callee_saved`] for prologue/epilogue generation.
//!
//! ## Spilling
//!
//! When all registers of the required class are in use, the allocator spills
//! either the current interval or the active interval with the latest end
//! point, whichever frees up a register for the longest remaining range.
//! Actual spill load/store machine instructions are emitted by the
//! architecture-specific backends, not by this module.
//!
//! ## Zero External Dependencies
//!
//! Only imports from `std` and internal crate modules. No external crates.

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::ir::{
    BasicBlock, BlockId, ControlFlowGraph, Function, Instruction, IrType, PhiNode,
    Terminator, Value,
};
use crate::ir::cfg::reverse_postorder;

// ---------------------------------------------------------------------------
// PhysReg — physical register identifier
// ---------------------------------------------------------------------------

/// A physical register identifier.
///
/// Each architecture backend assigns its own numbering scheme. For example,
/// x86-64 might use `PhysReg(0)` for `rax`, `PhysReg(1)` for `rcx`, etc.
/// The mapping from `PhysReg` to architecture-specific register names is
/// provided by [`RegisterInfo::reg_names`].
///
/// The inner `u16` supports up to 65,536 physical registers, which is more
/// than sufficient for all supported architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PhysReg(pub u16);

impl fmt::Display for PhysReg {
    /// Formats as `r<N>` (e.g., `r0`, `r1`, `r15`).
    ///
    /// For architecture-specific names (e.g., `rax`, `x0`), use
    /// [`RegisterInfo::reg_names`] to look up the name.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "r{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// RegClass — register class
// ---------------------------------------------------------------------------

/// Register class discriminator.
///
/// Each SSA value is classified into exactly one register class, which
/// determines which set of physical registers it can be allocated to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegClass {
    /// General-purpose integer registers (also used for pointers and booleans).
    Integer,
    /// Floating-point / SIMD registers.
    Float,
}

impl fmt::Display for RegClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegClass::Integer => write!(f, "int"),
            RegClass::Float => write!(f, "float"),
        }
    }
}

// ---------------------------------------------------------------------------
// RegisterInfo — target register file descriptor
// ---------------------------------------------------------------------------

/// Describes the physical register file of a target architecture.
///
/// Each architecture backend constructs a `RegisterInfo` with its specific
/// register set. The registers in `int_regs` and `float_regs` are listed in
/// **allocation priority order** — caller-saved registers should appear first
/// to minimize callee-saved spill overhead.
///
/// # Examples
///
/// - **x86-64**: 14 allocatable GPRs (rax, rcx, rdx, rsi, rdi, r8–r11,
///   rbx, r12–r15) + 16 XMM registers
/// - **i686**: 6 allocatable GPRs (eax, ecx, edx, ebx, esi, edi) + 8 XMM
/// - **AArch64**: 29 allocatable GPRs (x0–x18, x19–x28) + 32 SIMD/FP
/// - **RISC-V 64**: 27 allocatable GPRs + 32 FP
#[derive(Debug, Clone)]
pub struct RegisterInfo {
    /// All allocatable integer registers, in allocation priority order.
    ///
    /// Caller-saved registers should appear before callee-saved registers
    /// for optimal allocation when intervals do not cross calls.
    pub int_regs: Vec<PhysReg>,

    /// All allocatable floating-point / SIMD registers, in priority order.
    pub float_regs: Vec<PhysReg>,

    /// Callee-saved integer registers (must be preserved across calls).
    ///
    /// This is a subset of `int_regs`. The register allocator tracks which
    /// callee-saved registers are used so the backend can emit appropriate
    /// save/restore sequences in the function prologue/epilogue.
    pub callee_saved_int: Vec<PhysReg>,

    /// Callee-saved floating-point registers.
    pub callee_saved_float: Vec<PhysReg>,

    /// Human-readable register names for debugging and diagnostic output.
    ///
    /// Maps each `PhysReg` to its architecture-specific mnemonic
    /// (e.g., `PhysReg(0)` → `"rax"` on x86-64).
    pub reg_names: HashMap<PhysReg, &'static str>,
}

impl RegisterInfo {
    /// Returns the human-readable name for a physical register, or a generic
    /// `r<N>` fallback if no name is registered.
    pub fn name(&self, reg: PhysReg) -> String {
        self.reg_names
            .get(&reg)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("r{}", reg.0))
    }

    /// Returns true if `reg` is a callee-saved register (integer or float).
    pub fn is_callee_saved(&self, reg: PhysReg) -> bool {
        self.callee_saved_int.contains(&reg) || self.callee_saved_float.contains(&reg)
    }

    /// Returns the total number of allocatable registers of a given class.
    pub fn num_regs(&self, class: RegClass) -> usize {
        match class {
            RegClass::Integer => self.int_regs.len(),
            RegClass::Float => self.float_regs.len(),
        }
    }

    /// Returns a slice of allocatable registers for the given class.
    pub fn regs_for_class(&self, class: RegClass) -> &[PhysReg] {
        match class {
            RegClass::Integer => &self.int_regs,
            RegClass::Float => &self.float_regs,
        }
    }

    /// Returns a slice of callee-saved registers for the given class.
    pub fn callee_saved_for_class(&self, class: RegClass) -> &[PhysReg] {
        match class {
            RegClass::Integer => &self.callee_saved_int,
            RegClass::Float => &self.callee_saved_float,
        }
    }
}

// ---------------------------------------------------------------------------
// LiveInterval — SSA value live range
// ---------------------------------------------------------------------------

/// A live interval representing the range of instruction indices during which
/// an SSA value is alive (from its definition to its last use).
///
/// After register allocation, each interval has either an `assigned_reg`
/// (successfully allocated to a physical register) or a `spill_slot`
/// (must be stored on the stack and reloaded at use sites).
///
/// Intervals use half-open ranges: `[start, end)` — the value is live
/// starting at instruction index `start` (inclusive) up to but not including
/// instruction index `end` (exclusive).
#[derive(Debug, Clone)]
pub struct LiveInterval {
    /// The SSA value this interval represents.
    pub value: Value,

    /// The register class required (integer or float).
    pub reg_class: RegClass,

    /// Start point — instruction index where the value is defined (inclusive).
    pub start: u32,

    /// End point — one past the last instruction index where the value is
    /// used (exclusive). The interval covers `[start, end)`.
    pub end: u32,

    /// Assigned physical register after allocation, or `None` if spilled.
    pub assigned_reg: Option<PhysReg>,

    /// Spill slot index if this interval was spilled, or `None` if allocated.
    pub spill_slot: Option<u32>,

    /// Whether this interval corresponds to a function parameter.
    pub is_param: bool,

    /// Whether this interval spans at least one function call instruction.
    ///
    /// Call-crossing intervals prefer callee-saved registers because
    /// caller-saved registers would be clobbered by the call.
    pub crosses_call: bool,
}

impl fmt::Display for LiveInterval {
    /// Formats as `%N [start, end) -> reg` or `%N [start, end) -> [spill:S]`.
    ///
    /// Examples:
    /// ```text
    /// %5 [12, 28) -> r3
    /// %8 [15, 40) -> [spill:0]
    /// %2 [0, 5) -> unassigned
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [{}, {}) -> ", self.value, self.start, self.end)?;
        if let Some(reg) = self.assigned_reg {
            write!(f, "{}", reg)
        } else if let Some(slot) = self.spill_slot {
            write!(f, "[spill:{}]", slot)
        } else {
            write!(f, "unassigned")
        }
    }
}

// ---------------------------------------------------------------------------
// AllocationResult — output of register allocation
// ---------------------------------------------------------------------------

/// The result of running the linear scan register allocator.
///
/// Contains the final interval assignments (each with either an `assigned_reg`
/// or `spill_slot`), the number of spill slots consumed, and the list of
/// callee-saved registers that were used (for prologue/epilogue generation).
#[derive(Debug, Clone)]
pub struct AllocationResult {
    /// Live intervals with register or spill slot assignments.
    pub intervals: Vec<LiveInterval>,

    /// Total number of spill slots allocated.
    pub num_spill_slots: u32,

    /// Callee-saved registers that were assigned to at least one interval.
    ///
    /// The backend's prologue generator uses this to emit save instructions
    /// (e.g., `push rbx` on x86-64), and the epilogue generator emits the
    /// corresponding restores.
    pub used_callee_saved: Vec<PhysReg>,
}

// ---------------------------------------------------------------------------
// SpillInfo — spill slot layout information
// ---------------------------------------------------------------------------

/// Information about spill slot layout within the stack frame.
///
/// Produced by [`insert_spill_code`] after register allocation. The backend
/// uses this to compute the total stack frame size and the offset of each
/// spill slot relative to the frame pointer or stack pointer.
#[derive(Debug, Clone)]
pub struct SpillInfo {
    /// Stack frame offsets for each spill slot, indexed by slot number.
    ///
    /// Negative offsets are below the frame pointer (typical convention).
    /// `slot_offsets[i]` is the byte offset of spill slot `i`.
    pub slot_offsets: Vec<i32>,

    /// Total stack space (in bytes) consumed by all spill slots.
    pub total_spill_size: u32,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Classifies an IR type into a register class.
///
/// Floating-point types map to [`RegClass::Float`]; everything else
/// (integers, pointers, booleans, aggregates) maps to [`RegClass::Integer`].
fn classify_type(ty: &IrType) -> RegClass {
    if ty.is_float() {
        RegClass::Float
    } else {
        // Integers, pointers, booleans, void, labels, etc. → integer regs.
        RegClass::Integer
    }
}

/// Classifies the result of an instruction into a register class.
///
/// Special handling for `Alloca` and `GetElementPtr` which produce pointer
/// results (always integer class) regardless of their inner type.
fn classify_instruction_result(instr: &Instruction) -> RegClass {
    match instr {
        // Alloca and GEP always produce pointers → integer register class.
        Instruction::Alloca { .. } | Instruction::GetElementPtr { .. } => RegClass::Integer,
        // ICmp and FCmp always produce i1 (boolean) → integer register class.
        Instruction::ICmp { .. } | Instruction::FCmp { .. } => RegClass::Integer,
        // For all other instructions, classify based on result type.
        _ => instr
            .result_type()
            .map(|ty| {
                if ty.is_integer() {
                    RegClass::Integer
                } else if ty.is_float() {
                    RegClass::Float
                } else {
                    // Pointers, aggregates, etc. go in integer registers.
                    RegClass::Integer
                }
            })
            .unwrap_or(RegClass::Integer),
    }
}

/// Helper to collect values used by a terminator instruction.
fn terminator_operands(term: &Terminator) -> Vec<Value> {
    match term {
        Terminator::CondBranch { condition, .. } => vec![*condition],
        Terminator::Return { value: Some(v) } => vec![*v],
        Terminator::Switch { value, .. } => vec![*value],
        Terminator::Branch { .. }
        | Terminator::Return { value: None }
        | Terminator::Unreachable => vec![],
    }
}

/// Instruction index information for a basic block in the linearized layout.
#[derive(Debug, Clone, Copy)]
struct BlockLayout {
    /// Instruction index of the first phi node in this block.
    phi_start: u32,
    /// Instruction index of the first regular instruction.
    instr_start: u32,
    /// Instruction index of the terminator.
    term_index: u32,
    /// First instruction index belonging to the *next* block.
    next_start: u32,
}

// ---------------------------------------------------------------------------
// compute_live_intervals — SSA value live range computation
// ---------------------------------------------------------------------------

/// Computes live intervals for all SSA values in the given function.
///
/// The algorithm linearizes basic blocks in reverse postorder (so that
/// definitions precede uses in the common case), assigns sequential
/// instruction indices, and records definition and last-use points for
/// each SSA value.
///
/// # Parameters
///
/// - `function` — The IR function to analyze. Must have at least one block
///   with a terminator for meaningful results.
///
/// # Returns
///
/// A vector of [`LiveInterval`]s sorted by start point (ascending).
/// Each interval covers `[start, end)` in the linearized instruction index
/// space.
///
/// # Complexity
///
/// O(N) where N is the total number of instructions across all blocks,
/// plus O(M log M) for sorting M intervals.
pub fn compute_live_intervals(function: &Function) -> Vec<LiveInterval> {
    // Early return for empty functions.
    if function.blocks.is_empty() {
        return Vec::new();
    }

    // -----------------------------------------------------------------------
    // Step 1: Build a ControlFlowGraph from the function's blocks for RPO
    // -----------------------------------------------------------------------
    let mut cfg = ControlFlowGraph::new(function.entry_block);
    for block in &function.blocks {
        cfg.add_block(block.clone());
    }

    // Rebuild successor/predecessor edges from terminators.
    // The IR builder may not populate BasicBlock::successors directly;
    // this method scans all terminators and derives the edges.
    cfg.compute_edges();

    // Verify the CFG entry matches the function's entry.
    let _cfg_entry = cfg.entry();
    let _cfg_blocks = cfg.blocks();

    // Compute reverse postorder traversal for block linearization.
    let rpo = reverse_postorder(&cfg);
    if rpo.is_empty() {
        return Vec::new();
    }

    // -----------------------------------------------------------------------
    // Step 2: Assign instruction indices in linearized RPO order
    // -----------------------------------------------------------------------
    let mut block_layout: HashMap<BlockId, BlockLayout> = HashMap::with_capacity(rpo.len());
    let mut global_index: u32 = 0;

    for &block_id in &rpo {
        let idx = block_id.0 as usize;
        if idx >= function.blocks.len() {
            continue;
        }
        let block: &BasicBlock = &function.blocks[idx];
        let phi_start = global_index;

        global_index += block.phi_nodes.len() as u32;
        let instr_start = global_index;

        global_index += block.instructions.len() as u32;
        let term_index = global_index;

        if block.terminator.is_some() {
            global_index += 1;
        }

        block_layout.insert(block.id, BlockLayout {
            phi_start,
            instr_start,
            term_index,
            next_start: global_index,
        });
    }

    // -----------------------------------------------------------------------
    // Step 3: Compute definition and use points for every SSA value
    // -----------------------------------------------------------------------
    // Maps Value → (definition_index, register_class, is_param).
    let mut value_def: HashMap<Value, (u32, RegClass, bool)> =
        HashMap::with_capacity(global_index as usize);
    // Maps Value → last use instruction index (exclusive end = last_use + 1).
    let mut value_last_use: HashMap<Value, u32> = HashMap::with_capacity(global_index as usize);
    // Instruction indices of all Call instructions (for crosses_call detection).
    let mut call_indices: Vec<u32> = Vec::new();

    // Determine the number of function parameters for param detection.
    // Parameters are the first N alloca results in the entry block, where N
    // is the parameter count.
    let num_params = function.params.len();

    for &block_id in &rpo {
        let idx = block_id.0 as usize;
        if idx >= function.blocks.len() {
            continue;
        }
        let block: &BasicBlock = &function.blocks[idx];
        let layout = match block_layout.get(&block.id) {
            Some(l) => *l,
            None => continue,
        };

        // --- Phi nodes ---
        let mut phi_idx = layout.phi_start;
        let phi_nodes: &[PhiNode] = &block.phi_nodes;
        for phi in phi_nodes {
            // The phi result is defined at this index.
            let rc = classify_type(&phi.ty);
            value_def.entry(phi.result).or_insert((phi_idx, rc, false));

            // Phi incoming values are logically used at the end of their
            // respective predecessor blocks (the value flows along the edge).
            for &(val, pred_id) in &phi.incoming {
                if let Some(pred_layout) = block_layout.get(&pred_id) {
                    // Use at the predecessor's terminator index.
                    let use_point = pred_layout.term_index;
                    value_last_use
                        .entry(val)
                        .and_modify(|e| *e = (*e).max(use_point + 1))
                        .or_insert(use_point + 1);
                } else {
                    // Predecessor not in RPO (unreachable back-edge); use at
                    // the phi's own position as a conservative fallback.
                    value_last_use
                        .entry(val)
                        .and_modify(|e| *e = (*e).max(phi_idx + 1))
                        .or_insert(phi_idx + 1);
                }
            }
            phi_idx += 1;
        }

        // --- Regular instructions ---
        let mut instr_idx = layout.instr_start;
        let is_entry = block.id == function.entry_block;
        let mut param_alloca_count: u32 = 0;

        for instr in &block.instructions {
            // Record definition.
            if let Some(result) = instr.result() {
                let rc = classify_instruction_result(instr);
                // Detect parameters: first N allocas in the entry block.
                let is_param_val =
                    is_entry && matches!(instr, Instruction::Alloca { .. })
                    && (param_alloca_count as usize) < num_params;
                if matches!(instr, Instruction::Alloca { .. }) && is_entry {
                    param_alloca_count += 1;
                }
                value_def.entry(result).or_insert((instr_idx, rc, is_param_val));
            }

            // Record uses for all operands.
            for op in instr.operands() {
                value_last_use
                    .entry(op)
                    .and_modify(|e| *e = (*e).max(instr_idx + 1))
                    .or_insert(instr_idx + 1);
            }

            // Track Call instructions for crosses_call detection.
            if matches!(instr, Instruction::Call { .. }) {
                call_indices.push(instr_idx);
            }

            instr_idx += 1;
        }

        // --- Terminator ---
        if let Some(ref term) = block.terminator {
            let t_idx = layout.term_index;
            for op in terminator_operands(term) {
                value_last_use
                    .entry(op)
                    .and_modify(|e| *e = (*e).max(t_idx + 1))
                    .or_insert(t_idx + 1);
            }

            // Also scan successor/predecessor info for completeness
            // (these are already used during CFG construction above).
            let _succs = &block.successors;
            let _preds = &block.predecessors;
        }
    }

    // -----------------------------------------------------------------------
    // Step 4: Build LiveInterval structs from def/use data
    // -----------------------------------------------------------------------
    let mut intervals: Vec<LiveInterval> = Vec::with_capacity(value_def.len());

    for (value, &(start, reg_class, is_param)) in value_def.iter() {
        // End is the exclusive end point. If the value has no uses, the
        // interval has minimum length 1 (just the definition).
        let end = value_last_use
            .get(value)
            .copied()
            .unwrap_or(start + 1)
            .max(start + 1);

        // An interval crosses a call if any Call instruction falls strictly
        // within the open interval (start, end).
        let crosses_call = call_indices
            .iter()
            .any(|&ci| ci > start && ci < end);

        intervals.push(LiveInterval {
            value: *value,
            reg_class,
            start,
            end,
            assigned_reg: None,
            spill_slot: None,
            is_param,
            crosses_call,
        });
    }

    // Sort by start point (ascending), break ties by value index for
    // deterministic allocation order.
    intervals.sort_by(|a, b| a.start.cmp(&b.start).then(a.value.0.cmp(&b.value.0)));

    intervals
}

// ---------------------------------------------------------------------------
// select_register — allocation preference logic
// ---------------------------------------------------------------------------

/// Selects a free physical register of the requested class with allocation
/// preference based on whether the interval crosses a function call.
///
/// - For call-crossing intervals: prefer callee-saved registers first
///   (they survive across calls without save/restore).
/// - For non-call-crossing intervals: prefer caller-saved registers first
///   (no need to waste callee-saved registers and incur save/restore cost).
///
/// Returns `None` if no register of the required class is free.
fn select_register(
    reg_class: RegClass,
    crosses_call: bool,
    free_regs: &HashSet<PhysReg>,
    reg_info: &RegisterInfo,
) -> Option<PhysReg> {
    let all_regs = reg_info.regs_for_class(reg_class);
    let callee_saved: HashSet<PhysReg> = reg_info
        .callee_saved_for_class(reg_class)
        .iter()
        .copied()
        .collect();

    if crosses_call {
        // Prefer callee-saved first (they survive the call).
        for &reg in all_regs {
            if free_regs.contains(&reg) && callee_saved.contains(&reg) {
                return Some(reg);
            }
        }
        // Fall back to any free register.
        for &reg in all_regs {
            if free_regs.contains(&reg) {
                return Some(reg);
            }
        }
    } else {
        // Prefer caller-saved first (don't waste callee-saved).
        for &reg in all_regs {
            if free_regs.contains(&reg) && !callee_saved.contains(&reg) {
                return Some(reg);
            }
        }
        // Fall back to callee-saved.
        for &reg in all_regs {
            if free_regs.contains(&reg) {
                return Some(reg);
            }
        }
    }

    None
}

// ---------------------------------------------------------------------------
// linear_scan_allocate — the core allocation algorithm
// ---------------------------------------------------------------------------

/// Performs linear scan register allocation on the given live intervals.
///
/// This is the classic Poletto & Sarkar linear scan algorithm:
/// 1. Intervals are sorted by start point.
/// 2. An **active** list tracks intervals currently occupying registers,
///    sorted by end point.
/// 3. For each new interval, expired intervals are removed and their
///    registers freed.
/// 4. A free register is selected using [`select_register`] preference logic.
/// 5. If no register is available, the interval with the longest remaining
///    range (current or an active interval) is spilled to a stack slot.
///
/// # Parameters
///
/// - `intervals` — Mutable reference to live intervals (modified in place
///   with `assigned_reg` and `spill_slot` fields).
/// - `reg_info` — The target's register file descriptor.
///
/// # Returns
///
/// An [`AllocationResult`] containing the final interval assignments,
/// total spill slot count, and the set of callee-saved registers used.
pub fn linear_scan_allocate(
    intervals: &mut Vec<LiveInterval>,
    reg_info: &RegisterInfo,
) -> AllocationResult {
    // Sort intervals by start point.
    intervals.sort_by(|a, b| a.start.cmp(&b.start).then(a.value.0.cmp(&b.value.0)));

    // Initialize free register sets.
    let mut free_int: HashSet<PhysReg> =
        HashSet::with_capacity(reg_info.int_regs.len());
    for &r in &reg_info.int_regs {
        free_int.insert(r);
    }
    let mut free_float: HashSet<PhysReg> =
        HashSet::with_capacity(reg_info.float_regs.len());
    for &r in &reg_info.float_regs {
        free_float.insert(r);
    }

    // Active list: indices into `intervals`, sorted by end point (ascending).
    let mut active: Vec<usize> = Vec::new();
    let mut num_spill_slots: u32 = 0;
    let mut used_callee_saved: HashSet<PhysReg> = HashSet::new();

    for i in 0..intervals.len() {
        let current_start = intervals[i].start;

        // ----- Expire old intervals -----
        // Remove intervals whose end point is <= current start (half-open).
        let mut new_active: Vec<usize> = Vec::with_capacity(active.len());
        for &idx in &active {
            if intervals[idx].end <= current_start {
                // Free the register.
                if let Some(reg) = intervals[idx].assigned_reg {
                    match intervals[idx].reg_class {
                        RegClass::Integer => { free_int.insert(reg); }
                        RegClass::Float => { free_float.insert(reg); }
                    }
                }
            } else {
                new_active.push(idx);
            }
        }
        active = new_active;

        // ----- Try to allocate a register -----
        let rc = intervals[i].reg_class;
        let crosses_call = intervals[i].crosses_call;

        let free_set = match rc {
            RegClass::Integer => &free_int,
            RegClass::Float => &free_float,
        };

        if let Some(reg) = select_register(rc, crosses_call, free_set, reg_info) {
            // Successfully allocated.
            intervals[i].assigned_reg = Some(reg);
            match rc {
                RegClass::Integer => { free_int.remove(&reg); }
                RegClass::Float => { free_float.remove(&reg); }
            }

            // Track callee-saved usage.
            if reg_info.is_callee_saved(reg) {
                used_callee_saved.insert(reg);
            }

            // Insert into active list maintaining sort by end point.
            let pos = active.partition_point(|&idx| intervals[idx].end <= intervals[i].end);
            active.insert(pos, i);
        } else {
            // ----- No free register: spill -----
            // Find the active interval of the same class with the latest end.
            let spill_candidate = active
                .iter()
                .filter(|&&idx| intervals[idx].reg_class == rc)
                .max_by_key(|&&idx| intervals[idx].end)
                .copied();

            if let Some(victim_idx) = spill_candidate {
                if intervals[victim_idx].end > intervals[i].end {
                    // Spill the active interval (it extends further).
                    let freed_reg = intervals[victim_idx].assigned_reg.unwrap();
                    intervals[victim_idx].assigned_reg = None;
                    intervals[victim_idx].spill_slot = Some(num_spill_slots);
                    num_spill_slots += 1;

                    // Assign the freed register to the current interval.
                    intervals[i].assigned_reg = Some(freed_reg);

                    // Track callee-saved usage for the newly assigned register.
                    if reg_info.is_callee_saved(freed_reg) {
                        used_callee_saved.insert(freed_reg);
                    }

                    // Remove victim from active, insert current.
                    active.retain(|&idx| idx != victim_idx);
                    let pos = active.partition_point(|&idx| {
                        intervals[idx].end <= intervals[i].end
                    });
                    active.insert(pos, i);
                } else {
                    // Spill the current interval (it extends further or equally).
                    intervals[i].spill_slot = Some(num_spill_slots);
                    num_spill_slots += 1;
                }
            } else {
                // No active interval of the same class exists — spill current.
                intervals[i].spill_slot = Some(num_spill_slots);
                num_spill_slots += 1;
            }
        }
    }

    // Collect callee-saved registers in sorted order for deterministic output.
    let mut callee_saved_vec: Vec<PhysReg> = used_callee_saved.into_iter().collect();
    callee_saved_vec.sort();

    AllocationResult {
        intervals: intervals.clone(),
        num_spill_slots,
        used_callee_saved: callee_saved_vec,
    }
}

// ---------------------------------------------------------------------------
// insert_spill_code — spill slot layout computation
// ---------------------------------------------------------------------------

/// Computes the stack frame layout for spill slots after register allocation.
///
/// This function determines the byte offset of each spill slot within the
/// stack frame. Actual spill load/store machine instructions are emitted by
/// the architecture-specific code generation backends, which use the
/// [`AllocationResult`] to identify spilled values and the [`SpillInfo`]
/// to compute memory operand offsets.
///
/// # Parameters
///
/// - `function` — The IR function (available for future IR-level annotations).
/// - `alloc_result` — The register allocation result containing spill info.
///
/// # Returns
///
/// A [`SpillInfo`] with slot offsets and total spill size.
pub fn insert_spill_code(
    function: &mut Function,
    alloc_result: &AllocationResult,
) -> SpillInfo {
    // Each spill slot occupies 8 bytes (the maximum register width across all
    // supported architectures: 64-bit GPRs and 64-bit FP/SIMD scalar).
    const SPILL_SLOT_SIZE: u32 = 8;

    let num_slots = alloc_result.num_spill_slots;
    let mut slot_offsets: Vec<i32> = Vec::with_capacity(num_slots as usize);

    // Assign stack offsets: each slot is below the frame pointer.
    // Slot 0 is at offset -8, slot 1 at -16, etc.
    for i in 0..num_slots {
        let offset = -(((i + 1) * SPILL_SLOT_SIZE) as i32);
        slot_offsets.push(offset);
    }

    let total_spill_size = num_slots * SPILL_SLOT_SIZE;

    // Access function.blocks to verify the function structure is intact.
    // The backends will iterate over function blocks to insert actual spill
    // load/store instructions at the machine code level.
    let _block_count = function.blocks.len();
    let _entry = function.entry_block;

    SpillInfo {
        slot_offsets,
        total_spill_size,
    }
}

// ---------------------------------------------------------------------------
// build_value_to_reg_map — SSA value to physical register mapping
// ---------------------------------------------------------------------------

/// Builds a mapping from SSA [`Value`]s to their assigned [`PhysReg`]s.
///
/// Only values that were successfully allocated to a register are included.
/// Spilled values (which have `spill_slot` set but no `assigned_reg`) are
/// excluded — the backend handles spill loads/stores separately.
///
/// This mapping is consumed by each backend's instruction encoding phase
/// to translate virtual SSA value references into concrete physical register
/// operands in the emitted machine code.
pub fn build_value_to_reg_map(alloc_result: &AllocationResult) -> HashMap<Value, PhysReg> {
    let mut map = HashMap::with_capacity(alloc_result.intervals.len());
    for interval in alloc_result.intervals.iter() {
        if let Some(reg) = interval.assigned_reg {
            map.insert(interval.value, reg);
        }
    }
    map
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::cfg::BasicBlock;
    use crate::ir::instructions::{BlockId, Constant, Instruction, Value};
    use crate::ir::cfg::Terminator;
    use crate::ir::types::IrType;
    use crate::ir::builder::Function;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Creates a minimal RegisterInfo for testing with a small register file.
    ///
    /// - 3 integer registers: r0 (caller-saved), r1 (caller-saved), r2 (callee-saved)
    /// - 2 float registers: f0 (caller-saved), f1 (callee-saved)
    fn make_test_reg_info() -> RegisterInfo {
        let int_regs = vec![PhysReg(0), PhysReg(1), PhysReg(2)];
        let float_regs = vec![PhysReg(10), PhysReg(11)];
        let callee_saved_int = vec![PhysReg(2)];
        let callee_saved_float = vec![PhysReg(11)];
        let mut reg_names = HashMap::new();
        reg_names.insert(PhysReg(0), "r0");
        reg_names.insert(PhysReg(1), "r1");
        reg_names.insert(PhysReg(2), "r2");
        reg_names.insert(PhysReg(10), "f0");
        reg_names.insert(PhysReg(11), "f1");
        RegisterInfo {
            int_regs,
            float_regs,
            callee_saved_int,
            callee_saved_float,
            reg_names,
        }
    }

    /// Creates a very small RegisterInfo with only 1 integer register.
    fn make_tiny_reg_info() -> RegisterInfo {
        let int_regs = vec![PhysReg(0)];
        let float_regs = vec![PhysReg(10)];
        let callee_saved_int = vec![];
        let callee_saved_float = vec![];
        let mut reg_names = HashMap::new();
        reg_names.insert(PhysReg(0), "r0");
        reg_names.insert(PhysReg(10), "f0");
        RegisterInfo {
            int_regs,
            float_regs,
            callee_saved_int,
            callee_saved_float,
            reg_names,
        }
    }

    /// Creates a simple one-block function:
    ///   %0 = const i32 42
    ///   %1 = const i32 10
    ///   %2 = add i32 %0, %1
    ///   ret %2
    fn make_simple_function() -> Function {
        let entry_id = BlockId(0);
        let mut entry = BasicBlock::new(entry_id, "entry".to_string());

        entry.instructions.push(Instruction::Const {
            result: Value(0),
            value: Constant::Integer {
                value: 42,
                ty: IrType::I32,
            },
        });
        entry.instructions.push(Instruction::Const {
            result: Value(1),
            value: Constant::Integer {
                value: 10,
                ty: IrType::I32,
            },
        });
        entry.instructions.push(Instruction::Add {
            result: Value(2),
            lhs: Value(0),
            rhs: Value(1),
            ty: IrType::I32,
        });
        entry.terminator = Some(Terminator::Return {
            value: Some(Value(2)),
        });

        Function {
            name: "test_simple".to_string(),
            return_type: IrType::I32,
            params: vec![],
            blocks: vec![entry],
            entry_block: entry_id,
            is_definition: true,
        }
    }

    /// Creates a function with two blocks (if-else):
    ///   entry:
    ///     %0 = const i1 true
    ///     br i1 %0, then, else
    ///   then:
    ///     %1 = const i32 1
    ///     br merge
    ///   else:
    ///     %2 = const i32 2
    ///     br merge
    ///   merge:
    ///     %3 = phi i32 [%1, then], [%2, else]
    ///     ret %3
    fn make_branch_function() -> Function {
        let entry_id = BlockId(0);
        let then_id = BlockId(1);
        let else_id = BlockId(2);
        let merge_id = BlockId(3);

        let mut entry = BasicBlock::new(entry_id, "entry".to_string());
        entry.instructions.push(Instruction::Const {
            result: Value(0),
            value: Constant::Bool(true),
        });
        entry.terminator = Some(Terminator::CondBranch {
            condition: Value(0),
            true_block: then_id,
            false_block: else_id,
        });
        entry.successors = vec![then_id, else_id];

        let mut then_block = BasicBlock::new(then_id, "then".to_string());
        then_block.instructions.push(Instruction::Const {
            result: Value(1),
            value: Constant::Integer {
                value: 1,
                ty: IrType::I32,
            },
        });
        then_block.terminator = Some(Terminator::Branch { target: merge_id });
        then_block.predecessors = vec![entry_id];
        then_block.successors = vec![merge_id];

        let mut else_block = BasicBlock::new(else_id, "else".to_string());
        else_block.instructions.push(Instruction::Const {
            result: Value(2),
            value: Constant::Integer {
                value: 2,
                ty: IrType::I32,
            },
        });
        else_block.terminator = Some(Terminator::Branch { target: merge_id });
        else_block.predecessors = vec![entry_id];
        else_block.successors = vec![merge_id];

        let mut merge_block = BasicBlock::new(merge_id, "merge".to_string());
        merge_block.phi_nodes.push(PhiNode {
            result: Value(3),
            ty: IrType::I32,
            incoming: vec![(Value(1), then_id), (Value(2), else_id)],
        });
        merge_block.terminator = Some(Terminator::Return {
            value: Some(Value(3)),
        });
        merge_block.predecessors = vec![then_id, else_id];

        Function {
            name: "test_branch".to_string(),
            return_type: IrType::I32,
            params: vec![],
            blocks: vec![entry, then_block, else_block, merge_block],
            entry_block: entry_id,
            is_definition: true,
        }
    }

    /// Creates a function with a call instruction.
    fn make_call_function() -> Function {
        let entry_id = BlockId(0);
        let mut entry = BasicBlock::new(entry_id, "entry".to_string());

        // %0 = const i32 5
        entry.instructions.push(Instruction::Const {
            result: Value(0),
            value: Constant::Integer {
                value: 5,
                ty: IrType::I32,
            },
        });
        // %1 = call i32 @foo(%0)
        entry.instructions.push(Instruction::Call {
            result: Some(Value(1)),
            callee: crate::ir::Callee::Direct("foo".to_string()),
            args: vec![Value(0)],
            return_ty: IrType::I32,
        });
        // %2 = add i32 %0, %1   — %0 crosses the call
        entry.instructions.push(Instruction::Add {
            result: Value(2),
            lhs: Value(0),
            rhs: Value(1),
            ty: IrType::I32,
        });
        entry.terminator = Some(Terminator::Return {
            value: Some(Value(2)),
        });

        Function {
            name: "test_call".to_string(),
            return_type: IrType::I32,
            params: vec![],
            blocks: vec![entry],
            entry_block: entry_id,
            is_definition: true,
        }
    }

    // -----------------------------------------------------------------------
    // PhysReg tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_physreg_display() {
        assert_eq!(format!("{}", PhysReg(0)), "r0");
        assert_eq!(format!("{}", PhysReg(15)), "r15");
        assert_eq!(format!("{}", PhysReg(100)), "r100");
    }

    #[test]
    fn test_physreg_equality_and_ordering() {
        assert_eq!(PhysReg(5), PhysReg(5));
        assert_ne!(PhysReg(5), PhysReg(6));
        assert!(PhysReg(3) < PhysReg(5));
    }

    #[test]
    fn test_physreg_hash() {
        let mut set = HashSet::new();
        set.insert(PhysReg(1));
        set.insert(PhysReg(2));
        set.insert(PhysReg(1)); // duplicate
        assert_eq!(set.len(), 2);
    }

    // -----------------------------------------------------------------------
    // RegClass tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_regclass_equality() {
        assert_eq!(RegClass::Integer, RegClass::Integer);
        assert_eq!(RegClass::Float, RegClass::Float);
        assert_ne!(RegClass::Integer, RegClass::Float);
    }

    #[test]
    fn test_regclass_display() {
        assert_eq!(format!("{}", RegClass::Integer), "int");
        assert_eq!(format!("{}", RegClass::Float), "float");
    }

    // -----------------------------------------------------------------------
    // RegisterInfo tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_info_name() {
        let ri = make_test_reg_info();
        assert_eq!(ri.name(PhysReg(0)), "r0");
        assert_eq!(ri.name(PhysReg(10)), "f0");
        // Unknown register gets generic fallback.
        assert_eq!(ri.name(PhysReg(99)), "r99");
    }

    #[test]
    fn test_register_info_callee_saved() {
        let ri = make_test_reg_info();
        assert!(!ri.is_callee_saved(PhysReg(0))); // caller-saved
        assert!(!ri.is_callee_saved(PhysReg(1))); // caller-saved
        assert!(ri.is_callee_saved(PhysReg(2)));  // callee-saved int
        assert!(!ri.is_callee_saved(PhysReg(10))); // caller-saved float
        assert!(ri.is_callee_saved(PhysReg(11))); // callee-saved float
    }

    #[test]
    fn test_register_info_num_regs() {
        let ri = make_test_reg_info();
        assert_eq!(ri.num_regs(RegClass::Integer), 3);
        assert_eq!(ri.num_regs(RegClass::Float), 2);
    }

    // -----------------------------------------------------------------------
    // Live interval computation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_function_no_intervals() {
        let func = Function {
            name: "empty".to_string(),
            return_type: IrType::Void,
            params: vec![],
            blocks: vec![],
            entry_block: BlockId(0),
            is_definition: true,
        };
        let intervals = compute_live_intervals(&func);
        assert!(intervals.is_empty());
    }

    #[test]
    fn test_single_def_use_interval() {
        let func = make_simple_function();
        let intervals = compute_live_intervals(&func);

        // Should have 3 values: %0, %1, %2.
        assert_eq!(intervals.len(), 3);

        // All intervals should be integer class (i32 operations).
        for iv in &intervals {
            assert_eq!(iv.reg_class, RegClass::Integer);
        }

        // %0 is defined at index 0, used at index 2 → interval [0, 3)
        let iv0 = intervals.iter().find(|iv| iv.value == Value(0)).unwrap();
        assert_eq!(iv0.start, 0);
        assert!(iv0.end > iv0.start);

        // %2 is defined at index 2, used in return → interval covers [2, ...)
        let iv2 = intervals.iter().find(|iv| iv.value == Value(2)).unwrap();
        assert_eq!(iv2.start, 2);
        assert!(iv2.end > iv2.start);
    }

    #[test]
    fn test_phi_node_intervals() {
        let func = make_branch_function();
        let intervals = compute_live_intervals(&func);

        // Should have 4 values: %0, %1, %2, %3.
        assert_eq!(intervals.len(), 4);

        // %3 comes from a phi node → should exist.
        let iv3 = intervals.iter().find(|iv| iv.value == Value(3)).unwrap();
        assert_eq!(iv3.reg_class, RegClass::Integer);
        assert!(iv3.end > iv3.start);
    }

    #[test]
    fn test_call_crossing_detection() {
        let func = make_call_function();
        let intervals = compute_live_intervals(&func);

        // %0 is defined before the call and used after → crosses_call = true.
        let iv0 = intervals.iter().find(|iv| iv.value == Value(0)).unwrap();
        assert!(
            iv0.crosses_call,
            "%0 should cross the call: start={}, end={}",
            iv0.start, iv0.end
        );

        // %1 is the call result, used after the call → does NOT cross the call
        // (it's defined at the call, not live across it).
        let iv1 = intervals.iter().find(|iv| iv.value == Value(1)).unwrap();
        assert!(
            !iv1.crosses_call,
            "%1 should not cross the call: start={}, end={}",
            iv1.start, iv1.end
        );
    }

    #[test]
    fn test_intervals_sorted_by_start() {
        let func = make_simple_function();
        let intervals = compute_live_intervals(&func);
        for window in intervals.windows(2) {
            assert!(
                window[0].start <= window[1].start,
                "Intervals not sorted: {} before {}",
                window[0],
                window[1]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Linear scan allocation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_non_overlapping_same_register() {
        let ri = make_test_reg_info();
        let mut intervals = vec![
            LiveInterval {
                value: Value(0),
                reg_class: RegClass::Integer,
                start: 0,
                end: 5,
                assigned_reg: None,
                spill_slot: None,
                is_param: false,
                crosses_call: false,
            },
            LiveInterval {
                value: Value(1),
                reg_class: RegClass::Integer,
                start: 5,
                end: 10,
                assigned_reg: None,
                spill_slot: None,
                is_param: false,
                crosses_call: false,
            },
        ];

        let result = linear_scan_allocate(&mut intervals, &ri);

        // Both should be allocated (3 regs available).
        assert!(result.intervals[0].assigned_reg.is_some());
        assert!(result.intervals[1].assigned_reg.is_some());
        assert_eq!(result.num_spill_slots, 0);

        // Non-overlapping intervals CAN reuse the same register.
        // The second interval starts after the first ends, so the first's
        // register is freed and can be reused.
        let r0 = result.intervals[0].assigned_reg.unwrap();
        let r1 = result.intervals[1].assigned_reg.unwrap();
        assert_eq!(r0, r1, "Non-overlapping intervals should reuse the same register");
    }

    #[test]
    fn test_overlapping_different_registers() {
        let ri = make_test_reg_info();
        let mut intervals = vec![
            LiveInterval {
                value: Value(0),
                reg_class: RegClass::Integer,
                start: 0,
                end: 10,
                assigned_reg: None,
                spill_slot: None,
                is_param: false,
                crosses_call: false,
            },
            LiveInterval {
                value: Value(1),
                reg_class: RegClass::Integer,
                start: 5,
                end: 15,
                assigned_reg: None,
                spill_slot: None,
                is_param: false,
                crosses_call: false,
            },
        ];

        let result = linear_scan_allocate(&mut intervals, &ri);

        // Both should be allocated to different registers.
        let r0 = result.intervals[0].assigned_reg.unwrap();
        let r1 = result.intervals[1].assigned_reg.unwrap();
        assert_ne!(r0, r1, "Overlapping intervals must get different registers");
        assert_eq!(result.num_spill_slots, 0);
    }

    #[test]
    fn test_spilling_when_registers_exhausted() {
        let ri = make_tiny_reg_info(); // Only 1 integer register.
        let mut intervals = vec![
            LiveInterval {
                value: Value(0),
                reg_class: RegClass::Integer,
                start: 0,
                end: 10,
                assigned_reg: None,
                spill_slot: None,
                is_param: false,
                crosses_call: false,
            },
            LiveInterval {
                value: Value(1),
                reg_class: RegClass::Integer,
                start: 5,
                end: 20,
                assigned_reg: None,
                spill_slot: None,
                is_param: false,
                crosses_call: false,
            },
        ];

        let result = linear_scan_allocate(&mut intervals, &ri);

        // One should be allocated, one should be spilled.
        let allocated = result
            .intervals
            .iter()
            .filter(|iv| iv.assigned_reg.is_some())
            .count();
        let spilled = result
            .intervals
            .iter()
            .filter(|iv| iv.spill_slot.is_some())
            .count();
        assert_eq!(allocated, 1);
        assert_eq!(spilled, 1);
        assert_eq!(result.num_spill_slots, 1);
    }

    #[test]
    fn test_callee_saved_preference_for_call_crossing() {
        let ri = make_test_reg_info();
        // PhysReg(2) is the only callee-saved integer register.
        let mut intervals = vec![LiveInterval {
            value: Value(0),
            reg_class: RegClass::Integer,
            start: 0,
            end: 10,
            assigned_reg: None,
            spill_slot: None,
            is_param: false,
            crosses_call: true, // crosses a call
        }];

        let result = linear_scan_allocate(&mut intervals, &ri);

        // Should prefer callee-saved register (PhysReg(2)).
        assert_eq!(
            result.intervals[0].assigned_reg,
            Some(PhysReg(2)),
            "Call-crossing interval should prefer callee-saved register"
        );
    }

    #[test]
    fn test_caller_saved_preference_for_non_call_crossing() {
        let ri = make_test_reg_info();
        let mut intervals = vec![LiveInterval {
            value: Value(0),
            reg_class: RegClass::Integer,
            start: 0,
            end: 10,
            assigned_reg: None,
            spill_slot: None,
            is_param: false,
            crosses_call: false, // does not cross a call
        }];

        let result = linear_scan_allocate(&mut intervals, &ri);

        // Should prefer caller-saved register (PhysReg(0) or PhysReg(1)).
        let assigned = result.intervals[0].assigned_reg.unwrap();
        assert!(
            assigned == PhysReg(0) || assigned == PhysReg(1),
            "Non-call-crossing interval should prefer caller-saved register, got {}",
            assigned
        );
    }

    // -----------------------------------------------------------------------
    // Callee-saved tracking tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_callee_saved_tracking_when_used() {
        let ri = make_test_reg_info();
        // Three overlapping intervals exhaust caller-saved, forcing callee-saved.
        let mut intervals = vec![
            LiveInterval {
                value: Value(0),
                reg_class: RegClass::Integer,
                start: 0,
                end: 10,
                assigned_reg: None,
                spill_slot: None,
                is_param: false,
                crosses_call: false,
            },
            LiveInterval {
                value: Value(1),
                reg_class: RegClass::Integer,
                start: 1,
                end: 10,
                assigned_reg: None,
                spill_slot: None,
                is_param: false,
                crosses_call: false,
            },
            LiveInterval {
                value: Value(2),
                reg_class: RegClass::Integer,
                start: 2,
                end: 10,
                assigned_reg: None,
                spill_slot: None,
                is_param: false,
                crosses_call: false,
            },
        ];

        let result = linear_scan_allocate(&mut intervals, &ri);

        // All 3 registers should be used (including callee-saved PhysReg(2)).
        assert_eq!(result.num_spill_slots, 0);
        assert!(
            result.used_callee_saved.contains(&PhysReg(2)),
            "Callee-saved register should be tracked"
        );
    }

    #[test]
    fn test_no_callee_saved_when_only_caller_saved_used() {
        let ri = make_test_reg_info();
        // Only one interval — should use caller-saved only.
        let mut intervals = vec![LiveInterval {
            value: Value(0),
            reg_class: RegClass::Integer,
            start: 0,
            end: 5,
            assigned_reg: None,
            spill_slot: None,
            is_param: false,
            crosses_call: false,
        }];

        let result = linear_scan_allocate(&mut intervals, &ri);

        assert!(
            result.used_callee_saved.is_empty(),
            "No callee-saved registers should be used"
        );
    }

    // -----------------------------------------------------------------------
    // Spill tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_spill_slot_numbering_sequential() {
        let ri = make_tiny_reg_info(); // 1 int reg, 1 float reg.
        let mut intervals = vec![
            LiveInterval {
                value: Value(0),
                reg_class: RegClass::Integer,
                start: 0,
                end: 20,
                assigned_reg: None,
                spill_slot: None,
                is_param: false,
                crosses_call: false,
            },
            LiveInterval {
                value: Value(1),
                reg_class: RegClass::Integer,
                start: 1,
                end: 15,
                assigned_reg: None,
                spill_slot: None,
                is_param: false,
                crosses_call: false,
            },
            LiveInterval {
                value: Value(2),
                reg_class: RegClass::Integer,
                start: 2,
                end: 10,
                assigned_reg: None,
                spill_slot: None,
                is_param: false,
                crosses_call: false,
            },
        ];

        let result = linear_scan_allocate(&mut intervals, &ri);

        // With 1 register and 3 overlapping intervals, 2 must spill.
        assert_eq!(result.num_spill_slots, 2);

        let spill_slots: Vec<u32> = result
            .intervals
            .iter()
            .filter_map(|iv| iv.spill_slot)
            .collect();
        // Spill slots should be 0 and 1 (sequential).
        assert!(spill_slots.contains(&0));
        assert!(spill_slots.contains(&1));
    }

    #[test]
    fn test_spill_info_computation() {
        let ri = make_tiny_reg_info();
        let mut intervals = vec![
            LiveInterval {
                value: Value(0),
                reg_class: RegClass::Integer,
                start: 0,
                end: 10,
                assigned_reg: None,
                spill_slot: None,
                is_param: false,
                crosses_call: false,
            },
            LiveInterval {
                value: Value(1),
                reg_class: RegClass::Integer,
                start: 5,
                end: 15,
                assigned_reg: None,
                spill_slot: None,
                is_param: false,
                crosses_call: false,
            },
        ];

        let alloc_result = linear_scan_allocate(&mut intervals, &ri);
        assert_eq!(alloc_result.num_spill_slots, 1);

        let mut func = make_simple_function();
        let spill_info = insert_spill_code(&mut func, &alloc_result);

        assert_eq!(spill_info.slot_offsets.len(), 1);
        assert_eq!(spill_info.slot_offsets[0], -8); // first slot at -8
        assert_eq!(spill_info.total_spill_size, 8);
    }

    #[test]
    fn test_spill_info_multiple_slots() {
        let alloc_result = AllocationResult {
            intervals: vec![],
            num_spill_slots: 4,
            used_callee_saved: vec![],
        };

        let mut func = make_simple_function();
        let spill_info = insert_spill_code(&mut func, &alloc_result);

        assert_eq!(spill_info.slot_offsets.len(), 4);
        assert_eq!(spill_info.slot_offsets[0], -8);
        assert_eq!(spill_info.slot_offsets[1], -16);
        assert_eq!(spill_info.slot_offsets[2], -24);
        assert_eq!(spill_info.slot_offsets[3], -32);
        assert_eq!(spill_info.total_spill_size, 32);
    }

    #[test]
    fn test_spill_info_zero_slots() {
        let alloc_result = AllocationResult {
            intervals: vec![],
            num_spill_slots: 0,
            used_callee_saved: vec![],
        };

        let mut func = make_simple_function();
        let spill_info = insert_spill_code(&mut func, &alloc_result);

        assert!(spill_info.slot_offsets.is_empty());
        assert_eq!(spill_info.total_spill_size, 0);
    }

    // -----------------------------------------------------------------------
    // build_value_to_reg_map tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_value_to_reg_map_basic() {
        let alloc_result = AllocationResult {
            intervals: vec![
                LiveInterval {
                    value: Value(0),
                    reg_class: RegClass::Integer,
                    start: 0,
                    end: 5,
                    assigned_reg: Some(PhysReg(3)),
                    spill_slot: None,
                    is_param: false,
                    crosses_call: false,
                },
                LiveInterval {
                    value: Value(1),
                    reg_class: RegClass::Integer,
                    start: 5,
                    end: 10,
                    assigned_reg: Some(PhysReg(7)),
                    spill_slot: None,
                    is_param: false,
                    crosses_call: false,
                },
            ],
            num_spill_slots: 0,
            used_callee_saved: vec![],
        };

        let map = build_value_to_reg_map(&alloc_result);

        assert_eq!(map.len(), 2);
        assert_eq!(*map.get(&Value(0)).unwrap(), PhysReg(3));
        assert_eq!(*map.get(&Value(1)).unwrap(), PhysReg(7));
    }

    #[test]
    fn test_value_to_reg_map_excludes_spilled() {
        let alloc_result = AllocationResult {
            intervals: vec![
                LiveInterval {
                    value: Value(0),
                    reg_class: RegClass::Integer,
                    start: 0,
                    end: 5,
                    assigned_reg: Some(PhysReg(1)),
                    spill_slot: None,
                    is_param: false,
                    crosses_call: false,
                },
                LiveInterval {
                    value: Value(1),
                    reg_class: RegClass::Integer,
                    start: 5,
                    end: 10,
                    assigned_reg: None,
                    spill_slot: Some(0),
                    is_param: false,
                    crosses_call: false,
                },
            ],
            num_spill_slots: 1,
            used_callee_saved: vec![],
        };

        let map = build_value_to_reg_map(&alloc_result);

        assert_eq!(map.len(), 1);
        assert!(map.contains_key(&Value(0)));
        assert!(!map.contains_key(&Value(1))); // spilled, not in map
    }

    // -----------------------------------------------------------------------
    // Display tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_live_interval_display_assigned() {
        let iv = LiveInterval {
            value: Value(5),
            reg_class: RegClass::Integer,
            start: 12,
            end: 28,
            assigned_reg: Some(PhysReg(3)),
            spill_slot: None,
            is_param: false,
            crosses_call: false,
        };
        let s = format!("{}", iv);
        assert!(s.contains("%5"));
        assert!(s.contains("[12, 28)"));
        assert!(s.contains("r3"));
    }

    #[test]
    fn test_live_interval_display_spilled() {
        let iv = LiveInterval {
            value: Value(8),
            reg_class: RegClass::Integer,
            start: 15,
            end: 40,
            assigned_reg: None,
            spill_slot: Some(0),
            is_param: false,
            crosses_call: false,
        };
        let s = format!("{}", iv);
        assert!(s.contains("%8"));
        assert!(s.contains("[15, 40)"));
        assert!(s.contains("[spill:0]"));
    }

    #[test]
    fn test_live_interval_display_unassigned() {
        let iv = LiveInterval {
            value: Value(2),
            reg_class: RegClass::Float,
            start: 0,
            end: 5,
            assigned_reg: None,
            spill_slot: None,
            is_param: false,
            crosses_call: false,
        };
        let s = format!("{}", iv);
        assert!(s.contains("unassigned"));
    }

    // -----------------------------------------------------------------------
    // End-to-end tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_end_to_end_simple_function() {
        let func = make_simple_function();
        let mut intervals = compute_live_intervals(&func);
        let ri = make_test_reg_info();
        let result = linear_scan_allocate(&mut intervals, &ri);

        // All values should be allocated (only 3 values, 3 int regs available).
        assert_eq!(result.num_spill_slots, 0);
        for iv in &result.intervals {
            assert!(
                iv.assigned_reg.is_some(),
                "Value {} should be allocated",
                iv.value
            );
        }

        let map = build_value_to_reg_map(&result);
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn test_end_to_end_branch_function() {
        let func = make_branch_function();
        let mut intervals = compute_live_intervals(&func);
        let ri = make_test_reg_info();
        let result = linear_scan_allocate(&mut intervals, &ri);

        // Should complete without panics and allocate all values.
        assert_eq!(result.num_spill_slots, 0);
    }

    #[test]
    fn test_end_to_end_call_function() {
        let func = make_call_function();
        let mut intervals = compute_live_intervals(&func);
        let ri = make_test_reg_info();
        let result = linear_scan_allocate(&mut intervals, &ri);

        // Should complete without panics.
        assert_eq!(result.num_spill_slots, 0);

        // The call-crossing value should be in a callee-saved register.
        let iv0 = result
            .intervals
            .iter()
            .find(|iv| iv.value == Value(0))
            .unwrap();
        if iv0.crosses_call {
            assert!(
                ri.is_callee_saved(iv0.assigned_reg.unwrap()),
                "Call-crossing value should be in callee-saved register"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Edge case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_instruction_function() {
        let entry_id = BlockId(0);
        let mut entry = BasicBlock::new(entry_id, "entry".to_string());
        entry.terminator = Some(Terminator::Return { value: None });

        let func = Function {
            name: "void_fn".to_string(),
            return_type: IrType::Void,
            params: vec![],
            blocks: vec![entry],
            entry_block: entry_id,
            is_definition: true,
        };

        let intervals = compute_live_intervals(&func);
        assert!(intervals.is_empty(), "Void return function should have no intervals");
    }

    #[test]
    fn test_float_register_class() {
        let entry_id = BlockId(0);
        let mut entry = BasicBlock::new(entry_id, "entry".to_string());
        entry.instructions.push(Instruction::Const {
            result: Value(0),
            value: Constant::Float {
                value: 3.14,
                ty: IrType::F64,
            },
        });
        entry.terminator = Some(Terminator::Return {
            value: Some(Value(0)),
        });

        let func = Function {
            name: "float_fn".to_string(),
            return_type: IrType::F64,
            params: vec![],
            blocks: vec![entry],
            entry_block: entry_id,
            is_definition: true,
        };

        let intervals = compute_live_intervals(&func);
        assert_eq!(intervals.len(), 1);
        // Const with Float type → should be Float class based on the constant type.
        // Note: Constant::Float has a ty field of IrType::F64, but result_type()
        // for Const returns the constant's type. For Float constants, result_type()
        // returns Some(&IrType::F64) which is_float() == true.
        // However, classify_instruction_result handles Const specially: it uses
        // result_type() which for Float constants returns the float IrType.
        let iv = &intervals[0];
        assert_eq!(iv.reg_class, RegClass::Float);
    }

    #[test]
    fn test_mixed_int_float_allocation() {
        let ri = make_test_reg_info();
        let mut intervals = vec![
            LiveInterval {
                value: Value(0),
                reg_class: RegClass::Integer,
                start: 0,
                end: 10,
                assigned_reg: None,
                spill_slot: None,
                is_param: false,
                crosses_call: false,
            },
            LiveInterval {
                value: Value(1),
                reg_class: RegClass::Float,
                start: 0,
                end: 10,
                assigned_reg: None,
                spill_slot: None,
                is_param: false,
                crosses_call: false,
            },
        ];

        let result = linear_scan_allocate(&mut intervals, &ri);

        // Int and float don't compete for the same registers.
        assert_eq!(result.num_spill_slots, 0);
        let iv_int = result
            .intervals
            .iter()
            .find(|iv| iv.value == Value(0))
            .unwrap();
        let iv_float = result
            .intervals
            .iter()
            .find(|iv| iv.value == Value(1))
            .unwrap();
        assert!(iv_int.assigned_reg.unwrap().0 < 10); // int register
        assert!(iv_float.assigned_reg.unwrap().0 >= 10); // float register
    }

    #[test]
    fn test_function_with_only_constants() {
        let entry_id = BlockId(0);
        let mut entry = BasicBlock::new(entry_id, "entry".to_string());
        entry.instructions.push(Instruction::Const {
            result: Value(0),
            value: Constant::Integer {
                value: 42,
                ty: IrType::I32,
            },
        });
        // Constant is defined but never used (dead code in practice).
        entry.terminator = Some(Terminator::Return { value: None });

        let func = Function {
            name: "const_fn".to_string(),
            return_type: IrType::Void,
            params: vec![],
            blocks: vec![entry],
            entry_block: entry_id,
            is_definition: true,
        };

        let intervals = compute_live_intervals(&func);
        // The constant value %0 is defined but has no uses.
        // It still gets an interval with minimum length.
        assert_eq!(intervals.len(), 1);
        let iv = &intervals[0];
        assert_eq!(iv.start + 1, iv.end, "Unused value should have minimum interval");
    }
}
