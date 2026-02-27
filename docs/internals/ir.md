# Intermediate Representation

The **bcc** compiler uses a target-independent, SSA-form (Static Single Assignment) intermediate representation as the central bridge between the C11 frontend and the four architecture-specific backends (x86-64, i686, AArch64, RISC-V 64). The IR is designed to be rich enough to faithfully express all C11 semantics — including GCC extensions — while remaining sufficiently abstract that optimization passes and code generators can operate without knowledge of source-level constructs. Every value in the IR is defined exactly once, enabling efficient dataflow analysis and a clean separation between analysis and transformation.

## Design Principles

The IR is governed by five core design principles:

1. **SSA Form** — Every IR value is defined by exactly one instruction. When a variable may receive different values along different control-flow paths, a `Phi` node merges those definitions at the join point. This single-definition property enables efficient use-def chain traversal, constant propagation, dead code elimination, and common subexpression elimination without expensive reaching-definition analysis.

2. **Target Independence** — IR operations abstract over machine-specific details such as register counts, instruction encodings, and addressing modes. Target-specific parameters (pointer width, `long` size, struct alignment rules) are supplied through a `TargetConfig` value that the IR builder consults during construction. The IR itself never references a specific architecture.

3. **Typed Values** — Every IR value carries a type drawn from the IR type system (see [IR Type System](#ir-type-system)). Types enable the IR to validate transformations at construction time, ensure that instruction operands are compatible, and communicate operand sizes to downstream code generators.

4. **Explicit Control Flow** — Control flow is represented as a graph of basic blocks connected by explicit terminator instructions (`Br`, `CondBr`, `Switch`, `Ret`). There is no implicit fall-through between blocks. Every block must end with exactly one terminator, making control-flow analysis straightforward.

5. **Memory-Explicit Operations** — All memory accesses are represented as explicit `Load`, `Store`, and `Alloca` instructions. There is no implicit memory access hidden inside arithmetic or comparison instructions. This explicitness allows the `mem2reg` pass to promote stack allocations to SSA registers in a clean, well-defined manner.

## Source Module Reference

The IR implementation resides in `src/ir/` and is organized into the following source files:

| File | Responsibility |
|---|---|
| `src/ir/mod.rs` | Module re-exports; defines the public IR API surface |
| `src/ir/types.rs` | IR type system — integer, float, pointer, aggregate, void, function types |
| `src/ir/instructions.rs` | IR instruction set — all 24 instruction opcodes with operand definitions |
| `src/ir/builder.rs` | IR builder — translates the typed AST from semantic analysis into IR |
| `src/ir/cfg.rs` | Control flow graph — basic blocks, edges, dominance tree, dominance frontier |
| `src/ir/ssa.rs` | SSA construction — iterated dominance frontier phi placement and variable renaming |

## Pipeline Position

The IR occupies the central position in the compilation pipeline:

```
Source Code
    │
    ▼
┌──────────────┐
│ Preprocessor │   src/frontend/preprocessor/
├──────────────┤
│    Lexer     │   src/frontend/lexer/
├──────────────┤
│    Parser    │   src/frontend/parser/
└──────┬───────┘
       │  Untyped AST
       ▼
┌──────────────┐
│   Semantic   │   src/sema/
│   Analysis   │
└──────┬───────┘
       │  Typed AST + Symbol Table
       ▼
┌──────────────┐
│  IR Builder  │   src/ir/builder.rs         ◄── AST → IR translation
├──────────────┤
│  SSA Constr. │   src/ir/ssa.rs             ◄── Phi node insertion + renaming
└──────┬───────┘
       │  SSA-form IR
       ▼
┌──────────────┐
│ Optimization │   src/passes/               ◄── IR → IR transformations
│    Passes    │
└──────┬───────┘
       │  Optimized IR
       ▼
┌──────────────┐
│    Code      │   src/codegen/{x86_64,i686,aarch64,riscv64}/
│  Generation  │
└──────────────┘
```

The IR builder (`src/ir/builder.rs`) consumes the typed AST produced by semantic analysis (`src/sema/`) and emits IR instructions organized into basic blocks within functions. After initial construction, SSA construction (`src/ir/ssa.rs`) inserts phi nodes and renames variables. Optimization passes (`src/passes/`) then transform the IR in place. Finally, architecture-specific code generators (`src/codegen/`) lower the optimized IR to machine code.

---

## IR Type System

> **Source file:** `src/ir/types.rs`

The IR type system provides a minimal but complete set of types sufficient to represent all C11 data types across all four target architectures. Types are target-independent in their definition — but target-dependent in their size and alignment, which are resolved by querying the `TargetConfig`.

### Integer Types

| IR Type | Bit Width | C Types Mapped |
|---|---|---|
| `i1` | 1 bit | `_Bool`, comparison results (ICmp/FCmp output) |
| `i8` | 8 bits | `char`, `signed char`, `unsigned char` |
| `i16` | 16 bits | `short`, `unsigned short` |
| `i32` | 32 bits | `int`, `unsigned int`, `long` (i686 only) |
| `i64` | 64 bits | `long long`, `unsigned long long`, `long` (x86-64, AArch64, RISC-V 64) |

> **Note on `long`:** The C `long` type maps to `i32` on the i686 target (where pointers are 32 bits) and to `i64` on all 64-bit targets (x86-64, AArch64, RISC-V 64). This mapping is resolved at IR construction time by the IR builder, which consults `TargetConfig::long_width()`. The IR itself does not contain a "long" type — only concrete fixed-width integer types.

Integer types do not encode signedness. Instructions that behave differently for signed versus unsigned operands (such as `Div`, `Mod`, `Shr`, and comparison predicates) carry an explicit signed/unsigned flag or use distinct predicates. This design avoids type proliferation while preserving full C semantics.

### Floating-Point Types

| IR Type | Precision | C Types Mapped |
|---|---|---|
| `f32` | IEEE 754 binary32 (single precision) | `float` |
| `f64` | IEEE 754 binary64 (double precision) | `double`, `long double` (simplified) |

> **Note on `long double`:** The bcc compiler maps `long double` to `f64` (IEEE 754 double precision) on all targets. This is a simplification — x86-64 hardware supports 80-bit extended precision, and AArch64 supports 128-bit quad precision. The `f64` mapping is sufficient for the vast majority of real-world C code, including SQLite, Lua, zlib, and Redis.

### Pointer Type

```
Ptr(pointee_type: IrType)
```

- **Definition:** A typed pointer to any IR type. The pointee type is used by `GetElementPtr` for address arithmetic and by `Load`/`Store` for determining the accessed data width.
- **Size:** Target-dependent — 4 bytes (32 bits) on i686, 8 bytes (64 bits) on x86-64, AArch64, and RISC-V 64. Size is obtained from `TargetConfig::pointer_width()`.
- **Usage:** All C pointer types, array-to-pointer decay, function pointers, and `void*` (represented as `Ptr(i8)` by convention).

### Aggregate Types

#### Struct

```
Struct(fields: Vec<IrType>)
```

- An ordered sequence of field types. Field layout (offsets, padding, total size) is computed per the target ABI's alignment rules using `TargetConfig`.
- Structs are structurally typed — two `Struct` types with identical field lists are the same type.
- Field access is performed via `GetElementPtr` with an index identifying the field position.

#### Array

```
Array(element_type: IrType, count: u64)
```

- A fixed-size array of `count` elements, each of type `element_type`.
- Total size = `size_of(element_type) × count`, with alignment equal to the element's alignment.
- Element access is performed via `GetElementPtr` with an index identifying the element position.

Aggregate types are the basis for `GetElementPtr` (GEP) instructions, which compute addresses within structs and arrays without performing memory access.

### Special Types

#### Void

```
Void
```

- Used as the return type of void functions and as the result type of statements that produce no value (e.g., `Store`, `Ret` with no operand).
- `Void` cannot be used as an instruction operand or as a variable type.

#### Function

```
Function(return_type: IrType, param_types: Vec<IrType>)
```

- Represents a function signature. Used for type-checking `Call` instructions — the callee's function type must match the provided arguments and expected return type.
- Function types are never used as value types directly; instead, functions are referenced through `Ptr(Function(...))` when used as function pointers.

### Type Size and Alignment Rules

| Property | Rule |
|---|---|
| Integer sizes | `i1` = 1 byte (stored), `i8` = 1 byte, `i16` = 2 bytes, `i32` = 4 bytes, `i64` = 8 bytes |
| Float sizes | `f32` = 4 bytes, `f64` = 8 bytes |
| Pointer size | `TargetConfig::pointer_width()` — 4 bytes (i686) or 8 bytes (64-bit targets) |
| Struct layout | Fields laid out sequentially with padding inserted to satisfy each field's alignment; total size rounded up to the struct's alignment (max of all field alignments) |
| Array size | `size_of(element) × count`; alignment = `align_of(element)` |
| Alignment | Natural alignment (equal to size) for scalars; target ABI rules for aggregates |

---

## IR Instruction Set

> **Source file:** `src/ir/instructions.rs`

The IR defines 24 instruction opcodes organized into five categories: arithmetic, bitwise, comparison, memory, control flow, and miscellaneous. Each instruction produces zero or one result value, takes typed operands, and is assigned to a basic block.

### Arithmetic Instructions

Arithmetic instructions operate on integer or floating-point values. Both operands must have the same type, and the result has the same type as the operands.

| Instruction | Operands | Result Type | Description |
|---|---|---|---|
| `Add(lhs, rhs)` | Two integer or float values | Same as operands | Integer or floating-point addition |
| `Sub(lhs, rhs)` | Two integer or float values | Same as operands | Integer or floating-point subtraction |
| `Mul(lhs, rhs)` | Two integer or float values | Same as operands | Integer or floating-point multiplication |
| `Div(lhs, rhs)` | Two integer or float values | Same as operands | Division — signed for integers by default; carries a `signed` flag for unsigned division (`UDiv`) |
| `Mod(lhs, rhs)` | Two integer values | Same as operands | Integer remainder — signed by default; carries a `signed` flag for unsigned remainder (`URem`) |

> **Signed vs. unsigned:** The `Div` and `Mod` instructions carry an explicit `is_signed: bool` flag to distinguish signed division/remainder from unsigned. This is more compact than defining separate `SDiv`/`UDiv`/`SRem`/`URem` opcodes while preserving full semantic clarity. `Add`, `Sub`, and `Mul` produce identical bit patterns for signed and unsigned operands (two's complement), so no flag is needed.

### Bitwise Instructions

Bitwise instructions operate exclusively on integer types. Both operands must have the same type for `And`/`Or`/`Xor`; for shifts, the value and shift amount must both be integers (the shift amount may be a different width).

| Instruction | Operands | Result Type | Description |
|---|---|---|---|
| `And(lhs, rhs)` | Two integer values of the same type | Same as operands | Bitwise AND |
| `Or(lhs, rhs)` | Two integer values of the same type | Same as operands | Bitwise OR |
| `Xor(lhs, rhs)` | Two integer values of the same type | Same as operands | Bitwise XOR |
| `Shl(value, amount)` | Integer value + integer shift amount | Same as `value` | Left shift — zeros shifted in from the right |
| `Shr(value, amount)` | Integer value + integer shift amount | Same as `value` | Right shift — carries an `is_arithmetic: bool` flag: `true` for arithmetic shift (sign bit preserved), `false` for logical shift (zeros shifted in) |

> **Shift semantics:** `Shr` uses an `is_arithmetic` flag rather than separate `AShr`/`LShr` opcodes. The IR builder sets this flag based on the signedness of the C source operand. Shift amounts exceeding the bit width produce undefined behavior (matching C semantics).

### Comparison Instructions

Comparison instructions compare two values and produce an `i1` (boolean) result.

| Instruction | Operands | Result Type | Description |
|---|---|---|---|
| `ICmp(predicate, lhs, rhs)` | Comparison predicate + two integer values | `i1` | Integer comparison |
| `FCmp(predicate, lhs, rhs)` | Comparison predicate + two float values | `i1` | Floating-point comparison |

#### Integer Comparison Predicates (`ICmpPredicate`)

| Predicate | Meaning | C Equivalent |
|---|---|---|
| `Eq` | Equal | `==` |
| `Ne` | Not equal | `!=` |
| `Slt` | Signed less than | `<` (signed) |
| `Sle` | Signed less or equal | `<=` (signed) |
| `Sgt` | Signed greater than | `>` (signed) |
| `Sge` | Signed greater or equal | `>=` (signed) |
| `Ult` | Unsigned less than | `<` (unsigned) |
| `Ule` | Unsigned less or equal | `<=` (unsigned) |
| `Ugt` | Unsigned greater than | `>` (unsigned) |
| `Uge` | Unsigned greater or equal | `>=` (unsigned) |

#### Floating-Point Comparison Predicates (`FCmpPredicate`)

| Predicate | Meaning |
|---|---|
| `OEq` | Ordered and equal — both operands are not NaN, and they are equal |
| `ONe` | Ordered and not equal |
| `OLt` | Ordered and less than |
| `OLe` | Ordered and less or equal |
| `OGt` | Ordered and greater than |
| `OGe` | Ordered and greater or equal |
| `UEq` | Unordered or equal — true if either operand is NaN, or they are equal |
| `UNe` | Unordered or not equal |
| `ULt` | Unordered or less than |
| `ULe` | Unordered or less or equal |
| `UGt` | Unordered or greater than |
| `UGe` | Unordered or greater or equal |
| `Ord` | Ordered — true if neither operand is NaN |
| `Uno` | Unordered — true if either operand is NaN |

> **Ordered vs. unordered:** IEEE 754 defines that comparisons involving NaN are unordered. Ordered predicates (`O*`) return false when either operand is NaN. Unordered predicates (`U*`) return true when either operand is NaN. The `Ord` and `Uno` predicates test the ordering property directly.

### Memory Instructions

Memory instructions perform explicit interactions with the address space — allocating stack memory, reading from memory, writing to memory, and computing addresses within aggregate data structures.

| Instruction | Operands | Result Type | Description |
|---|---|---|---|
| `Alloca(type, count)` | Element type + optional element count (default 1) | `Ptr(type)` | Allocate space on the stack frame for `count` elements of `type`; returns a pointer to the allocation |
| `Load(pointer)` | Pointer value of type `Ptr(T)` | `T` (pointee type) | Read the value of type `T` from the address held in `pointer` |
| `Store(value, pointer)` | Value of type `T` + pointer of type `Ptr(T)` | `Void` | Write `value` to the memory location addressed by `pointer` |
| `GetElementPtr(base, indices)` | Base pointer + list of integer index values | Pointer type | Compute the address of a sub-element within an aggregate or array |

#### GetElementPtr (GEP) Semantics

The `GetElementPtr` instruction is the IR's mechanism for structured address computation. It takes a base pointer and a sequence of indices, and computes a new pointer without accessing memory.

**Index interpretation:**

- **First index:** Offsets the base pointer as if it pointed to an array. `GEP %ptr, 3` computes `%ptr + 3 * sizeof(pointee_type)`.
- **Subsequent indices:** Navigate into aggregate types. For a `Struct`, the index selects a field by position. For an `Array`, the index selects an element.

**Key properties:**

- GEP **only computes addresses** — it does not load or store any data.
- GEP is the canonical way to access struct fields and array elements.
- GEP results are pointers — use `Load` to read the value at the computed address.

**Example:**

```
; Given: %s = Ptr(Struct(i32, i64, Ptr(i8)))
; Access field 2 (the Ptr(i8) field):
%field_ptr = GetElementPtr %s, 0, 2      ; → Ptr(Ptr(i8))
%field_val = Load %field_ptr             ; → Ptr(i8)
```

The first index `0` selects the struct at `%s` itself (no array offset); the second index `2` selects the third field of the struct.

### Control Flow Instructions (Terminators)

Terminator instructions end a basic block and define the control flow edges to successor blocks. Every basic block must end with exactly one terminator. There is no implicit fall-through.

| Instruction | Operands | Description |
|---|---|---|
| `Ret(value?)` | Optional return value (type matches function return type) | Return from the current function. If the function returns `Void`, the value is omitted. |
| `Br(target)` | Target basic block label | Unconditional branch — transfers control to `target` |
| `CondBr(cond, true_bb, false_bb)` | `i1` condition + two basic block labels | Conditional branch — if `cond` is `i1(1)`, transfer to `true_bb`; otherwise transfer to `false_bb` |
| `Switch(value, default_bb, cases)` | Integer value + default label + list of `(constant, label)` pairs | Multi-way branch — transfers to the label matching `value`, or `default_bb` if no case matches |

> **No fall-through:** Unlike machine code, the IR does not permit implicit fall-through between blocks. Every block must explicitly branch to its successor(s). This makes control-flow analysis trivially correct — the CFG edges are exactly the terminator targets.

### Miscellaneous Instructions

| Instruction | Operands | Result Type | Description |
|---|---|---|---|
| `Call(func, args)` | Callee (function pointer or direct reference) + argument value list | Callee's return type | Invoke a function. Arguments must match the callee's parameter types. If the callee returns `Void`, the `Call` produces no result value. |
| `Phi(incoming)` | List of `(ValueId, BlockId)` pairs | Same type as all incoming values | SSA phi node — selects the value corresponding to the predecessor block from which control arrived. Must appear at the beginning of a basic block, before any non-phi instructions. |
| `Cast(value, target_type)` | Source value + target IR type | `target_type` | Type conversion — changes the type of a value through sign extension, zero extension, truncation, or float/int conversion (see sub-operations below) |
| `BitCast(value, target_type)` | Source value + target IR type | `target_type` | Reinterpret the bits of `value` as `target_type` without performing any conversion. Source and target must have the same bit width. Used for pointer casts and type punning. |

#### Cast Sub-Operations

The `Cast` instruction carries an explicit `CastKind` that specifies the conversion operation:

| CastKind | From → To | Description |
|---|---|---|
| `SExt` | Narrow integer → wider integer | Sign-extend: replicate the sign bit to fill the wider type |
| `ZExt` | Narrow integer → wider integer | Zero-extend: fill upper bits with zeros |
| `Trunc` | Wide integer → narrower integer | Truncate: discard upper bits |
| `FPToSI` | Float → signed integer | Convert floating-point to the nearest signed integer (round toward zero) |
| `FPToUI` | Float → unsigned integer | Convert floating-point to the nearest unsigned integer (round toward zero) |
| `SIToFP` | Signed integer → float | Convert signed integer to floating-point |
| `UIToFP` | Unsigned integer → float | Convert unsigned integer to floating-point |
| `FPTrunc` | Wide float → narrow float | Truncate floating-point precision (e.g., `f64` → `f32`) |
| `FPExt` | Narrow float → wide float | Extend floating-point precision (e.g., `f32` → `f64`) |

### Instruction Summary

All 24 IR instructions in one view:

| # | Instruction | Category | Produces Value |
|---|---|---|---|
| 1 | `Add` | Arithmetic | Yes |
| 2 | `Sub` | Arithmetic | Yes |
| 3 | `Mul` | Arithmetic | Yes |
| 4 | `Div` | Arithmetic | Yes |
| 5 | `Mod` | Arithmetic | Yes |
| 6 | `And` | Bitwise | Yes |
| 7 | `Or` | Bitwise | Yes |
| 8 | `Xor` | Bitwise | Yes |
| 9 | `Shl` | Bitwise | Yes |
| 10 | `Shr` | Bitwise | Yes |
| 11 | `ICmp` | Comparison | Yes (`i1`) |
| 12 | `FCmp` | Comparison | Yes (`i1`) |
| 13 | `Alloca` | Memory | Yes (pointer) |
| 14 | `Load` | Memory | Yes (pointee) |
| 15 | `Store` | Memory | No (`Void`) |
| 16 | `GetElementPtr` | Memory | Yes (pointer) |
| 17 | `Call` | Misc | Yes/No (depends on callee) |
| 18 | `Ret` | Control Flow | No |
| 19 | `Br` | Control Flow | No |
| 20 | `CondBr` | Control Flow | No |
| 21 | `Switch` | Control Flow | No |
| 22 | `Phi` | Misc (SSA) | Yes |
| 23 | `Cast` | Misc | Yes |
| 24 | `BitCast` | Misc | Yes |

---

## Values and Instruction Identity

Every instruction that produces a result is assigned a unique `ValueId` — an opaque handle that other instructions reference as operands. This creates the def-use graph that is central to SSA-based analysis and transformation.

### Value Kinds

The IR recognizes the following value kinds:

| Value Kind | Description |
|---|---|
| `InstructionResult(ValueId)` | The result of an instruction — assigned when the instruction is created |
| `IntConst(type, i128)` | An integer constant of the given IR type with the given value |
| `FloatConst(type, f64)` | A floating-point constant of the given IR type with the given value |
| `NullPtr(type)` | A null pointer constant for the given pointer type |
| `Parameter(index, type)` | A function parameter — the `index`-th parameter with the given type |
| `GlobalRef(name, type)` | A reference to a global variable or function by name |
| `Undefined(type)` | An undefined value of the given type (used for uninitialized variables before SSA construction) |

### Value Numbering

- `ValueId` values are monotonically increasing unsigned integers, scoped per function.
- Parameters are assigned `ValueId`s starting from 0 in order of declaration.
- Each instruction that produces a result receives the next available `ValueId`.
- Constants and global references are interned — identical constants share the same `ValueId`.

### Instruction Metadata

Each instruction carries the following metadata:

- **Result `ValueId`** — The unique identifier for this instruction's result (if it produces one).
- **Result type** — The IR type of the produced value, used for validation and downstream code generation.
- **Source location** — An optional `SourceLocation` recording the original C source file, line, and column. This is propagated to DWARF debug information when `-g` is specified.
- **Parent block** — A reference to the basic block that contains this instruction.

---

## Basic Blocks and Control Flow Graph

> **Source file:** `src/ir/cfg.rs`

### Basic Block Structure

A **basic block** is a straight-line sequence of instructions with the following properties:

1. **Phi nodes first:** Zero or more `Phi` instructions at the beginning of the block.
2. **Body instructions:** Zero or more non-terminator, non-phi instructions in sequential order.
3. **Exactly one terminator:** A single terminator instruction (`Ret`, `Br`, `CondBr`, or `Switch`) at the end.
4. **Unique label:** Each block has a unique `BlockId` identifier within its function.

Instructions within a block execute sequentially from top to bottom. Control flow can only enter a block at the top (at the first phi node or body instruction) and can only exit at the terminator.

```
bb3:                                    ; ← Block label
  %x = Phi [(i32 %a, bb1), (i32 %b, bb2)]  ; ← Phi nodes first
  %y = Add i32 %x, IntConst(i32, 1)        ; ← Body instructions
  %z = ICmp Slt %y, IntConst(i32, 100)     ; ← Body instructions
  CondBr %z, bb4, bb5                       ; ← Terminator (exactly one)
```

### CFG Construction

The **Control Flow Graph (CFG)** is a directed graph where:

- **Nodes** are basic blocks.
- **Edges** are derived from terminator instructions:

| Terminator | Edges Created |
|---|---|
| `Br(target)` | One edge: current block → `target` |
| `CondBr(_, true_bb, false_bb)` | Two edges: current → `true_bb`, current → `false_bb` |
| `Switch(_, default_bb, cases)` | N+1 edges: current → `default_bb`, current → each case target |
| `Ret(_)` | Zero edges (function exit) |

**Entry block:** The first basic block of every function is the designated entry point. All `Alloca` instructions for local variables are placed in the entry block to ensure they dominate all uses.

**Predecessor and successor lists:** The CFG maintains both predecessor and successor adjacency lists for each block, enabling efficient forward and backward traversal:

```
Block bb3:
  Predecessors: [bb1, bb2]
  Successors:   [bb4, bb5]
```

### CFG Well-Formedness Requirements

A well-formed CFG satisfies:

1. Every block is reachable from the entry block (unreachable blocks are removed by DCE).
2. Every non-return block ends with a branch to one or more successors.
3. The entry block has no predecessors (it is the sole entry point).
4. Every successor reference names a valid block within the same function.
5. Phi node incoming edges match the block's predecessor list exactly.

---

## Dominance Analysis

> **Source file:** `src/ir/cfg.rs`

Dominance analysis is a fundamental CFG analysis used to determine which blocks must execute before other blocks. It is the foundation for SSA construction (phi node placement) and various optimization passes.

### Definitions

- **Dominance:** Block A **dominates** block B (written A dom B) if every path from the entry block to B must pass through A. By convention, every block dominates itself.
- **Strict dominance:** A **strictly dominates** B if A dominates B and A ≠ B.
- **Immediate dominator (idom):** The immediate dominator of B is the unique block that strictly dominates B and is dominated by all other strict dominators of B. In simpler terms, idom(B) is the closest dominator of B.
- **Dominance tree:** A tree rooted at the entry block where each node's parent is its immediate dominator. The dominance tree encodes the dominance relation compactly — A dominates B if and only if A is an ancestor of B in the tree.

### Dominance Tree Computation

The bcc compiler uses the **Cooper-Harvey-Kennedy algorithm** for computing the dominance tree. This iterative algorithm is simple to implement, efficient for real-world CFGs, and does not require preprocessing beyond reverse post-order numbering.

**Algorithm outline:**

1. Compute a **reverse post-order (RPO)** numbering of all blocks. This numbering ensures that a block's dominators are visited before the block itself (except for loop back edges).

2. Initialize: `idom(entry) = entry`. All other blocks have `idom = undefined`.

3. Iterate in RPO order until no changes occur:
   ```
   for each block B (in RPO, excluding entry):
       new_idom = first processed predecessor of B
       for each other predecessor P of B:
           if idom(P) is defined:
               new_idom = intersect(new_idom, P)
       if idom(B) ≠ new_idom:
           idom(B) = new_idom
           changed = true
   ```

4. The `intersect(b1, b2)` function walks both `b1` and `b2` up the dominator tree (using RPO numbers) until they meet:
   ```
   fn intersect(b1, b2):
       while b1 ≠ b2:
           while rpo(b1) > rpo(b2): b1 = idom(b1)
           while rpo(b2) > rpo(b1): b2 = idom(b2)
       return b1
   ```

5. When the iteration reaches a fixed point, `idom(B)` is the immediate dominator of every block B.

**Complexity:** O(N²) in the worst case for pathological CFGs, but typically linear for structured control flow produced by C programs.

### Dominance Frontier

The **dominance frontier** of block A is the set of blocks where A's dominance "ends" — formally, the set of blocks B such that A dominates a predecessor of B but does not strictly dominate B itself.

The dominance frontier is critical for SSA construction: it identifies exactly where phi nodes must be placed.

**Algorithm (from Cytron et al.):**

```
for each block B with two or more predecessors:
    for each predecessor P of B:
        runner = P
        while runner ≠ idom(B):
            add B to DF(runner)
            runner = idom(runner)
```

This algorithm walks up the dominator tree from each predecessor of a join point, adding the join point to each block's dominance frontier along the way.

**Example:**

```
     entry
     / \
   bb1  bb2
     \ /
     bb3
```

In this diamond CFG:
- `DF(bb1) = {bb3}` — bb1 dominates itself but not bb3, yet bb1 is a predecessor of bb3.
- `DF(bb2) = {bb3}` — same reasoning.
- `DF(entry) = {}` — entry dominates everything.
- `DF(bb3) = {}` — bb3 has no successors where its dominance ends (assuming bb3 returns).

---

## SSA Construction

> **Source file:** `src/ir/ssa.rs`

SSA construction transforms the IR from an initial form (where variables may be assigned multiple times via `Alloca`/`Store`/`Load` sequences) into SSA form (where every value has exactly one definition, with `Phi` nodes at join points). The bcc compiler uses the **iterated dominance frontier** algorithm, which proceeds in two main steps.

### Why Two-Phase Construction?

The IR builder (`src/ir/builder.rs`) initially emits local variables as `Alloca` instructions in the entry block, with `Store` for assignments and `Load` for uses. This "memory-form" IR is simple to generate from the AST because it avoids the complexity of computing phi nodes during translation.

The `mem2reg` optimization pass (`src/passes/mem2reg.rs`) then promotes eligible `Alloca`/`Load`/`Store` sequences to SSA registers with phi nodes — using the SSA construction algorithm described here. An `Alloca` is eligible for promotion if its address is never taken (i.e., it is only used as the pointer operand of `Load` and `Store` instructions).

### Step 1 — Phi Node Placement

Phi nodes must be placed at every point in the CFG where two or more definitions of the same variable "meet." The iterated dominance frontier identifies these points precisely.

**Algorithm:**

```
for each promotable variable V:
    DefBlocks(V) = set of blocks that contain a Store to V's Alloca
    
    // Compute the iterated dominance frontier of DefBlocks(V)
    worklist = DefBlocks(V)
    visited = ∅
    PhiBlocks(V) = ∅
    
    while worklist is not empty:
        block = worklist.pop()
        for each block F in DF(block):
            if F ∉ PhiBlocks(V):
                PhiBlocks(V) = PhiBlocks(V) ∪ {F}
                insert a Phi node for V at the start of F
                if F ∉ visited:
                    worklist.push(F)
                    visited = visited ∪ {F}
```

The "iterated" aspect is important: when a phi node is inserted in block F, F becomes a new definition site for V, which may in turn require phi nodes at F's dominance frontier blocks — and so on until convergence.

### Step 2 — Variable Renaming

After phi nodes are placed, the renaming pass walks the dominator tree to connect each use of a variable to its reaching definition.

**Algorithm:**

```
fn rename(block, var_stacks):
    // Process phi nodes: each phi is a new definition
    for each Phi node P in block:
        push P's result ValueId onto var_stacks[P.variable]
    
    // Process non-phi instructions
    for each instruction I in block:
        // Replace uses: replace Load(V) with the current SSA value
        for each operand of I that is a Load from a promoted Alloca:
            replace with top of var_stacks[variable]
        
        // Record definitions: Store(val, V) creates a new definition
        if I is Store(value, promoted_alloca):
            push value onto var_stacks[variable]
    
    // Fill in phi node operands in successor blocks
    for each successor S of block:
        for each Phi node P in S:
            set P's incoming value from 'block' = top of var_stacks[P.variable]
    
    // Recurse into dominator tree children
    for each child C in dominator_tree(block):
        rename(C, var_stacks)
    
    // Pop definitions pushed in this block
    restore var_stacks to state before processing this block
```

The dominator tree traversal ensures that each block sees the definitions from its dominating blocks, exactly matching SSA's definition-dominates-use invariant.

### Phi Node Semantics

A `Phi` instruction:

```
%x = Phi [(val1, bb1), (val2, bb2), ...]
```

- **Selects** `val_i` when control flow arrives from predecessor block `bb_i`.
- **Must appear** at the beginning of its block, before any non-phi instructions.
- **All incoming values** must have the same IR type.
- **Must have exactly one** incoming `(value, predecessor)` pair for each predecessor of the block.
- **Conceptually executes simultaneously** with all other phi nodes in the same block — phi nodes in the same block do not observe each other's results (they all read the incoming values from their predecessors).

### Relationship Between Alloca/Load/Store and SSA

| Phase | Variable Representation |
|---|---|
| After IR building | `Alloca` allocates a stack slot; `Store` writes to it; `Load` reads from it |
| After `mem2reg` | Promoted variables become SSA values; `Phi` nodes inserted at join points; `Alloca`/`Load`/`Store` for promoted variables are removed |
| Variables with address taken | Remain as `Alloca`/`Load`/`Store` — not promoted to SSA registers |

---

## IR Builder

> **Source file:** `src/ir/builder.rs`

The IR builder translates the typed AST produced by semantic analysis (`src/sema/`) into IR instructions organized into basic blocks within functions. It operates on one function at a time, maintaining a "current block" insertion point and a mapping from AST symbols to their IR `Alloca` pointers.

### Expression Lowering

| C Expression | IR Translation |
|---|---|
| Arithmetic (`a + b`, `a - b`, `a * b`, `a / b`, `a % b`) | `Add`, `Sub`, `Mul`, `Div`, `Mod` instructions with appropriate signedness flags |
| Comparison (`a < b`, `a == b`, etc.) | `ICmp` or `FCmp` with the appropriate predicate |
| Bitwise (`a & b`, `a \| b`, `a ^ b`, `a << b`, `a >> b`) | `And`, `Or`, `Xor`, `Shl`, `Shr` instructions |
| Pointer arithmetic (`p + n`, `p[i]`) | `GetElementPtr` to compute the address, then `Load` to read the value |
| Type cast (`(int)f`, `(float)i`) | `Cast` with appropriate `CastKind` (`FPToSI`, `SIToFP`, `SExt`, `ZExt`, `Trunc`, etc.) |
| Pointer cast (`(void*)p`) | `BitCast` to reinterpret the pointer type |
| Function call (`f(a, b)`) | `Call` instruction with evaluated argument values |
| Variable read (`x`) | `Load` from the variable's `Alloca` pointer |
| Variable write (`x = expr`) | `Store` the expression result to the variable's `Alloca` pointer |
| Address-of (`&x`) | Return the `Alloca` pointer directly (no `Load`) |
| Dereference (`*p`) | `Load` from the pointer value |
| Struct field access (`s.field`) | `GetElementPtr` to the field, then `Load` |
| Array element access (`a[i]`) | `GetElementPtr` with the index, then `Load` |
| Comma expression (`a, b`) | Evaluate both; result is the value of `b` |
| Ternary (`c ? a : b`) | `CondBr` to two blocks computing `a` and `b`, merge with `Phi` |
| Short-circuit `&&` / `||` | `CondBr` with lazy evaluation — right operand only evaluated if needed |
| `sizeof` / `_Alignof` | Compile-time constant — emitted as `IntConst` |

### Statement Lowering

#### `if / else`

```
    <condition code>
    CondBr %cond, then_bb, else_bb

then_bb:
    <then body>
    Br merge_bb

else_bb:              ; (omitted if no else clause)
    <else body>
    Br merge_bb

merge_bb:
    <continuation>
```

#### `while` Loop

```
    Br cond_bb

cond_bb:
    <condition code>
    CondBr %cond, body_bb, exit_bb

body_bb:
    <loop body>
    Br cond_bb          ; ← back edge

exit_bb:
    <continuation>
```

#### `for` Loop

```
    <init statement>
    Br cond_bb

cond_bb:
    <condition code>
    CondBr %cond, body_bb, exit_bb

body_bb:
    <loop body>
    Br inc_bb

inc_bb:
    <increment expression>
    Br cond_bb          ; ← back edge

exit_bb:
    <continuation>
```

#### `do-while` Loop

```
    Br body_bb

body_bb:
    <loop body>
    <condition code>
    CondBr %cond, body_bb, exit_bb    ; ← back edge on true

exit_bb:
    <continuation>
```

#### `switch`

```
    <value code>
    Switch %val, default_bb, [(const1, case1_bb), (const2, case2_bb), ...]

case1_bb:
    <case 1 body>
    Br next_bb          ; or fall through to case2_bb if no break

case2_bb:
    <case 2 body>
    Br next_bb

default_bb:
    <default body>
    Br next_bb

next_bb:
    <continuation>
```

#### `break` / `continue`

- `break` → `Br` to the current loop's or switch's exit block.
- `continue` → `Br` to the current loop's condition block (for `while`/`for`) or body block (for `do-while`).

#### `return`

- `return expr;` → Evaluate `expr`, then `Ret %value`.
- `return;` → `Ret` with no operand (void function).

#### `goto` / Labels

- Labeled statement `label:` → Create a new basic block with the label name.
- `goto label;` → `Br` to the labeled block. Forward references (goto before the label is seen) are resolved in a post-pass over the function.

### Declaration Lowering

| C Declaration | IR Translation |
|---|---|
| Local variable `int x;` | `%x.addr = Alloca i32` in the entry block |
| Local with initializer `int x = 5;` | `Alloca` + `Store IntConst(i32, 5), %x.addr` |
| Global variable `int g = 42;` | `GlobalVariable @g : i32, init = IntConst(i32, 42)` in the module |
| Function definition | New `Function` with parameters, entry block, and body blocks |
| Function declaration (extern) | `Function` with no body (external linkage) |

> **Alloca placement:** All `Alloca` instructions for local variables are placed in the function's entry block, regardless of the variable's lexical scope. This ensures every `Alloca` dominates all its uses, which is required for `mem2reg` to function correctly. Scope-based lifetime is not modeled in the IR; the stack frame is flat.

---

## Module Structure

The top-level IR container is the `Module`, which represents an entire C translation unit.

### Module Contents

```
Module
├── globals: Vec<GlobalVariable>
│   ├── @g1 : i32, init = IntConst(i32, 42), linkage = Internal
│   ├── @g2 : Ptr(i8), init = NullPtr, linkage = External
│   └── ...
├── functions: Vec<Function>
│   ├── Function @main : (i32, Ptr(Ptr(i8))) -> i32
│   │   ├── entry:
│   │   │   ├── %0 = Parameter 0 : i32
│   │   │   ├── %1 = Parameter 1 : Ptr(Ptr(i8))
│   │   │   ├── %retval = Alloca i32
│   │   │   ├── Store i32 IntConst(i32, 0), %retval
│   │   │   └── Br %return
│   │   └── return:
│   │       ├── %rv = Load i32, %retval
│   │       └── Ret i32 %rv
│   ├── Function @helper : (i32) -> i32
│   │   └── ...
│   └── ...
└── target: TargetConfig
```

### Function Components

Each `Function` contains:

| Component | Description |
|---|---|
| `name` | The function's symbol name (e.g., `"main"`, `"helper"`) |
| `return_type` | The IR return type |
| `params` | Ordered list of `(name, IrType)` pairs, each assigned a `ValueId` |
| `blocks` | Ordered list of `BasicBlock` values; the first block is the entry |
| `entry_block` | Reference to the first block (convenience accessor) |
| `linkage` | `Internal` (static), `External` (visible to linker), `Weak`, etc. |
| `is_declaration` | `true` for extern declarations with no body |

### GlobalVariable Components

Each `GlobalVariable` contains:

| Component | Description |
|---|---|
| `name` | The global symbol name |
| `ty` | The IR type of the global |
| `initializer` | Optional initial value (constant expression) |
| `linkage` | `Internal` (file-scoped `static`), `External` (visible), `Common` |
| `is_const` | `true` for `const`-qualified globals (placed in `.rodata`) |
| `alignment` | Alignment in bytes (from target ABI or explicit `__attribute__((aligned(...)))`) |

---

## IR Textual Representation

For debugging and testing purposes, the bcc IR supports a human-readable textual format that can be printed to stderr or written to a file. This textual format is not the authoritative representation — the in-memory data structures in `src/ir/` are authoritative — but the textual format provides a convenient way to inspect the IR at various pipeline stages.

### Syntax

```
; Module-level comments begin with semicolons

target = "x86_64-linux-gnu"

@global_var : i32 = IntConst(i32, 42)

define i32 @main(i32 %argc, Ptr(Ptr(i8)) %argv) {
entry:
    %retval = Alloca i32
    Store i32 IntConst(i32, 0), Ptr(i32) %retval
    %cmp = ICmp Sgt i32 %argc, IntConst(i32, 1)
    CondBr i1 %cmp, %then, %else

then:
    %msg = GetElementPtr Ptr(Array(i8, 14)) @hello_str, i64 0, i64 0
    %r = Call i32 @puts(Ptr(i8) %msg)
    Br %merge

else:
    Br %merge

merge:
    %rv = Load i32, Ptr(i32) %retval
    Ret i32 %rv
}

define i32 @puts(Ptr(i8) %s) {
    ; external declaration — no body
}
```

### Formatting Conventions

| Element | Format |
|---|---|
| Function definition | `define <return_type> @<name>(<params>) {` |
| External declaration | `define <return_type> @<name>(<params>) { ; external }` |
| Basic block label | `<name>:` (indented 0 spaces) |
| Instructions | 4-space indent: `%<name> = <opcode> <type> <operands>` |
| Value references | `%<name>` for local values, `@<name>` for globals/functions |
| Constants | `IntConst(<type>, <value>)`, `FloatConst(<type>, <value>)`, `NullPtr(<type>)` |
| Type annotations | Prefixed to operands: `i32 %x`, `Ptr(i8) %p`, `f64 %f` |
| Comments | `; <text>` — line comments starting with semicolons |

> **Note:** The textual format is intended for developer use during debugging and in test assertions. It is not parsed back into IR (there is no IR text parser). The IR is always constructed programmatically via the IR builder.

---

## Interaction with Optimization Passes

> **Reference:** `src/passes/` and `docs/architecture.md` Section 3.5

The IR is designed to be transformed in place by optimization passes. The IR module provides three categories of APIs that passes consume:

### Iteration APIs

| API | Description |
|---|---|
| `module.functions()` | Iterate over all functions in the module |
| `function.blocks()` | Iterate over all basic blocks in a function (in layout order) |
| `block.instructions()` | Iterate over all instructions in a block (phi nodes first, then body, then terminator) |
| `function.blocks_rpo()` | Iterate in reverse post-order (for forward dataflow analysis) |
| `function.blocks_po()` | Iterate in post-order (for backward dataflow analysis) |

### Mutation APIs

| API | Description |
|---|---|
| `replace_instruction(old, new)` | Replace an instruction with a different instruction, updating all uses |
| `remove_instruction(inst)` | Remove an instruction (must have no remaining uses) |
| `insert_before(anchor, inst)` | Insert an instruction before another instruction |
| `insert_after(anchor, inst)` | Insert an instruction after another instruction |
| `split_block(inst)` | Split a block at the given instruction — instructions from `inst` onward move to a new block |
| `merge_blocks(a, b)` | Merge block `b` into `a` (requires `a` has only `Br(b)` as terminator and `b` has only `a` as predecessor) |
| `remove_block(block)` | Remove a basic block entirely (must have no predecessors) |
| `replace_all_uses(old_val, new_val)` | Replace all uses of `old_val` with `new_val` across the entire function |

### Analysis APIs

| API | Description |
|---|---|
| `value.uses()` | All instructions that use this value as an operand (use-def chain) |
| `value.def()` | The instruction that defines this value (def-use chain) |
| `value.has_uses()` | Whether the value is used anywhere (for dead code detection) |
| `block.dominates(other)` | Whether this block dominates `other` (from the dominance tree) |
| `block.idom()` | The immediate dominator of this block |
| `block.dom_frontier()` | The dominance frontier set of this block |
| `instruction.is_pure()` | Whether the instruction has no side effects (useful for DCE) |

### Pass Pipeline Configuration

The optimization level (`-O0`, `-O1`, `-O2`) determines which passes run and in what order. The pass pipeline is configured in `src/passes/pipeline.rs`:

| Level | Passes (in order) | Behavior |
|---|---|---|
| `-O0` | *(none)* | No optimization — IR goes directly to code generation as-is |
| `-O1` | `mem2reg` → `constant_fold` → `dce` | Basic optimization: promote memory to registers, fold constants, remove dead code |
| `-O2` | `mem2reg` → `constant_fold` → `cse` → `simplify` → `dce` (iterated) | Aggressive optimization: adds common subexpression elimination and algebraic simplification; the entire sequence iterates until no further changes occur (fixed-point iteration) |

#### Pass Descriptions

| Pass | Source File | Description |
|---|---|---|
| `mem2reg` | `src/passes/mem2reg.rs` | Promotes `Alloca`/`Load`/`Store` sequences to SSA registers with phi nodes (uses SSA construction from `src/ir/ssa.rs`) |
| `constant_fold` | `src/passes/constant_fold.rs` | Evaluates arithmetic on known constants at compile time: `Add(IntConst(2), IntConst(3))` → `IntConst(5)` |
| `dce` | `src/passes/dce.rs` | Removes unreachable blocks and instructions whose results are never used |
| `cse` | `src/passes/cse.rs` | Detects repeated identical computations via value numbering and replaces redundant instances with the first computation's result |
| `simplify` | `src/passes/simplify.rs` | Algebraic simplification and strength reduction: `x + 0` → `x`, `x * 1` → `x`, `x * 2` → `x << 1` |

> **See also:** [Architecture Overview](../architecture.md) for the overall pass manager design and per-pass details.

---

## Interaction with Code Generation

> **Reference:** `src/codegen/` — x86-64, i686, AArch64, RISC-V 64 backends

Code generators consume the optimized IR and produce machine code for the target architecture. The IR-to-machine-code translation follows a structured process:

### Function-Level Processing

The code generator processes one function at a time:

1. **Instruction selection (isel):** Each IR instruction is pattern-matched to one or more target machine instructions. IR operations map to architecture-specific opcodes (e.g., `Add i32` → x86-64 `ADD r32, r32`; `Add i32` → AArch64 `ADD Wn, Wn, Wn`).

2. **Register allocation:** IR `ValueId` references correspond to virtual registers. The register allocator (shared across all backends, in `src/codegen/regalloc.rs`) maps virtual registers to the target's physical register file. When physical registers are exhausted, spill code (`Store`/`Load` to stack slots) is inserted.

3. **Machine code encoding:** The architecture-specific encoder translates machine instructions into their binary encoding (bytes). This is the integrated assembler — no external `as` is invoked.

### IR → Machine Code Mapping

| IR Concept | Machine Code Equivalent |
|---|---|
| `ValueId` (virtual register) | Physical register or stack slot (after register allocation) |
| IR type (`i32`, `i64`, `f64`, etc.) | Operand size selector (32-bit, 64-bit, double-precision) |
| `Alloca` | Stack frame slot at a fixed offset from the frame/stack pointer |
| `Load` / `Store` | Memory access instructions (e.g., `MOV`, `LDR`, `LW`) |
| `Call` | ABI-specific calling sequence — argument register/stack assignment, call instruction, return value retrieval |
| `Phi` | Resolved during register allocation via parallel copy insertion or eliminated by coalescing |
| `Br` / `CondBr` / `Switch` | Branch instructions (`JMP`, `Jcc`, `B`, `BEQ`, `BNE`, etc.) |
| `Ret` | Function epilogue — restore callee-saved registers, deallocate frame, return instruction |
| `GetElementPtr` | Address computation instructions (addition, shift, multiply) |
| `Cast` / `BitCast` | Conversion instructions (`MOVSX`, `MOVZX`, `CVTSI2SD`, etc.) or no-op (for bitcast with same-size registers) |

### Phi Node Resolution

SSA phi nodes do not have a direct hardware equivalent. They are resolved during or after register allocation:

1. **Parallel copy insertion:** Before register allocation, each phi node is decomposed into copies placed at the end of each predecessor block. For `%x = Phi [(val1, bb1), (val2, bb2)]`, a copy `%x = val1` is inserted at the end of `bb1`, and `%x = val2` at the end of `bb2` (before the terminator).

2. **Copy coalescing:** The register allocator attempts to assign the same physical register to the phi result and all its incoming values, eliminating the copies entirely.

3. **Critical edge splitting:** When a predecessor has multiple successors and a successor has multiple predecessors, a new "split" block is inserted along the edge to provide a safe location for the copies.

### Target-Specific Considerations

While the IR is target-independent, the code generator must account for target-specific details:

| Aspect | How IR Communicates It |
|---|---|
| Pointer size | IR types `Ptr(T)` — the code generator queries `TargetConfig::pointer_width()` |
| Calling convention | `Call` instructions — the code generator implements the target ABI (System V AMD64, cdecl, AAPCS64, LP64D) |
| Endianness | Not encoded in IR — all targets are little-endian; the code generator handles byte ordering |
| Alignment | Struct layout — the code generator queries `TargetConfig` for field alignment |
| Security hardening | Not represented in IR — the x86-64 backend applies retpoline, CET, and stack probing during code emission based on CLI flags |

The IR intentionally does **not** encode target-specific details. All architecture-specific decisions are made by the code generator, which reads the target configuration and applies the appropriate instruction sequences.

---

## Appendix: Complete IR Example

The following example shows the complete IR for a simple C function:

**C source:**

```c
int sum(int n) {
    int total = 0;
    for (int i = 0; i < n; i++) {
        total += i;
    }
    return total;
}
```

**IR (before optimization, before `mem2reg`):**

```
define i32 @sum(i32 %n) {
entry:
    %n.addr = Alloca i32
    %total = Alloca i32
    %i = Alloca i32
    Store i32 %n, Ptr(i32) %n.addr
    Store i32 IntConst(i32, 0), Ptr(i32) %total
    Store i32 IntConst(i32, 0), Ptr(i32) %i
    Br %for.cond

for.cond:
    %i.val = Load i32, Ptr(i32) %i
    %n.val = Load i32, Ptr(i32) %n.addr
    %cmp = ICmp Slt i32 %i.val, %n.val
    CondBr i1 %cmp, %for.body, %for.end

for.body:
    %i.val2 = Load i32, Ptr(i32) %i
    %total.val = Load i32, Ptr(i32) %total
    %add = Add i32 %total.val, %i.val2
    Store i32 %add, Ptr(i32) %total
    %i.val3 = Load i32, Ptr(i32) %i
    %inc = Add i32 %i.val3, IntConst(i32, 1)
    Store i32 %inc, Ptr(i32) %i
    Br %for.cond

for.end:
    %result = Load i32, Ptr(i32) %total
    Ret i32 %result
}
```

**IR (after `mem2reg` at `-O1`):**

```
define i32 @sum(i32 %n) {
entry:
    Br %for.cond

for.cond:
    %total.0 = Phi [(IntConst(i32, 0), entry), (i32 %add, for.body)]
    %i.0 = Phi [(IntConst(i32, 0), entry), (i32 %inc, for.body)]
    %cmp = ICmp Slt i32 %i.0, %n
    CondBr i1 %cmp, %for.body, %for.end

for.body:
    %add = Add i32 %total.0, %i.0
    %inc = Add i32 %i.0, IntConst(i32, 1)
    Br %for.cond

for.end:
    Ret i32 %total.0
}
```

Note how `mem2reg` eliminated all `Alloca`, `Load`, and `Store` instructions for the local variables `n`, `total`, and `i`, replacing them with phi nodes at the loop header (`for.cond`) that merge values from the entry block (initial values) and the loop body (updated values). The resulting SSA form is compact and directly exposes the dataflow to subsequent optimization passes.
