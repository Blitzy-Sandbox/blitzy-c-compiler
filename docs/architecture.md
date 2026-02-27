# Compiler Architecture

## Executive Summary

**bcc** is a complete, self-contained C11 compiler written in pure Rust targeting Linux
ELF output across four processor architectures: x86-64, i686, AArch64, and RISC-V 64.
It implements the full compilation pipeline — from raw source text to linked ELF binary —
with an integrated preprocessor, assembler, and linker. The compiler requires zero
external toolchain invocations and zero crate dependencies beyond the Rust standard
library.

### Key Architectural Principles

| Principle | Description |
|---|---|
| **Zero External Dependencies** | The compiler uses exclusively the Rust standard library (`std`). Every `[dependencies]` section in `Cargo.toml` is empty. All functionality — parsing, code generation, ELF emission, DWARF encoding, archive reading — is implemented from scratch. |
| **Sequential Pipeline** | Compilation proceeds through well-defined phases with strict data flow contracts at each boundary. Each phase consumes the output of the previous phase and produces input for the next. |
| **Target-Independent Middle-End** | The intermediate representation (IR) and optimization passes operate on a target-independent SSA-form IR. Target-specific concerns are isolated to the code generation backends. |
| **Integrated Assembler and Linker** | Machine code encoding and ELF linking are built into the compiler. No external `as`, `gas`, `ld`, `lld`, `gold`, or any other tool is invoked during compilation. |
| **Cross-Cutting Services** | Diagnostics, source mapping, string interning, and arena allocation are shared services consumed by every pipeline phase through the `common` module. |
| **Four-Architecture Support** | A single `--target` flag selects the code generator, ABI, ELF format, and relocation types at compile time. All four backends share a common `CodeGen` trait and register allocator. |

---

## High-Level Pipeline

The compiler processes source code through a strict sequential pipeline. Each box
represents a compilation phase, and the annotations between boxes describe the data
flowing across each boundary.

```
Source Code (.c)
    │
    ▼
┌─────────────────────┐
│  Preprocessor        │  #include, #define, #if, macro expansion,
│  (src/frontend/      │  conditional compilation, include resolution
│   preprocessor/)     │
└──────────┬──────────┘
           │ Preprocessed Source (String)
           ▼
┌─────────────────────┐
│  Lexer               │  Tokenization with source position tracking
│  (src/frontend/      │
│   lexer/)            │
└──────────┬──────────┘
           │ Vec<Token>
           ▼
┌─────────────────────┐
│  Parser              │  Recursive descent, C11 + GCC extensions,
│  (src/frontend/      │  error recovery with synchronization tokens
│   parser/)           │
└──────────┬──────────┘
           │ Untyped AST (TranslationUnit)
           ▼
┌─────────────────────┐
│  Semantic Analysis   │  Type checking, scope resolution,
│  (src/sema/)         │  symbol tables, implicit conversions
└──────────┬──────────┘
           │ Typed AST + Symbol Tables
           ▼
┌─────────────────────┐
│  IR Generator        │  Lowering to SSA-form intermediate
│  (src/ir/)           │  representation with phi nodes
└──────────┬──────────┘
           │ SSA IR (BasicBlocks, Instructions, Phi nodes)
           ▼
┌─────────────────────┐
│  Optimizer           │  -O0 (none), -O1 (basic), -O2 (aggressive)
│  (src/passes/)       │  Constant folding, DCE, CSE, mem2reg
└──────────┬──────────┘
           │ Optimized IR
           ▼
┌─────────────────────┐
│  Code Generator      │  Target-specific instruction selection,
│  (src/codegen/)      │  register allocation, machine code encoding
│                      │  ┌────────┬────────┬───────────┐
│                      │  │ x86-64 │  i686  │  AArch64  │
│                      │  │        │        │  RISC-V64 │
│                      │  └────────┴────────┴───────────┘
└──────────┬──────────┘
           │ Machine Code + Relocations + Symbols
           ▼
┌─────────────────────┐              ┌─────────────────────┐
│  Linker              │◄──(if -g)───│  Debug Info          │
│  (src/linker/)       │             │  (src/debug/)        │
│                      │             │  DWARF v4 sections   │
└──────────┬──────────┘              └─────────────────────┘
           │
           ▼
      ELF Binary
      (executable / shared library / relocatable object)
```

The **Driver** module (`src/driver/`) orchestrates this entire pipeline. It parses
CLI arguments, constructs the `TargetConfig`, sequences each phase, propagates errors,
and manages the process exit code. The entry point `src/main.rs` delegates immediately
to the driver.

---

## Module Descriptions

### 1. Driver Module — `src/driver/`

**Purpose:** CLI argument parsing, target configuration, and pipeline orchestration.

**Key Files:**

| File | Responsibility |
|---|---|
| `mod.rs` | Module exports; pipeline orchestration coordinating all compilation phases |
| `cli.rs` | GCC-compatible CLI argument parsing (`-c`, `-o`, `-I`, `-D`, `-U`, `-L`, `-l`, `-g`, `-O[012]`, `-shared`, `-fPIC`, `-mretpoline`, `-fcf-protection`, `-static`, `--target`) |
| `target.rs` | Target triple parsing; maps `--target` values to architecture-specific ABI, ELF format, and code generator selection via the `TargetConfig` struct |
| `pipeline.rs` | Compilation pipeline sequencing: preprocessor → lexer → parser → semantic analysis → IR generation → optimization → code generation → linking |

**Entry Point:** `src/main.rs` parses `std::env::args()`, constructs the driver, and
delegates execution. The driver returns a process exit code: 0 on success, 1 on any
compilation error.

**Responsibilities:**
- Parse all GCC-compatible CLI flags
- Resolve `--target` to a `TargetConfig` struct containing pointer width, endianness, ABI parameters, ELF class, and relocation type set
- Sequence all pipeline phases in the correct order
- Handle error propagation — if any phase emits an error diagnostic, subsequent phases are skipped
- Manage output modes: `-c` (compile to object), default (link to executable), `-shared` (shared library)

**Dependencies:** Imports from all other modules (`frontend`, `sema`, `ir`, `passes`, `codegen`, `linker`, `debug`, `common`) to orchestrate the full pipeline.

---

### 2. Frontend Module — `src/frontend/`

**Purpose:** C11 source text processing — preprocessing, tokenization, and parsing into an abstract syntax tree.

The frontend is organized into three submodules that execute sequentially:

#### 2.1 Preprocessor — `src/frontend/preprocessor/`

| File | Responsibility |
|---|---|
| `mod.rs` | Entry point: `Preprocessor::process()` accepts source text and `-I`/`-D`/`-U` options, returns fully expanded source |
| `directives.rs` | Directive dispatcher for `#include`, `#define`, `#undef`, `#if`/`#ifdef`/`#ifndef`/`#elif`/`#else`/`#endif`, `#pragma`, `#error`, `#warning`, `#line` |
| `macros.rs` | Macro definition storage and expansion: object-like, function-like, recursive expansion guard, stringification (`#`), token pasting (`##`), variadic macros (`__VA_ARGS__`) |
| `conditional.rs` | Conditional compilation stack tracking `#if`/`#else`/`#endif` nesting; branch selection based on expression evaluation |
| `expression.rs` | Preprocessor constant expression evaluator: integer arithmetic, `defined()` operator, logical and comparison operators |
| `include.rs` | Include path resolution: searches `-I` directories, bundled header path (`include/`), system header paths; supports `<header.h>` and `"header.h"` forms |

#### 2.2 Lexer — `src/frontend/lexer/`

| File | Responsibility |
|---|---|
| `mod.rs` | Entry point: `Lexer::tokenize()` consumes preprocessed source text, produces `Vec<Token>` |
| `token.rs` | `Token` struct and `TokenKind` enum (137+ variants for C11 keywords, operators, punctuation, literals, identifiers) with source span |
| `keywords.rs` | Keyword lookup table: 44 C11 keywords (including `_Alignas`, `_Alignof`, `_Atomic`, `_Bool`, `_Complex`, `_Generic`, `_Noreturn`, `_Static_assert`, `_Thread_local`) plus GCC extension keywords (`__attribute__`, `__builtin_*`, `__asm__`, `typeof`, `__typeof__`, `__extension__`) |
| `literals.rs` | Numeric literal parser (decimal, hex, octal, binary, float with exponents, suffixes) and string/character literal parser with C escape sequences |
| `source.rs` | `SourceLocation` struct with file ID, byte offset, line, column; span tracking for multi-character tokens |

#### 2.3 Parser — `src/frontend/parser/`

| File | Responsibility |
|---|---|
| `mod.rs` | Entry point: `Parser::parse()` recursive-descent parser producing `TranslationUnit` AST root; error recovery with synchronization tokens |
| `ast.rs` | Complete AST node hierarchy: `TranslationUnit`, `Declaration`, `FunctionDef`, `Statement` (15+ variants), `Expression` (25+ variants), `TypeSpecifier`, `GccAttribute`, `AsmStatement` |
| `declarations.rs` | Declaration parser: variables with initializers, function declarations, `typedef`, struct/union/enum definitions, forward declarations, `_Static_assert` |
| `expressions.rs` | Expression parser using precedence climbing for all C operators (15 precedence levels); handles ternary, comma, assignment, cast, `sizeof`, `_Alignof`, `_Generic` |
| `statements.rs` | Statement parser: compound blocks, `if`/`else`, `for`, `while`, `do`-`while`, `switch`/`case`/`default`, `break`, `continue`, `return`, `goto`, labeled statements |
| `types.rs` | Type specifier parser: base types, qualifiers (`const`, `volatile`, `restrict`, `_Atomic`), pointer/array/function-pointer declarators, abstract declarators |
| `gcc_extensions.rs` | GCC extension parser: `__attribute__((...))` with common attributes (packed, aligned, section, unused, deprecated, visibility, format), statement expressions, `typeof`, computed goto, inline assembly with constraints, `__extension__` |

**Input:** Raw C source file text + CLI options (`-I` include paths, `-D` macro definitions, `-U` macro undefinitions).

**Output:** Untyped AST — a `TranslationUnit` root node containing all top-level declarations and definitions with source locations preserved on every node.

**Dependencies:** Imports from `common` for `SourceLocation`, `Diagnostic`, `InternedString`, `SourceMap`.

---

### 3. Semantic Analysis Module — `src/sema/`

**Purpose:** Type checking, scope resolution, symbol table management, and implicit conversion insertion. This phase transforms the untyped AST into a fully typed AST where every expression carries a resolved type and every identifier is bound to its declaration.

**Key Files:**

| File | Responsibility |
|---|---|
| `mod.rs` | Entry point: `SemanticAnalyzer::analyze()` accepts untyped AST, returns typed AST or diagnostic errors |
| `types.rs` | Type representation with target-parametric sizes: `CType` enum covering `void`, integer types (char through long long), floating types (float, double, long double), pointers, arrays, structs, unions, enums, functions, typedefs |
| `type_check.rs` | Type checking for assignments, function call argument matching, return type validation, binary/unary operator type compatibility |
| `type_conversion.rs` | C11 implicit conversion rules: integer promotions, usual arithmetic conversions, pointer-to-integer conversions, array-to-pointer decay, function-to-pointer decay |
| `scope.rs` | Scope stack management: file scope, function scope, block scope, prototype scope; variable shadowing, scope entry/exit |
| `symbol_table.rs` | `SymbolTable` with scoped insertion and lookup; stores name, type, storage class, linkage, definition status; detects redeclaration conflicts |
| `storage.rs` | Storage class specifier validation: conflicting specifier detection, linkage resolution, `extern`/`static` scoping rules |

**Input:** Untyped AST (`TranslationUnit`) from the parser.

**Output:** Typed AST with resolved symbols, type annotations on every expression, implicit conversion nodes inserted where required, and fully populated symbol tables.

**Contract:** Every expression has a resolved type. All identifiers are bound to their declarations. Type mismatches have been rejected as compilation errors. The typed AST is ready for IR lowering.

**Dependencies:** Imports from `frontend::parser` (AST node types) and `common` (diagnostics, source map, interning).

---

### 4. IR Module — `src/ir/`

**Purpose:** Target-independent intermediate representation in SSA (Static Single Assignment) form. The IR abstracts over target-specific differences, providing a uniform representation that the optimizer and code generator can work with.

**Key Files:**

| File | Responsibility |
|---|---|
| `mod.rs` | IR module exports; defines the IR API surface |
| `types.rs` | IR type system: integer types (`i1`, `i8`, `i16`, `i32`, `i64`), float types (`f32`, `f64`), pointer type, aggregate types, void |
| `instructions.rs` | IR instruction set: `Add`, `Sub`, `Mul`, `Div`, `Mod`, `And`, `Or`, `Xor`, `Shl`, `Shr`, `ICmp`, `FCmp`, `Load`, `Store`, `Alloca`, `GetElementPtr`, `Call`, `Ret`, `Br`, `CondBr`, `Switch`, `Phi`, `Cast`, `BitCast` |
| `builder.rs` | `IrBuilder` translating typed AST to IR: expression lowering, statement lowering, declaration lowering, function body construction |
| `cfg.rs` | Control flow graph: `BasicBlock` with instruction list, predecessor/successor edges; CFG construction, dominance tree, dominance frontier computation |
| `ssa.rs` | SSA construction: iterated dominance frontier algorithm for phi-node placement, variable renaming pass |

**Input:** Typed AST with resolved symbols from semantic analysis.

**Output:** SSA-form IR organized as a `Module` containing `Function` definitions, each composed of `BasicBlock`s connected by control flow edges, with `Phi` nodes at join points.

**IR Instruction Categories:**
- **Arithmetic:** `Add`, `Sub`, `Mul`, `Div`, `Mod`
- **Bitwise:** `And`, `Or`, `Xor`, `Shl`, `Shr`
- **Comparison:** `ICmp`, `FCmp`
- **Memory:** `Load`, `Store`, `Alloca`, `GetElementPtr`
- **Control Flow:** `Br`, `CondBr`, `Switch`, `Ret`
- **Other:** `Call`, `Phi`, `Cast`, `BitCast`

**Dependencies:** Imports from `sema` (typed AST, type info, symbol tables) and `common` (diagnostics, source map).

> For detailed IR design documentation, see [`docs/internals/ir.md`](internals/ir.md).

---

### 5. Optimization Passes Module — `src/passes/`

**Purpose:** Transform the SSA-form IR to improve code quality. The pass manager executes different pass pipelines depending on the optimization level selected by the user.

**Key Files:**

| File | Responsibility |
|---|---|
| `mod.rs` | Pass manager with `FunctionPass` and `ModulePass` traits; pass registration and execution |
| `constant_fold.rs` | Constant folding: evaluate `Add(Const(2), Const(3))` → `Const(5)` for all arithmetic, comparison, and logical operations |
| `dce.rs` | Dead code elimination: mark-sweep from function entry; remove unreachable blocks and instructions with no uses |
| `cse.rs` | Common subexpression elimination: value numbering to detect and eliminate redundant computations |
| `simplify.rs` | Algebraic simplification: identity removal (`x + 0` → `x`), strength reduction (`x * 2` → `x << 1`), constant propagation |
| `mem2reg.rs` | Promote stack `Alloca` instructions to SSA registers where the address is never taken; inserts phi nodes at join points |
| `pipeline.rs` | Pass pipeline configuration per optimization level |

**Pass Pipeline by Optimization Level:**

| Level | Passes Executed |
|---|---|
| `-O0` | No passes (identity transform — IR passes through unchanged) |
| `-O1` | `mem2reg` → `constant_fold` → `dce` |
| `-O2` | `mem2reg` → `constant_fold` → `cse` → `simplify` → `dce` (iterated to fixed point) |

**Input:** SSA-form IR from the IR generator.

**Output:** Optimized IR — the same IR types, but with dead code eliminated, constants folded, common subexpressions removed, and stack allocations promoted to registers. The optimized IR is semantically equivalent to the input.

**Dependencies:** Imports from `ir` (IR types, instructions, basic blocks, CFG).

---

### 6. Code Generation Module — `src/codegen/`

**Purpose:** Target-specific machine code generation with an integrated assembler. Each architecture backend implements the shared `CodeGen` trait and uses the shared register allocator.

**Shared Components:**

| File | Responsibility |
|---|---|
| `mod.rs` | `CodeGen` trait definition (`fn generate(&self, ir: &Module, target: &TargetConfig) -> ObjectCode`), target backend dispatch |
| `regalloc.rs` | Linear scan register allocator: live interval computation, register assignment, spill code insertion; parameterized by target register file |

**Architecture Backends:**

#### 6.1 x86-64 Backend — `src/codegen/x86_64/`

| File | Responsibility |
|---|---|
| `mod.rs` | x86-64 `CodeGen` implementation; coordinates instruction selection, register allocation, encoding, and security hardening |
| `isel.rs` | Instruction selection: pattern matching IR instructions to x86-64 `MachineInstr` sequences; handles addressing modes, immediate operands, instruction combining |
| `encoding.rs` | Integrated assembler: REX prefix generation, opcode emission, ModR/M byte construction, SIB byte, displacement/immediate encoding |
| `abi.rs` | System V AMD64 ABI: register argument passing (`rdi`, `rsi`, `rdx`, `rcx`, `r8`, `r9`), stack frame layout with 16-byte alignment, callee-saved register preservation (`rbx`, `r12`–`r15`, `rbp`), SSE argument passing (`xmm0`–`xmm7`) |
| `security.rs` | Retpoline thunks (`-mretpoline`), CET `endbr64` instrumentation (`-fcf-protection`), stack probing for frames exceeding 4096 bytes |

#### 6.2 i686 Backend — `src/codegen/i686/`

| File | Responsibility |
|---|---|
| `mod.rs` | i686 `CodeGen` implementation |
| `isel.rs` | Instruction selection: 32-bit register set, no REX prefix, 32-bit addressing modes, 64-bit arithmetic via register pairs |
| `encoding.rs` | 32-bit x86 instruction encoding without REX; legacy opcode map |
| `abi.rs` | System V i386 cdecl ABI: all arguments on stack (right-to-left push), caller cleanup, `eax`/`edx` return pair, 16-byte stack alignment at call site |

#### 6.3 AArch64 Backend — `src/codegen/aarch64/`

| File | Responsibility |
|---|---|
| `mod.rs` | AArch64 `CodeGen` implementation |
| `isel.rs` | Instruction selection: fixed-width 32-bit instructions, barrel shifter operands, conditional select, load/store pair instructions |
| `encoding.rs` | Fixed-width 32-bit instruction encoding with field packing for data-processing, load/store, branch, and SIMD instructions |
| `abi.rs` | AAPCS64 ABI: `x0`–`x7` integer argument registers, `v0`–`v7` SIMD/FP argument registers, `x30` link register, SP 16-byte alignment, callee-saved `x19`–`x28` |

#### 6.4 RISC-V 64 Backend — `src/codegen/riscv64/`

| File | Responsibility |
|---|---|
| `mod.rs` | RISC-V 64 `CodeGen` implementation |
| `isel.rs` | Instruction selection: RV64I base integer, M extension (multiply/divide), A extension (atomics), F/D extensions (floating-point) |
| `encoding.rs` | Variable-length instruction encoding: 32-bit base formats (R/I/S/B/U/J), optional 16-bit compressed instructions (C extension) |
| `abi.rs` | LP64D ABI: `a0`–`a7` integer argument registers, `fa0`–`fa7` float argument registers, `ra` return address, `s0`–`s11` callee-saved, SP 16-byte alignment |

**Input:** Optimized IR + `TargetConfig` struct.

**Output:** Machine code bytes (`Vec<u8>`), relocation entries (describing unresolved references), symbol definitions (with binding and visibility), and section assignments (`.text`, `.data`, `.rodata`, `.bss`).

**Dependencies:** Imports from `ir` (optimized IR), `passes` (pass pipeline output), and `common` (target configuration, diagnostics).

---

### 7. Linker Module — `src/linker/`

**Purpose:** Integrated ELF linker that produces final binaries without invoking any external tools. Reads ELF relocatable objects and `ar` static archives, resolves symbols, applies relocations, merges sections, and emits complete ELF binaries.

**Key Files:**

| File | Responsibility |
|---|---|
| `mod.rs` | Linker orchestrator: `link()` function accepting compiled objects, CRT paths, library paths, and output mode |
| `elf.rs` | ELF format library: `Elf32Ehdr`/`Elf64Ehdr` header structs, section header parsing/writing, program header generation, string table construction; supports both reading (input `.o`) and writing (output binaries) |
| `archive.rs` | `ar` archive reader: parse archive magic, header entries; extract individual ELF object members for symbol resolution |
| `relocations.rs` | Relocation processing for all four architectures: `R_X86_64_64`, `R_X86_64_PC32`, `R_X86_64_PLT32`, `R_X86_64_GOT*`; `R_386_32`, `R_386_PC32`; `R_AARCH64_ABS64`, `R_AARCH64_CALL26`; `R_RISCV_*` |
| `sections.rs` | Section merging: combine `.text` sections from multiple objects; compute addresses and file offsets; `.bss`/`.rodata`/`.data` layout; section-to-segment mapping for `PT_LOAD` |
| `symbols.rs` | Symbol resolution: collect global/local symbols from input objects and archives; resolve undefined references; detect duplicates; handle weak symbols |
| `dynamic.rs` | Dynamic linking for `-shared` output: `.dynamic` section, `.dynsym`/`.dynstr` tables, `.plt`/`.got` stubs, `DT_NEEDED` entries, `DT_SONAME` |
| `script.rs` | Default linker script behavior: section ordering (`.text` → `.rodata` → `.data` → `.bss`), entry point (`_start`), page-aligned segment layout, `PT_INTERP` for dynamic executables |

**Output Modes:**

| Mode | Flag | Description |
|---|---|---|
| Relocatable Object | `-c` | Emits a single `.o` file; no linking performed |
| Static Executable | (default) | Links with CRT objects and `libc.a`; produces a directly runnable ELF binary |
| Shared Library | `-shared` | Produces a `.so` with dynamic symbol tables, PLT/GOT, and `PT_DYNAMIC` |

**ELF Format Support:**

| Target | ELF Class | Relocation Types |
|---|---|---|
| x86-64 | ELF64 | `R_X86_64_*` |
| i686 | ELF32 | `R_386_*` |
| AArch64 | ELF64 | `R_AARCH64_*` |
| RISC-V 64 | ELF64 | `R_RISCV_*` |

**Dependencies:** Imports from `codegen` (machine code, relocations, symbols) and `common` (diagnostics, target configuration).

> For detailed linker internals, see [`docs/internals/linker.md`](internals/linker.md).

---

### 8. Debug Info Module — `src/debug/`

**Purpose:** DWARF v4 debug information generation for source-level debugging with `gdb` and `lldb`. Activated only when the `-g` CLI flag is specified.

**Key Files:**

| File | Responsibility |
|---|---|
| `mod.rs` | Entry point: accepts IR, source map, and machine code mappings; produces DWARF v4 sections |
| `dwarf.rs` | DWARF v4 core: compilation unit header emission, abbreviation table construction, DIE tree serialization with LEB128-encoded attributes |
| `line_program.rs` | Line number program generator: state machine opcode emission mapping machine code addresses to source file/line/column; standard opcodes (`DW_LNS_advance_pc`, `DW_LNS_advance_line`), special opcodes for compact encoding |
| `info.rs` | DIE generation: `DW_TAG_compile_unit`, `DW_TAG_subprogram`, `DW_TAG_variable`, `DW_TAG_formal_parameter`, `DW_TAG_base_type`, `DW_TAG_pointer_type`, `DW_TAG_structure_type`, `DW_TAG_array_type`, `DW_TAG_typedef` |
| `frame.rs` | Call Frame Information (CFI): `.debug_frame` section with CIE (Common Information Entry) and FDE (Frame Description Entry) for stack unwinding per architecture |

**Generated DWARF Sections:**

| Section | Content |
|---|---|
| `.debug_info` | Compilation unit DIEs, subprogram DIEs, variable DIEs, type DIEs, scope DIEs |
| `.debug_abbrev` | Abbreviation table defining DIE tag/attribute encodings |
| `.debug_line` | Line number program: source-to-address mappings via state machine |
| `.debug_str` | String table for DWARF attribute values |
| `.debug_aranges` | Address ranges for fast lookup of compilation units by address |
| `.debug_frame` | Call Frame Information for stack unwinding |
| `.debug_loc` | Location lists for variables with multiple locations |

**Input:** IR (for type and variable information), source map (for file/line mappings), machine code address mappings (from code generator).

**Output:** DWARF sections as byte vectors (`Vec<u8>`) with associated relocation entries, delivered to the linker for inclusion in the final ELF binary.

**Dependencies:** Imports from `ir` (type info, symbol info), `codegen` (address mappings), and `common` (source map, diagnostics).

> For detailed DWARF implementation notes, see [`docs/internals/dwarf.md`](internals/dwarf.md).

---

### 9. Common Module — `src/common/`

**Purpose:** Cross-cutting utilities shared by every pipeline phase. This is the foundation layer with zero internal dependencies — it imports only from the Rust standard library.

**Key Files:**

| File | Responsibility |
|---|---|
| `mod.rs` | Re-export all common types: `SourceLocation`, `Diagnostic`, `InternedString`, `Arena` |
| `diagnostics.rs` | `DiagnosticEmitter` implementing GCC-compatible `file:line:col: error: message` format on stderr; severity levels (error, warning, note); compilation fails with exit code 1 on any error |
| `source_map.rs` | `SourceMap` tracking file registry, byte-offset-to-line/column mapping, macro expansion chains, `#line` directive overrides |
| `intern.rs` | String interner using `HashMap<&str, InternId>` with arena-backed storage; returns compact `InternId` handles for O(1) equality comparison |
| `arena.rs` | Typed arena allocator for AST and IR nodes; bump allocation with O(1) per-allocation cost and batch deallocation |
| `numeric.rs` | Arbitrary-width integer representation for compile-time constant evaluation in the preprocessor and constant folder |

**Components in Detail:**

- **`DiagnosticEmitter`** — Every pipeline phase reports errors, warnings, and notes through this shared emitter. Output format matches GCC: `file:line:col: error: description`. The emitter accumulates diagnostics and signals whether any errors occurred, which the driver uses to determine the exit code.

- **`SourceMap`** — Maintains a registry of all source files processed during compilation. Maps byte offsets to line/column positions, tracks macro expansion chains (so diagnostics can show "in expansion of macro X"), and respects `#line` directive overrides. The DWARF line program generator also consumes the source map.

- **String Interning** — All identifier strings are interned at lexing time. The interner stores each unique string once in arena-backed memory and returns a compact `InternId` handle. Subsequent phases (parser, semantic analysis, IR, codegen) compare identifiers by handle rather than by string content, providing O(1) equality checks and reduced memory usage.

- **Arena Allocator** — AST and IR nodes are allocated from typed arenas using bump allocation. This provides O(1) allocation performance, excellent cache locality (nodes are contiguous in memory), and batch deallocation (the entire arena is freed at once when a compilation unit is complete). This design is critical for meeting the performance target of compiling SQLite in under 60 seconds.

- **Numeric** — Provides arbitrary-width integer arithmetic for compile-time constant evaluation. Used by the preprocessor expression evaluator and the optimizer's constant folding pass.

**Dependencies:** Imports only from `std`. This module is the dependency root of the entire project.

---

## Data Flow Contracts

Each phase boundary has a strict data flow contract specifying the types exchanged and the invariants guaranteed. Violating a contract indicates a bug in the producing phase.

| Boundary | Interface Type | Invariants Guaranteed |
|---|---|---|
| **Preprocessor → Lexer** | `String` (preprocessed source text) | All preprocessor directives consumed; all macros expanded; all includes resolved; output is syntactically valid C source ready for tokenization |
| **Lexer → Parser** | `Vec<Token>` | Every token classified unambiguously with `TokenKind`; each token carries accurate `SourceLocation` (file, line, column); source positions valid through macro expansions |
| **Parser → Semantic Analysis** | `TranslationUnit` (untyped AST) | AST is syntactically valid; source locations preserved on all nodes; GCC extension nodes (`__attribute__`, statement expressions, `typeof`, inline asm) represented in the tree |
| **Semantic Analysis → IR Builder** | Typed AST + `SymbolTable` | Every expression has a resolved `CType`; all identifiers bound to declarations; implicit conversion nodes inserted; type mismatches rejected as errors; symbol tables fully populated |
| **IR Builder → Optimizer** | `Module` (SSA-form IR) | IR preserves source program semantics; all IR values have types; CFG is well-formed (every block has a terminator, entry block has no predecessors); phi nodes placed at dominance frontiers |
| **Optimizer → Code Generator** | `Module` (optimized IR) | Optimized IR is semantically equivalent to input IR; dead code eliminated; constants folded; all IR invariants still hold |
| **Code Generator → Linker** | `ObjectCode` (machine code + metadata) | Machine code bytes are valid for the selected target architecture; relocations correctly describe all unresolved references; symbols include binding (local/global/weak) and visibility attributes; section assignments are correct |
| **Debug Info → Linker** | DWARF sections (`Vec<u8>`) + relocations | DWARF data conforms to version 4 encoding; section cross-references use proper relocation entries; line program state machine is well-formed |

---

## Module Dependency Graph

The following graph shows the import relationships between modules. Arrows point from the importing module to the module it imports. The `common` module sits at the root — it has no internal dependencies and is imported by every other module.

```
common ◄──────────────────────────────────────────────────┐
  (no internal deps; std only)                            │
  ▲                                                       │
  │                                                       │
  ├── frontend::lexer ◄─── frontend::preprocessor         │
  │         ▲                                             │
  │         │                                             │
  │         └── frontend::parser                          │
  │                   ▲                                   │
  │                   │                                   │
  │                   └── sema                            │
  │                         ▲                             │
  │                         │                             │
  │                         └── ir                        │
  │                               ▲                       │
  │                               │                       │
  │                               └── passes              │
  │                                     ▲                 │
  │                                     │                 │
  │                                     └── codegen ──► linker
  │                                                       ▲
  │                                          debug ───────┘
  │
  └── driver (imports ALL modules to orchestrate the pipeline)
```

**Dependency Rules:**
- Each module imports only from modules below it in the pipeline (earlier phases) and from `common`
- No circular dependencies exist — the graph is a strict DAG (directed acyclic graph)
- The `common` module is the sole foundation layer with zero internal imports
- The `driver` module is the sole top-level orchestrator, importing from every other module
- The `debug` module feeds into the `linker` (DWARF sections are embedded in the ELF output)
- The `codegen` module feeds into the `linker` (machine code and relocations)

**Detailed Import Table:**

| Module | Imports From | Key Types Imported |
|---|---|---|
| `common` | `std` only | Foundation layer |
| `frontend::lexer` | `common` | `SourceLocation`, `Diagnostic`, `InternedString` |
| `frontend::preprocessor` | `frontend::lexer`, `common` | `Token`, `TokenStream`, `SourceMap` |
| `frontend::parser` | `frontend::lexer`, `common` | `Token`, `TokenStream`, AST node types |
| `sema` | `frontend::parser`, `common` | AST types, `SymbolTable`, `CType` |
| `ir` | `sema`, `common` | Typed AST, `CType`, `SymbolTable` |
| `passes` | `ir` | IR types, `Instruction`, `BasicBlock`, `CFG` |
| `codegen` | `ir`, `passes`, `common` | Optimized IR, `TargetConfig` |
| `linker` | `codegen`, `common` | Machine code bytes, relocations, symbols |
| `debug` | `ir`, `codegen`, `common` | Source mappings, type info, address mappings |
| `driver` | All modules | All pipeline entry points |

---

## Cross-Cutting Concerns

Four shared services cut across all pipeline phases, providing consistent behavior throughout the compiler.

### Diagnostics

The `DiagnosticEmitter` (in `src/common/diagnostics.rs`) provides a unified interface for error and warning reporting. Every pipeline phase uses this emitter to report issues in GCC-compatible format:

```
file.c:42:17: error: implicit declaration of function 'foo'
file.c:42:17: note: did you mean 'bar'?
```

Diagnostic severity levels:
- **Error** — Compilation-blocking; causes exit code 1
- **Warning** — Non-blocking; compilation continues
- **Note** — Additional context attached to a preceding error or warning

### Source Map

The `SourceMap` (in `src/common/source_map.rs`) tracks:
- **File registry** — All source files opened during compilation, each assigned a unique file ID
- **Position mapping** — Byte offsets to line/column positions for accurate diagnostics
- **Macro expansion chains** — Records which macros expanded to produce a given token, enabling "in expansion of macro X" notes
- **`#line` overrides** — Respects `#line` directives that remap source positions

Consumers: diagnostics (for error locations), preprocessor (for include tracking), DWARF line program generator (for source-to-address mappings).

### String Interning

The string interner (in `src/common/intern.rs`) stores each unique identifier string once and returns a compact `InternId` handle. Benefits:
- **O(1) equality comparison** — Compare integer handles instead of string contents
- **Reduced memory usage** — Each unique string stored exactly once
- **Cache-friendly** — Handles are small integers; identity maps are compact

All identifiers are interned at lexing time. The parser, semantic analyzer, IR builder, and code generator reference identifiers by their interned handle.

### Target Configuration

The `TargetConfig` struct (constructed in `src/driver/target.rs`) propagates target-specific parameters through the entire pipeline:

| Parameter | Purpose | Consumers |
|---|---|---|
| Pointer width | Size of pointer types (4 or 8 bytes) | Semantic analysis, IR builder |
| `long` size | Size of `long` type (4 or 8 bytes) | Semantic analysis |
| Endianness | Byte order (little-endian for all current targets) | Code generator, linker |
| ABI | Calling convention parameters | Code generator |
| ELF class | ELF32 or ELF64 | Linker |
| Relocation types | Architecture-specific relocation set | Linker |
| Register file | Available registers and classifications | Code generator, register allocator |

---

## Target-Specific Data Flow

The `--target` flag creates a branching point where target-specific parameters flow through the entire pipeline. A single `TargetConfig` struct, constructed from the target triple, carries all architecture-specific settings.

```
CLI: --target <triple>
         │
         ▼
   ┌─────────────┐
   │ TargetConfig │
   └──────┬──────┘
          │
    ┌─────┼──────────────┬──────────────┐
    ▼     ▼              ▼              ▼
  Sema  Codegen        Linker        Debug
  (type  (instruction   (ELF format,  (DWARF register
  sizes) set, ABI)      relocations)  mappings)
```

**Per-Target Configuration:**

| Target Triple | Pointer | `long` | ELF Class | Endian | Register File |
|---|---|---|---|---|---|
| `x86_64-linux-gnu` | 8 bytes | 8 bytes | ELF64 | Little | `rax`–`r15`, `xmm0`–`xmm15` |
| `i686-linux-gnu` | 4 bytes | 4 bytes | ELF32 | Little | `eax`–`edi`, `xmm0`–`xmm7` |
| `aarch64-linux-gnu` | 8 bytes | 8 bytes | ELF64 | Little | `x0`–`x30`, `v0`–`v31` |
| `riscv64-linux-gnu` | 8 bytes | 8 bytes | ELF64 | Little | `x0`–`x31`, `f0`–`f31` |

The target configuration influences every level of the compiler:
- **Semantic analysis** uses pointer and `long` sizes to compute `sizeof` results and validate type compatibility
- **Code generation** selects the correct instruction set, ABI calling convention, and register allocation strategy
- **Linker** emits the correct ELF class (32 or 64-bit), applies the correct relocation types, and locates the appropriate CRT objects
- **Debug info** uses architecture-specific DWARF register number mappings

> For detailed per-architecture documentation, see [`docs/targets.md`](targets.md).

---

## Build and Runtime Architecture

### Build Configuration

- **Language:** Rust 2021 edition
- **Dependencies:** Zero external crates — `[dependencies]` is empty in `Cargo.toml`
- **Release profile:** `opt-level = 3` for maximum compiler performance
- **Build script:** `build.rs` embeds bundled freestanding header paths and generates target-detection constants

### Runtime Architecture

The compiler is a **single statically-linked binary** (`bcc`) with all functionality compiled in:

- **No external tool invocations** — The preprocessor, lexer, parser, semantic analyzer, IR generator, optimizer, code generator (integrated assembler), and linker all execute within the single `bcc` process
- **Bundled headers** — Nine freestanding C headers (`stddef.h`, `stdint.h`, `stdarg.h`, `stdbool.h`, `limits.h`, `float.h`, `stdalign.h`, `stdnoreturn.h`, `iso646.h`) are embedded in the binary or resolved from the installation `include/` directory
- **Filesystem interactions** — The only I/O operations are:
  - Reading input C source files
  - Reading system CRT objects (`crt1.o`, `crti.o`, `crtn.o`) and libraries (`libc.a`, `libc.so`) from the target system's library paths
  - Writing the output binary (executable, shared library, or relocatable object)

### CRT Object Locations

| Target | CRT Path | Library Path |
|---|---|---|
| x86-64 | `/usr/lib/x86_64-linux-gnu/` | `/usr/lib/x86_64-linux-gnu/` |
| i686 | `/usr/i686-linux-gnu/lib/` | `/usr/i686-linux-gnu/lib/` |
| AArch64 | `/usr/aarch64-linux-gnu/lib/` | `/usr/aarch64-linux-gnu/lib/` |
| RISC-V 64 | `/usr/riscv64-linux-gnu/lib/` | `/usr/riscv64-linux-gnu/lib/` |

---

## Performance Architecture

The compiler is designed to meet the performance target of compiling the SQLite amalgamation (~230,000 lines of C) in under 60 seconds on a single core at `-O0`, with peak resident set size not exceeding 2 GB.

### Memory Management Strategy

| Technique | Module | Impact |
|---|---|---|
| **Arena allocation** | `common/arena.rs` | AST and IR nodes allocated via bump allocation with O(1) cost; batch deallocation eliminates per-node `free` overhead; contiguous memory improves cache locality |
| **String interning** | `common/intern.rs` | Each unique identifier stored once; subsequent references use compact integer handles; eliminates redundant string allocations across 230K+ lines |
| **Streaming pipeline** | `driver/pipeline.rs` | Each phase processes and produces its output sequentially; intermediate results are released as soon as the next phase has consumed them |

### Computational Strategy

| Technique | Module | Impact |
|---|---|---|
| **O(n) tokenization** | `frontend/lexer/` | Single-pass tokenization with direct character classification |
| **Recursive descent parsing** | `frontend/parser/` | Single-pass parsing with O(1) lookahead for most constructs |
| **Linear scan register allocation** | `codegen/regalloc.rs` | O(n) register allocation instead of O(n²) graph coloring at `-O0` and `-O1` |
| **Pass iteration bound** | `passes/pipeline.rs` | `-O2` pass pipeline iterates to fixed point with a bounded iteration count |

### Performance Budget (SQLite at `-O0`)

| Phase | Estimated Time Share | Notes |
|---|---|---|
| Preprocessing | ~10% | I/O bound; include resolution, macro expansion |
| Lexing | ~5% | CPU bound; character-by-character scanning |
| Parsing | ~15% | CPU bound; AST construction with arena allocation |
| Semantic Analysis | ~15% | CPU bound; symbol table lookups, type checking |
| IR Generation | ~15% | CPU bound; AST-to-IR lowering with SSA construction |
| Code Generation | ~25% | CPU bound; instruction selection, register allocation, encoding |
| Linking | ~15% | I/O + CPU; CRT reading, relocation, ELF emission |

---

## Security Hardening (x86-64)

The x86-64 backend supports three security hardening features, all implemented in `src/codegen/x86_64/security.rs`:

### Retpoline (`-mretpoline`)

Replaces speculative-execution-vulnerable indirect branches (`jmp *%rax`, `call *%rax`) with retpoline thunk sequences that prevent Spectre v2 attacks. The compiler generates a `__x86_retpoline_rax` thunk and redirects all indirect calls/jumps through it.

### CET `endbr64` (`-fcf-protection`)

Inserts Intel Control-flow Enforcement Technology `endbr64` instructions at all indirect branch targets — function entry points and any address taken via function pointers or computed goto. This enables hardware-enforced indirect branch tracking.

### Stack Probing

For stack frames exceeding one page (4096 bytes), the code generator emits a probe loop that touches each page between the current stack pointer and the new frame boundary. This prevents skipping over the stack guard page and ensures the OS can grow the stack correctly.

---

## Related Documentation

| Document | Description |
|---|---|
| [`docs/cli.md`](cli.md) | Complete CLI reference with all supported flags and usage examples |
| [`docs/targets.md`](targets.md) | Detailed per-architecture documentation: ABI, ELF format, QEMU testing |
| [`docs/internals/ir.md`](internals/ir.md) | IR design: instruction set, type system, SSA form, CFG representation |
| [`docs/internals/linker.md`](internals/linker.md) | Linker internals: ELF format handling, relocation processing, CRT linkage |
| [`docs/internals/dwarf.md`](internals/dwarf.md) | DWARF v4 implementation: section layouts, DIE schemas, line program encoding |
