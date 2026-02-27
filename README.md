# bcc — Blitzy C Compiler

A complete, self-contained C11 compiler written in pure Rust targeting Linux ELF output across four processor architectures — x86-64, i686, AArch64, and RISC-V 64 — with an integrated preprocessor, assembler, and linker. The entire compilation pipeline executes within a single binary with **zero external dependencies** beyond the Rust standard library and **zero external toolchain invocations**.

## Key Differentiators

- **Zero external crate dependencies** — built exclusively on Rust `std`
- **No external toolchain invocations** — no `as`, `ld`, `gcc`, or `clang` required
- **Four architecture backends** — x86-64, i686, AArch64, RISC-V 64
- **Integrated ELF linker** — reads system CRT objects and `ar` archives directly
- **DWARF v4 debug information** — source-level debugging with `gdb` and `lldb`
- **GCC-compatible CLI** — drop-in replacement for common `gcc` invocations

---

## Supported Targets

| Target Triple | Architecture | ELF Class | ABI | Status |
|---|---|---|---|---|
| `x86_64-linux-gnu` | x86-64 | ELF64 | System V AMD64 | Primary |
| `i686-linux-gnu` | i686 | ELF32 | cdecl / System V i386 | Supported |
| `aarch64-linux-gnu` | AArch64 | ELF64 | AAPCS64 | Supported |
| `riscv64-linux-gnu` | RISC-V 64 | ELF64 | LP64D | Supported |

Each target includes full ABI compliance, architecture-specific instruction encoding (integrated assembler), and correct ELF relocation processing. Cross-architecture binaries can be tested via QEMU user-mode emulation on an x86-64 host.

---

## Build Instructions

### Prerequisites

- **Rust stable** (edition 2021) — install via [rustup](https://rustup.rs/)
- **Linux host** — the compiler targets Linux ELF output exclusively
- No external crates are required; the `[dependencies]` section in `Cargo.toml` is intentionally empty

### Build

```bash
# Debug build
cargo build

# Release build (optimized, recommended)
cargo build --release
```

The release binary is located at `target/release/bcc`.

### Verify

```bash
# Run the full test suite
cargo test

# Run clippy lint checks
cargo clippy

# Check formatting
cargo fmt --check
```

### Cross-Architecture Testing Dependencies

To run validation tests against non-native architectures, install the following system packages:

```bash
sudo apt-get install -y \
    qemu-user-static \
    libc6-dev \
    libc6-dev-i386-cross \
    libc6-dev-arm64-cross \
    libc6-dev-riscv64-cross
```

---

## Usage Examples

### Basic Compilation

Compile a C source file to an executable:

```bash
bcc hello.c -o hello
```

### Cross-Compilation

Target a different architecture using `--target`:

```bash
bcc --target aarch64-linux-gnu hello.c -o hello
bcc --target i686-linux-gnu hello.c -o hello
bcc --target riscv64-linux-gnu hello.c -o hello
```

### Compile to Object File

Produce a relocatable object file without linking:

```bash
bcc -c source.c -o source.o
```

### Shared Library

Build a position-independent shared library:

```bash
bcc -shared -fPIC lib.c -o lib.so
```

### Debug Information

Emit DWARF v4 debug sections for source-level debugging:

```bash
bcc -g debug.c -o debug
gdb ./debug
```

### Optimization Levels

Select an optimization level from `-O0` (none) through `-O2` (aggressive):

```bash
bcc -O0 unoptimized.c -o unoptimized
bcc -O1 basic_opt.c -o basic_opt
bcc -O2 optimized.c -o optimized
```

### Security Hardening (x86-64)

Enable retpoline sequences for speculative execution mitigation and Intel CET instrumentation:

```bash
bcc -mretpoline -fcf-protection secure.c -o secure
```

### Preprocessor Options

Define and undefine macros, add include search paths:

```bash
bcc -DNDEBUG -DVERSION=2 -I./include -Umacro source.c -o output
```

### Linking Options

Specify library search paths and libraries to link:

```bash
bcc -L/usr/local/lib -lm -lpthread source.c -o output
```

### Static Linking

Force static linking of all libraries:

```bash
bcc -static source.c -o output
```

---

## CLI Reference

`bcc` supports GCC-compatible command-line flags for seamless integration with existing build systems.

### Compilation Control

| Flag | Description |
|---|---|
| `-c` | Compile to relocatable object file; do not link |
| `-o <file>` | Write output to `<file>` |
| `-static` | Produce a statically linked executable |
| `-shared` | Produce a shared library (`.so`) |

### Preprocessor Flags

| Flag | Description |
|---|---|
| `-I <dir>` | Add `<dir>` to the include search path |
| `-D <macro>[=value]` | Define preprocessor macro `<macro>` with optional `value` |
| `-U <macro>` | Undefine preprocessor macro `<macro>` |

### Linker Flags

| Flag | Description |
|---|---|
| `-L <dir>` | Add `<dir>` to the library search path |
| `-l <lib>` | Link against library `lib<lib>.a` or `lib<lib>.so` |

### Code Generation Flags

| Flag | Description |
|---|---|
| `-g` | Emit DWARF v4 debug information |
| `-O0` | No optimization (default) |
| `-O1` | Basic optimizations (mem2reg, constant folding, DCE) |
| `-O2` | Aggressive optimizations (adds CSE, algebraic simplification) |
| `-fPIC` | Generate position-independent code |

### Security Hardening Flags (x86-64)

| Flag | Description |
|---|---|
| `-mretpoline` | Use retpoline sequences for indirect branches (Spectre v2 mitigation) |
| `-fcf-protection` | Insert Intel CET `endbr64` instructions at indirect branch targets |

### Target Selection

| Flag | Description |
|---|---|
| `--target <triple>` | Select target architecture (e.g., `x86_64-linux-gnu`, `aarch64-linux-gnu`) |

For the complete CLI reference with detailed descriptions and examples, see [`docs/cli.md`](docs/cli.md).

---

## Architecture Overview

`bcc` implements a sequential compilation pipeline where each phase transforms its input and passes the result to the next stage:

```
                         ┌─────────────────────────────────────────────────┐
                         │              bcc Compilation Pipeline           │
                         └─────────────────────────────────────────────────┘

  ┌──────────┐    ┌──────────────┐    ┌───────┐    ┌────────┐    ┌──────┐
  │  Source   │───▶│ Preprocessor │───▶│ Lexer │───▶│ Parser │───▶│ Sema │
  │  (.c)    │    │              │    │       │    │        │    │      │
  └──────────┘    └──────────────┘    └───────┘    └────────┘    └──┬───┘
                                                                    │
                                                                    ▼
  ┌──────────┐    ┌────────┐    ┌───────────┐    ┌────┐    ┌──────────────┐
  │   ELF    │◀───│ Linker │◀───│  Codegen  │◀───│ Opt│◀───│  IR Builder  │
  │ Binary   │    │        │    │ (4 archs) │    │    │    │              │
  └──────────┘    └────────┘    └───────────┘    └────┘    └──────────────┘
```

### Module Layout

| Directory | Purpose |
|---|---|
| `src/frontend/` | Preprocessor, Lexer, Parser — C11 with GCC extensions |
| `src/sema/` | Semantic analysis, type checking, symbol tables |
| `src/ir/` | SSA-form intermediate representation |
| `src/passes/` | Optimization passes — constant folding, DCE, CSE, mem2reg, simplification |
| `src/codegen/` | Four architecture backends with integrated assemblers |
| `src/codegen/x86_64/` | x86-64 backend — System V AMD64 ABI, security hardening |
| `src/codegen/i686/` | i686 backend — cdecl / System V i386 ABI |
| `src/codegen/aarch64/` | AArch64 backend — AAPCS64 ABI |
| `src/codegen/riscv64/` | RISC-V 64 backend — LP64D ABI |
| `src/linker/` | Integrated ELF linker — symbol resolution, relocation, section merging |
| `src/debug/` | DWARF v4 debug information generation |
| `src/common/` | Shared utilities — diagnostics, source map, string interning, arena allocator |
| `src/driver/` | CLI parsing, target configuration, pipeline orchestration |
| `include/` | Bundled freestanding C headers (9 headers) |

### Data Flow

Each pipeline phase transforms its input into a well-defined output type:

| Phase | Input | Output |
|---|---|---|
| Preprocessor | Source text + `-I`/`-D`/`-U` options | Expanded source text |
| Lexer | Expanded source text | `Vec<Token>` with source locations |
| Parser | Token stream | Untyped AST (`TranslationUnit`) |
| Semantic Analyzer | Untyped AST | Typed AST with resolved symbols |
| IR Builder | Typed AST | SSA-form IR with basic blocks |
| Optimizer | SSA IR | Optimized SSA IR |
| Code Generator | Optimized IR + target config | Machine code bytes + relocations |
| Linker | Object code + CRT objects + libraries | ELF executable/shared library |

For detailed architecture documentation, see [`docs/architecture.md`](docs/architecture.md).

---

## Features

### C11 Standard Compliance with GCC Extensions

`bcc` implements the full C11 standard including:

- All C11 keywords: `_Alignas`, `_Alignof`, `_Atomic`, `_Bool`, `_Complex`, `_Generic`, `_Noreturn`, `_Static_assert`, `_Thread_local`
- Complete operator precedence (15 levels) with all expression forms
- Struct, union, enum, typedef, function pointers, flexible array members
- Designated initializers, compound literals, variadic functions

GCC language extensions supported:

- `__attribute__((...))` — packed, aligned, section, unused, deprecated, visibility, format
- Statement expressions — `({ ... })`
- `typeof` / `__typeof__` — type inference
- Computed goto — `&&label` and `goto *ptr`
- Inline assembly — `asm`/`__asm__` with operand constraints
- `__extension__` — suppress pedantic warnings
- `__builtin_*` intrinsics

### Four Architecture Backends

Each backend implements the full `CodeGen` trait with:

- **Instruction selection** — IR-to-machine-instruction pattern matching
- **Integrated assembler** — direct machine code encoding (no external `as`)
- **Register allocation** — linear scan allocator parameterized by target register file
- **ABI compliance** — correct calling conventions per architecture

### Integrated ELF Linker

The linker operates entirely within the `bcc` process:

- Reads ELF relocatable objects (`.o`) and `ar` static archives (`.a`)
- Resolves symbols across compilation units, CRT objects, and libraries
- Applies architecture-specific relocations (`R_X86_64_*`, `R_386_*`, `R_AARCH64_*`, `R_RISCV_*`)
- Merges and lays out sections (`.text`, `.rodata`, `.data`, `.bss`)
- Emits ELF32 (i686) or ELF64 (x86-64, AArch64, RISC-V 64) output
- Supports three output modes: static executables, shared libraries (`-shared`), and relocatable objects (`-c`)
- Generates PLT/GOT stubs and `.dynamic` sections for shared libraries

### DWARF v4 Debug Information

When compiled with `-g`, `bcc` generates DWARF version 4 debug sections:

- `.debug_info` — compilation units, subprograms, variables, parameters, and type DIEs
- `.debug_abbrev` — abbreviation table for DIE encoding
- `.debug_line` — line number program mapping addresses to source locations
- `.debug_str` — string table for debug information
- `.debug_frame` — Call Frame Information (CFI) for stack unwinding
- `.debug_aranges` — address range lookup table

Generated binaries are compatible with `gdb` and `lldb` for source-level debugging.

### Optimization Pipeline

| Level | Passes | Description |
|---|---|---|
| `-O0` | None | No optimization; fastest compilation |
| `-O1` | mem2reg, constant folding, DCE | Basic optimizations for reasonable code quality |
| `-O2` | All `-O1` passes + CSE, algebraic simplification | Aggressive optimization with iterative fixed-point |

### x86-64 Security Hardening

- **Retpoline** (`-mretpoline`) — replaces indirect `jmp`/`call` instructions with retpoline thunk sequences to mitigate Spectre v2
- **Intel CET** (`-fcf-protection`) — inserts `endbr64` instructions at function entries and indirect branch targets for Control-flow Enforcement Technology
- **Stack probing** — automatically generates page-straddling probe sequences for stack frames exceeding 4096 bytes

### Bundled Freestanding Headers

Nine C standard freestanding headers are shipped with the compiler and automatically resolved by the preprocessor:

| Header | Contents |
|---|---|
| `stddef.h` | `size_t`, `ptrdiff_t`, `NULL`, `offsetof`, `max_align_t` |
| `stdint.h` | Fixed-width integer types (`int8_t` through `int64_t`, `uintptr_t`, etc.) |
| `stdarg.h` | `va_list`, `va_start`, `va_arg`, `va_end`, `va_copy` |
| `stdbool.h` | `bool`, `true`, `false` |
| `limits.h` | Integer limits (`INT_MAX`, `LONG_MAX`, etc.) — target-width-adaptive |
| `float.h` | Floating-point characteristics (`FLT_MAX`, `DBL_EPSILON`, etc.) |
| `stdalign.h` | `alignas`, `alignof` macros |
| `stdnoreturn.h` | `noreturn` macro |
| `iso646.h` | Alternative operator spellings (`and`, `or`, `not`, `xor`, etc.) |

### GCC-Compatible Diagnostics

All error and warning messages follow GCC-compatible format on stderr:

```
source.c:42:10: error: implicit declaration of function 'foo'
source.c:42:10: note: did you mean 'bar'?
```

The process exits with code 1 on any compilation error.

---

## Validation Suite

`bcc` is validated against real-world open-source C codebases to ensure correctness and performance across all four target architectures.

### Validation Targets

| Project | LOC | Validation Scope |
|---|---|---|
| **SQLite** | ~230K | Compile + run test suite |
| **Lua** | ~30K | Compile + run test suite |
| **zlib** | ~15K | Compile + run test suite |
| **Redis** | ~150K | Compile-only verification |

### Performance Targets

- **SQLite amalgamation** must compile in **under 60 seconds** on a single core at `-O0`
- Peak resident set size must not exceed **2 GB** during SQLite compilation

### Cross-Architecture Testing

Non-native architecture binaries are executed via **QEMU user-mode emulation** (`qemu-user-static`), enabling full test suite execution for i686, AArch64, and RISC-V 64 targets on an x86-64 host.

### Running the Validation Suite

```bash
# Run all validation tests
cargo test --test validation

# Individual validation targets
cargo test --test sqlite
cargo test --test lua
cargo test --test zlib
cargo test --test redis
```

The CI validation pipeline is defined in [`.github/workflows/validation.yml`](.github/workflows/validation.yml).

---

## CRT Object Locations

The integrated linker locates system CRT startup objects and C library archives from the following paths:

| Target | CRT Objects | C Library |
|---|---|---|
| x86-64 | `/usr/lib/x86_64-linux-gnu/crt{1,i,n}.o` | `/usr/lib/x86_64-linux-gnu/libc.a` |
| i686 | `/usr/i686-linux-gnu/lib/crt{1,i,n}.o` | `/usr/i686-linux-gnu/lib/libc.a` |
| AArch64 | `/usr/aarch64-linux-gnu/lib/crt{1,i,n}.o` | `/usr/aarch64-linux-gnu/lib/libc.a` |
| RISC-V 64 | `/usr/riscv64-linux-gnu/lib/crt{1,i,n}.o` | `/usr/riscv64-linux-gnu/lib/libc.a` |

Library search paths can be extended with `-L <dir>` flags.

---

## Documentation

| Document | Description |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Compiler architecture overview — pipeline stages, module relationships, data flow |
| [`docs/targets.md`](docs/targets.md) | Supported target architectures — ABI details, ELF format specifics, QEMU testing |
| [`docs/cli.md`](docs/cli.md) | Complete CLI reference — all supported flags with descriptions and examples |
| [`docs/internals/ir.md`](docs/internals/ir.md) | IR design — instruction set, type system, SSA form |
| [`docs/internals/linker.md`](docs/internals/linker.md) | Linker internals — ELF format handling, relocation processing, CRT linkage |
| [`docs/internals/dwarf.md`](docs/internals/dwarf.md) | DWARF v4 implementation notes — section layouts, DIE schemas, line program encoding |

---

## Design Constraints

- **Zero external crate dependencies** — the `[dependencies]` section in `Cargo.toml` is empty; every feature is implemented using only `std`
- **No external toolchain invocations** — no subprocess calls to `as`, `ld`, `gcc`, `clang`, or any other binary during compilation
- **Documented unsafe code** — every `unsafe` block carries an inline comment explaining the invariant, why a safe abstraction is insufficient, and the scope of the unsafe region
- **Linux-only targets** — all four architectures target Linux ELF; no PE/COFF or Mach-O support

---

## Project Info

**Package name:** `bcc`
**Language:** Rust (edition 2021)
**Build system:** Cargo
**Binary output:** `target/release/bcc`

### CI/CD

- **Build and test pipeline:** [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — build, unit tests, integration tests, clippy lint, rustfmt check
- **Validation pipeline:** [`.github/workflows/validation.yml`](.github/workflows/validation.yml) — compile and test SQLite, Lua, zlib, Redis across all architectures
