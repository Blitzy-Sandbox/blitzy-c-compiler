# CLI Reference

`bcc` is a self-contained C11 compiler written in pure Rust that supports GCC-compatible
command-line flags. It produces native ELF binaries for four processor architectures
without invoking any external toolchain components.

## Synopsis

```
bcc [options] <input-files...>
```

The compiler reads one or more C source files, preprocesses, compiles, and links them
into an output binary. By default the output is a statically linked ELF executable
named `a.out`. Use the flags documented below to control every stage of compilation.

All flag parsing is implemented in [`src/driver/cli.rs`](../src/driver/cli.rs).
For architectural details see [Architecture Overview](architecture.md).
For target-specific information see [Supported Targets](targets.md).

---

## Compilation Mode Flags

### `-c`

Compile each source file to a relocatable ELF object file without linking. The output
is one `.o` file per input. When combined with `-o`, the output path applies to the
single resulting object.

```bash
bcc -c main.c          # produces main.o
bcc -c util.c -o obj/util.o
```

### `-o <path>`

Specify the output file path explicitly.

| Context | Default when `-o` is omitted |
|---|---|
| Executable (default mode) | `a.out` |
| Object file (`-c`) | Input stem + `.o` (e.g. `foo.c` → `foo.o`) |
| Shared library (`-shared`) | `a.out` |

```bash
bcc hello.c -o hello
bcc -c main.c -o build/main.o
bcc -shared -fPIC lib.c -o libfoo.so
```

### `-shared`

Produce an ELF shared object (dynamic library) instead of an executable. This flag
should be combined with `-fPIC` to ensure position-independent code generation. The
linker emits `.dynamic`, `.dynsym`, `.dynstr`, `.plt`, and `.got` sections as needed.

```bash
bcc -shared -fPIC mylib.c -o libmylib.so
```

### `-static`

Produce a statically linked executable. All library references are resolved at link
time from static archives (`.a` files). This is the **default linking mode** — the
flag is accepted for GCC compatibility but does not change behavior unless `-shared`
was previously specified on the same command line.

```bash
bcc -static program.c -o program
```

---

## Preprocessor Flags

### `-I <dir>`

Add `<dir>` to the list of directories searched when resolving `#include` directives.
This flag may be specified multiple times; directories are searched in the order they
appear on the command line, **before** the bundled freestanding header path and any
system header paths.

```bash
bcc -I./include -I../common/include app.c -o app
```

### `-D <macro>[=<value>]`

Define a preprocessor macro before compilation begins. If `=<value>` is omitted the
macro is defined with a replacement value of `1`.

```bash
bcc -DNDEBUG main.c -o main            # #define NDEBUG 1
bcc -DVERSION=2 -DBUILD_TYPE=\"release\" app.c -o app
```

### `-U <macro>`

Undefine a preprocessor macro. This removes a macro that was previously defined via
`-D` or a built-in default definition. `-U` is processed after all `-D` options.

```bash
bcc -DFOO -UFOO test.c -o test         # FOO is undefined during compilation
```

---

## Linker Flags

### `-L <dir>`

Add `<dir>` to the list of directories searched when resolving `-l` library options.
This flag may be specified multiple times; directories are searched in the order they
appear on the command line, before the default system library paths.

```bash
bcc -L/usr/local/lib -L./lib app.c -lmylib -o app
```

### `-l <library>`

Link against the specified library. The integrated linker searches for
`lib<library>.a` (static archive) or `lib<library>.so` (shared object) in the
library search paths established by `-L` flags and the default system library
directories.

When `-static` is in effect (the default), only `.a` archives are considered. When
linking a shared library with `-shared`, both `.a` and `.so` files are candidates.

```bash
bcc app.c -lm -lpthread -o app         # links libm.a and libpthread.a
bcc -L/opt/lib app.c -lcustom -o app
```

---

## Code Generation Flags

### `-fPIC`

Generate position-independent code suitable for inclusion in shared libraries.
When this flag is active the code generator emits GOT-relative addressing for
global data accesses and PLT stubs for external function calls across all four
supported architectures.

`-fPIC` is **required** when producing shared libraries with `-shared`.

```bash
bcc -fPIC -shared mylib.c -o libmylib.so
```

### `-mretpoline`

**(x86-64 only)** Generate retpoline sequences for all indirect calls and jumps as
a mitigation against Spectre variant 2 (Branch Target Injection). Indirect branches
such as `jmp *%rax` are replaced with speculative-execution-safe retpoline thunks
(`__x86_retpoline_rax` and similar). The compiler automatically emits the required
thunk functions into the `.text` section.

This flag is silently ignored on non-x86-64 targets.

```bash
bcc -mretpoline server.c -o server
```

### `-fcf-protection`

**(x86-64 only)** Instrument all indirect branch targets with Intel Control-flow
Enforcement Technology (CET) `endbr64` instructions. When enabled, `endbr64` is
inserted at:

- Every function entry point
- Every indirect branch target (e.g., computed goto destinations, function pointer
  call targets)

This flag is silently ignored on non-x86-64 targets.

```bash
bcc -fcf-protection main.c -o main
```

---

## Debug and Optimization Flags

### `-g`

Generate DWARF version 4 debug information. The following ELF sections are produced:

| Section | Content |
|---|---|
| `.debug_info` | Compilation units, subprograms, variables, parameters, and type DIEs |
| `.debug_abbrev` | Abbreviation table for `.debug_info` encoding |
| `.debug_line` | Source file/line/column to machine address mappings |
| `.debug_str` | String table for debug info string references |
| `.debug_aranges` | Address ranges for fast compilation unit lookup |
| `.debug_frame` | Call Frame Information (CFI) for stack unwinding |
| `.debug_loc` | Location lists for variables with varying locations |

The generated debug information is sufficient for source-level debugging with
`gdb` and `lldb`, including setting breakpoints by file and line, inspecting
local variables, and stepping through source code.

```bash
bcc -g source.c -o debug_binary
gdb ./debug_binary
```

### `-O0`

No optimization (**default**). The compiler runs no optimization passes, producing
the fastest compilation times but the largest and slowest output code. This level is
recommended for debug builds when combined with `-g`.

```bash
bcc -O0 -g app.c -o app_debug
```

### `-O1`

Basic optimization. The following optimization passes are applied:

| Pass | Effect |
|---|---|
| `mem2reg` | Promotes stack allocations to SSA registers where possible |
| `constant_fold` | Evaluates constant arithmetic expressions at compile time |
| `dce` | Removes dead code and unreachable basic blocks |

```bash
bcc -O1 app.c -o app
```

### `-O2`

Aggressive optimization. Includes all `-O1` passes plus additional passes, iterated
to a fixed point:

| Pass | Effect |
|---|---|
| `mem2reg` | Promotes stack allocations to SSA registers |
| `constant_fold` | Evaluates constant expressions at compile time |
| `dce` | Removes dead code and unreachable blocks |
| `cse` | Eliminates common subexpressions via value numbering |
| `simplify` | Algebraic simplification and strength reduction (e.g. `x * 2` → `x << 1`) |

The pass pipeline iterates until no further changes are made (fixed-point convergence).

```bash
bcc -O2 app.c -o app_release
```

---

## Target Selection

### `--target <triple>`

Select the target architecture for compilation. This flag determines the code
generator, ABI conventions, ELF format (32-bit or 64-bit), relocation types, and
register allocation strategy. When `--target` is omitted, the compiler defaults to
the host architecture.

#### Supported Target Triples

| Triple | Architecture | ABI | ELF Class | Pointer Size |
|---|---|---|---|---|
| `x86_64-linux-gnu` | x86-64 | System V AMD64 | ELF64 | 8 bytes |
| `i686-linux-gnu` | i686 | cdecl / System V i386 | ELF32 | 4 bytes |
| `aarch64-linux-gnu` | AArch64 | AAPCS64 | ELF64 | 8 bytes |
| `riscv64-linux-gnu` | RISC-V 64 | LP64D | ELF64 | 8 bytes |

#### Cross-Compilation

Cross-compiled binaries can be executed on the host using QEMU user-mode emulation
(requires `qemu-user-static`):

```bash
# Compile for AArch64
bcc --target aarch64-linux-gnu hello.c -o hello_arm64

# Run via QEMU
qemu-aarch64-static ./hello_arm64
```

#### CRT Object Locations

The integrated linker automatically locates system CRT startup objects (`crt1.o`,
`crti.o`, `crtn.o`) and the C standard library archive for each target:

| Target | CRT Path |
|---|---|
| `x86_64-linux-gnu` | `/usr/lib/x86_64-linux-gnu/` |
| `i686-linux-gnu` | `/usr/i686-linux-gnu/lib/` |
| `aarch64-linux-gnu` | `/usr/aarch64-linux-gnu/lib/` |
| `riscv64-linux-gnu` | `/usr/riscv64-linux-gnu/lib/` |

For details on per-target ABI conventions, register usage, and ELF specifics, see
[Supported Targets](targets.md).

---

## Usage Examples

### Basic Compilation

Compile a single source file into a statically linked executable:

```bash
bcc hello.c -o hello
```

### Compile to Object File

Compile without linking to produce a relocatable `.o` file:

```bash
bcc -c main.c
bcc -c util.c
bcc main.o util.o -o program
```

### Include Paths and Macro Definitions

Pass include search directories and preprocessor definitions:

```bash
bcc -I./include -I../shared -DNDEBUG -DVERSION=3 -O2 app.c -o app
```

### Cross-Compilation

Target a different architecture:

```bash
bcc --target aarch64-linux-gnu program.c -o program_arm64
bcc --target riscv64-linux-gnu program.c -o program_riscv
bcc --target i686-linux-gnu program.c -o program_x86
```

### Shared Library

Build a shared library with position-independent code:

```bash
bcc -shared -fPIC mylib.c -o libmylib.so
```

### Debug Build

Generate DWARF v4 debug information for source-level debugging:

```bash
bcc -g source.c -o debug_binary
gdb ./debug_binary
```

### Security-Hardened Build (x86-64)

Enable retpoline and CET instrumentation for security-sensitive code:

```bash
bcc -mretpoline -fcf-protection secure.c -o secure
```

### Linking with Libraries

Specify library search paths and libraries to link against:

```bash
bcc -L/usr/local/lib -lm -lpthread app.c -o app
```

### Combined Example

A complex real-world invocation combining multiple flags:

```bash
bcc -I./include -DNDEBUG -O2 -g \
    -L./lib -lutil \
    --target x86_64-linux-gnu \
    -mretpoline -fcf-protection \
    main.c util.c -o myapp
```

---

## Diagnostic Output

`bcc` emits all diagnostic messages to **stderr** in GCC-compatible format. The
general format is:

```
<file>:<line>:<column>: <severity>: <message>
```

### Severity Levels

| Severity | Description | Effect on Compilation |
|---|---|---|
| `error` | A condition that prevents successful compilation | Compilation fails; exit code 1 |
| `warning` | A potential issue that does not block compilation | Compilation continues; exit code 0 |
| `note` | Additional context for a preceding error or warning | Informational only |

### Example Diagnostic Output

```
main.c:10:5: error: use of undeclared identifier 'foo'
main.c:10:5: note: did you mean 'bar'?
util.c:25:12: warning: implicit conversion loses precision
```

### Exit Codes

| Code | Meaning |
|---|---|
| `0` | Compilation succeeded (warnings may have been emitted) |
| `1` | Compilation failed (one or more errors were emitted) |

All diagnostic formatting is implemented in
[`src/common/diagnostics.rs`](../src/common/diagnostics.rs).

---

## Environment and Defaults

### Bundled Freestanding Headers

`bcc` ships nine freestanding C standard headers that are always available without
any `-I` configuration. These headers are resolved automatically by the preprocessor
before system header paths:

| Header | Contents |
|---|---|
| `stddef.h` | `size_t`, `ptrdiff_t`, `NULL`, `offsetof`, `max_align_t` |
| `stdint.h` | Fixed-width integer types (`int8_t` – `int64_t`, `uintptr_t`, etc.) |
| `stdarg.h` | `va_list`, `va_start`, `va_arg`, `va_end`, `va_copy` |
| `stdbool.h` | `bool`, `true`, `false` |
| `limits.h` | Integer limits (`INT_MAX`, `LONG_MAX`, etc.) — target-width-adaptive |
| `float.h` | Floating-point characteristics (`FLT_MAX`, `DBL_EPSILON`, etc.) |
| `stdalign.h` | `alignas`, `alignof` macros |
| `stdnoreturn.h` | `noreturn` macro |
| `iso646.h` | Alternative operator spellings (`and`, `or`, `not`, `xor`, etc.) |

Type definitions in these headers (such as `size_t` and `ptrdiff_t` in `stddef.h`,
and `uintptr_t` in `stdint.h`) are automatically adapted to the target architecture's
pointer width and type sizes.

### System CRT Linkage

When producing executables, the integrated linker automatically locates and links
the system C runtime startup objects:

- `crt1.o` — Program entry point (`_start`)
- `crti.o` — Init section prologue
- `crtn.o` — Init section epilogue
- `libc.a` / `libc.so` — C standard library

These are found in target-specific system directories. See the
[CRT Object Locations](#crt-object-locations) table in the Target Selection section.

### Default Behavior Summary

| Setting | Default |
|---|---|
| Output mode | Static executable |
| Output file | `a.out` |
| Optimization level | `-O0` (no optimization) |
| Debug info | Disabled (enable with `-g`) |
| Target | Host architecture |
| Code model | Non-PIC (enable with `-fPIC`) |
| Retpoline | Disabled (enable with `-mretpoline`) |
| CET protection | Disabled (enable with `-fcf-protection`) |

---

## Flag Summary

Quick reference of all supported flags:

| Flag | Category | Description |
|---|---|---|
| `-c` | Compilation mode | Compile to object file only, do not link |
| `-o <path>` | Compilation mode | Set output file path |
| `-shared` | Compilation mode | Produce shared library |
| `-static` | Compilation mode | Produce statically linked executable (default) |
| `-I <dir>` | Preprocessor | Add include search directory |
| `-D <macro>[=<value>]` | Preprocessor | Define preprocessor macro |
| `-U <macro>` | Preprocessor | Undefine preprocessor macro |
| `-L <dir>` | Linker | Add library search directory |
| `-l <library>` | Linker | Link against library |
| `-fPIC` | Code generation | Generate position-independent code |
| `-mretpoline` | Code generation | Retpoline for indirect branches (x86-64) |
| `-fcf-protection` | Code generation | CET `endbr64` instrumentation (x86-64) |
| `-g` | Debug | Generate DWARF v4 debug information |
| `-O0` | Optimization | No optimization (default) |
| `-O1` | Optimization | Basic optimization |
| `-O2` | Optimization | Aggressive optimization |
| `--target <triple>` | Target | Select target architecture |
