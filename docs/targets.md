# Supported Target Architectures

The `bcc` C compiler supports four Linux target architectures, selectable at compile time via the `--target` flag. Each target defines its own Application Binary Interface (ABI), register file, ELF binary format, relocation type set, and calling convention. When `--target` is not specified, `bcc` defaults to the host architecture (typically `x86_64-linux-gnu` on modern Linux systems).

## Target Overview

| Target Triple | Architecture | Pointer Size | `long` Size | ELF Class | Endianness |
|---|---|---|---|---|---|
| `x86_64-linux-gnu` | x86-64 | 8 bytes | 8 bytes | ELF64 | Little |
| `i686-linux-gnu` | i686 | 4 bytes | 4 bytes | ELF32 | Little |
| `aarch64-linux-gnu` | AArch64 | 8 bytes | 8 bytes | ELF64 | Little |
| `riscv64-linux-gnu` | RISC-V 64 | 8 bytes | 8 bytes | ELF64 | Little |

### Target Selection

Use the `--target` flag to select the compilation target:

```bash
# Compile for the default (host) target
bcc hello.c -o hello

# Compile for a specific target
bcc --target x86_64-linux-gnu hello.c -o hello
bcc --target i686-linux-gnu hello.c -o hello_32
bcc --target aarch64-linux-gnu hello.c -o hello_arm64
bcc --target riscv64-linux-gnu hello.c -o hello_riscv
```

### C Data Type Sizes

All four targets share the following fundamental type sizes unless otherwise noted:

| C Type | x86-64 | i686 | AArch64 | RISC-V 64 |
|---|---|---|---|---|
| `char` | 1 | 1 | 1 | 1 |
| `short` | 2 | 2 | 2 | 2 |
| `int` | 4 | 4 | 4 | 4 |
| `long` | 8 | 4 | 8 | 8 |
| `long long` | 8 | 8 | 8 | 8 |
| `float` | 4 | 4 | 4 | 4 |
| `double` | 8 | 8 | 8 | 8 |
| `long double` | 16 | 12 | 16 | 16 |
| pointer | 8 | 4 | 8 | 8 |
| `size_t` | 8 | 4 | 8 | 8 |
| `ptrdiff_t` | 8 | 4 | 8 | 8 |

---

## x86-64 (`x86_64-linux-gnu`)

### Architecture Overview

x86-64 (also known as AMD64 or x86_64) is the 64-bit extension of the Intel/AMD x86 instruction set architecture. It is the primary development target for `bcc` and the most widely used architecture on Linux servers and desktops.

### ABI: System V AMD64

The x86-64 target follows the **System V AMD64 ABI**, the standard calling convention for 64-bit Linux.

**Integer Argument Passing**

Arguments are passed in registers in the following order. Once the registers are exhausted, remaining arguments are passed on the stack (right-to-left push order).

| Argument Position | Register |
|---|---|
| 1st | `rdi` |
| 2nd | `rsi` |
| 3rd | `rdx` |
| 4th | `rcx` |
| 5th | `r8` |
| 6th | `r9` |
| 7th+ | stack |

**Floating-Point Argument Passing**

Floating-point and vector arguments are passed in the SSE registers:

| Argument Position | Register |
|---|---|
| 1st | `xmm0` |
| 2nd | `xmm1` |
| 3rd | `xmm2` |
| 4th | `xmm3` |
| 5th | `xmm4` |
| 6th | `xmm5` |
| 7th | `xmm6` |
| 8th | `xmm7` |
| 9th+ | stack |

**Return Values**

| Type | Register(s) |
|---|---|
| Integer (≤64 bits) | `rax` |
| Integer (65–128 bits) | `rax` + `rdx` |
| Floating-point | `xmm0` |
| Floating-point pair | `xmm0` + `xmm1` |

**Register Convention**

| Category | Registers |
|---|---|
| Caller-saved (volatile) | `rax`, `rcx`, `rdx`, `rsi`, `rdi`, `r8`–`r11`, `xmm0`–`xmm15` |
| Callee-saved (non-volatile) | `rbx`, `r12`–`r15`, `rbp` |
| Stack pointer | `rsp` |
| Frame pointer (optional) | `rbp` |

**Stack Layout**

- Stack alignment: 16-byte aligned at the `call` instruction site (i.e., `rsp % 16 == 0` immediately before `call`; after the `call` pushes a return address, `rsp % 16 == 8` at function entry).
- Red zone: 128 bytes below `rsp` are reserved and not clobbered by signal handlers or interrupts. Leaf functions may use this space without adjusting `rsp`. Note: the red zone is **not** used in kernel code.

### Register File

| Register Set | Registers | Count | Purpose |
|---|---|---|---|
| General-purpose | `rax`, `rbx`, `rcx`, `rdx`, `rsi`, `rdi`, `rbp`, `rsp`, `r8`–`r15` | 16 | Integer operations, addressing |
| SSE | `xmm0`–`xmm15` | 16 | Floating-point, SIMD operations |

Each general-purpose register has sub-register aliases: `rax` (64-bit) → `eax` (32-bit) → `ax` (16-bit) → `al`/`ah` (8-bit).

### ELF Format

| Property | Value |
|---|---|
| ELF class | ELF64 |
| Data encoding | Little-endian (ELFDATA2LSB) |
| Machine type | `EM_X86_64` (62) |
| ELF header size | 64 bytes |
| Section header entry size | 64 bytes |
| Program header entry size | 56 bytes |

### Relocation Types

The following relocation types are supported by the `bcc` integrated linker for x86-64:

| Relocation Type | Value | Description |
|---|---|---|
| `R_X86_64_NONE` | 0 | No relocation |
| `R_X86_64_64` | 1 | Direct 64-bit absolute |
| `R_X86_64_PC32` | 2 | PC-relative 32-bit signed |
| `R_X86_64_GOT32` | 3 | 32-bit GOT entry offset |
| `R_X86_64_PLT32` | 4 | 32-bit PLT-relative offset |
| `R_X86_64_GLOB_DAT` | 6 | Create GOT entry |
| `R_X86_64_JUMP_SLOT` | 7 | Create PLT entry |
| `R_X86_64_RELATIVE` | 8 | Adjust by program base |
| `R_X86_64_GOTPCREL` | 9 | 32-bit signed PC-relative GOT offset |
| `R_X86_64_32` | 10 | Direct 32-bit zero-extended |
| `R_X86_64_32S` | 11 | Direct 32-bit sign-extended |
| `R_X86_64_16` | 12 | Direct 16-bit zero-extended |
| `R_X86_64_PC16` | 13 | 16-bit PC-relative |
| `R_X86_64_8` | 14 | Direct 8-bit sign-extended |
| `R_X86_64_PC8` | 15 | 8-bit PC-relative |
| `R_X86_64_REX_GOTPCRELX` | 42 | Relaxable GOTPCREL |

### Security Hardening

The x86-64 target supports three security hardening features that are **exclusive to this architecture**:

#### Retpoline (`-mretpoline`)

Retpoline mitigates **Spectre variant 2** (Branch Target Injection) attacks by replacing indirect branch instructions (`jmp *%reg`, `call *%reg`) with a speculative-execution-safe thunk sequence.

When `-mretpoline` is enabled:
- All indirect jumps (`jmp *%rax`, etc.) are replaced with calls to `__x86_retpoline_rax` (or the appropriate register variant).
- The retpoline thunk uses the return stack buffer (RSB) to steer speculative execution to a safe infinite loop (the "pause-lfence" trap), while the actual indirect target executes non-speculatively via `ret`.

```
# Example retpoline thunk for rax
__x86_retpoline_rax:
    call .Lsetup
.Lcapture:
    pause
    lfence
    jmp .Lcapture
.Lsetup:
    mov %rax, (%rsp)
    ret
```

Usage:
```bash
bcc -mretpoline source.c -o hardened
```

#### CET: Control-flow Enforcement Technology (`-fcf-protection`)

Intel CET uses the `endbr64` instruction to mark valid indirect branch targets. Hardware enforcing CET will fault on indirect jumps or calls that land on any instruction other than `endbr64`.

When `-fcf-protection` is enabled:
- `endbr64` (4-byte NOP on non-CET hardware) is inserted at the beginning of every function.
- `endbr64` is inserted at every indirect branch target (function pointers, switch jump tables).

Usage:
```bash
bcc -fcf-protection source.c -o cet_hardened
```

#### Stack Probing (Automatic)

For functions with stack frames exceeding one page (4096 bytes), `bcc` automatically inserts stack probe sequences. This ensures that the guard page between the stack and other memory regions is touched in order, preventing stack clash attacks.

The probe loop touches each page from the current stack pointer down to the desired allocation:

```
# Stack probe for large frame (> 4096 bytes)
.Lprobe_loop:
    sub $4096, %rsp
    test %rsp, (%rsp)      # Touch the page
    sub $remaining, %rsp    # Final sub for remainder
```

Stack probing is always enabled for x86-64 when frames exceed 4096 bytes. No CLI flag is required.

### Position-Independent Code

When `-fPIC` is specified:
- Global variable accesses use `%rip`-relative addressing through the Global Offset Table (GOT): `mov symbol@GOTPCREL(%rip), %rax` followed by a load through the GOT entry.
- External function calls use Procedure Linkage Table (PLT) stubs: `call symbol@PLT`.
- Internal function calls within the same compilation unit may use direct PC-relative calls.

### CRT Objects Location

| Distribution | Path |
|---|---|
| Debian / Ubuntu | `/usr/lib/x86_64-linux-gnu/` |
| Fedora / RHEL | `/usr/lib64/` |

Required CRT objects: `crt1.o`, `crti.o`, `crtn.o`
Required library: `libc.a` (static) or `libc.so` (dynamic)

---

## i686 (`i686-linux-gnu`)

### Architecture Overview

i686 (also known as IA-32 or x86-32) is the 32-bit x86 architecture. It uses a more constrained register file compared to x86-64 and passes all function arguments on the stack rather than in registers.

### ABI: cdecl / System V i386

The i686 target follows the **System V i386 ABI** with the **cdecl** calling convention.

**Argument Passing**

All arguments are passed on the stack in right-to-left order (the last argument is pushed first). The caller is responsible for cleaning up the stack after the call returns.

```
# Calling foo(1, 2, 3):
push $3          # 3rd argument
push $2          # 2nd argument
push $1          # 1st argument
call foo
add $12, %esp    # Caller cleanup (3 args × 4 bytes)
```

**Return Values**

| Type | Register(s) |
|---|---|
| 32-bit integer | `eax` |
| 64-bit integer | `eax` (low) + `edx` (high) |
| Floating-point | `st(0)` (x87 FPU stack top) |
| Struct (small) | `eax` + `edx` (if ≤ 8 bytes) |
| Struct (large) | Hidden pointer in first stack argument |

**Register Convention**

| Category | Registers |
|---|---|
| Caller-saved (volatile) | `eax`, `ecx`, `edx` |
| Callee-saved (non-volatile) | `ebx`, `esi`, `edi`, `ebp` |
| Stack pointer | `esp` |
| Frame pointer (optional) | `ebp` |

**Stack Layout**

- Stack alignment: 16-byte aligned at the `call` instruction site (modern System V i386 convention).
- No red zone on i686.

### Register File

| Register Set | Registers | Count | Purpose |
|---|---|---|---|
| General-purpose | `eax`, `ebx`, `ecx`, `edx`, `esi`, `edi`, `ebp`, `esp` | 8 | Integer operations, addressing |
| SSE | `xmm0`–`xmm7` | 8 | Floating-point, SIMD operations |
| x87 FPU | `st(0)`–`st(7)` | 8 | Legacy floating-point |

Each general-purpose register has sub-register aliases: `eax` (32-bit) → `ax` (16-bit) → `al`/`ah` (8-bit).

### ELF Format

| Property | Value |
|---|---|
| ELF class | ELF32 |
| Data encoding | Little-endian (ELFDATA2LSB) |
| Machine type | `EM_386` (3) |
| ELF header size | 52 bytes |
| Section header entry size | 40 bytes |
| Program header entry size | 32 bytes |

### Relocation Types

| Relocation Type | Value | Description |
|---|---|---|
| `R_386_NONE` | 0 | No relocation |
| `R_386_32` | 1 | Direct 32-bit absolute |
| `R_386_PC32` | 2 | PC-relative 32-bit |
| `R_386_GOT32` | 3 | 32-bit GOT entry |
| `R_386_PLT32` | 4 | 32-bit PLT address |
| `R_386_COPY` | 5 | Copy symbol at runtime |
| `R_386_GLOB_DAT` | 6 | Create GOT entry |
| `R_386_JMP_SLOT` | 7 | Create PLT entry |
| `R_386_RELATIVE` | 8 | Adjust by program base |
| `R_386_GOTOFF` | 9 | 32-bit offset to GOT |
| `R_386_GOTPC` | 10 | 32-bit PC-relative offset to GOT |

### Encoding Notes

- **No REX prefix**: The i686 instruction encoding does not use REX prefixes (which are an x86-64 extension). All opcodes use the legacy 1- or 2-byte opcode map.
- **32-bit addressing modes**: ModR/M and SIB bytes encode 32-bit base + index × scale + displacement addressing.
- **64-bit arithmetic**: 64-bit integer operations (`long long`) are implemented using register pairs (`eax:edx` for results, pairs of instructions for arithmetic).

### Position-Independent Code

When `-fPIC` is specified:
- Because i686 lacks a PC-relative addressing mode (no `%rip`), PIC requires materializing the GOT base address at runtime using the `__x86.get_pc_thunk.*` pattern:
  ```
  call __x86.get_pc_thunk.bx    # Get PC into %ebx
  add $_GLOBAL_OFFSET_TABLE_, %ebx  # Compute GOT base
  ```
- Global variable accesses use `symbol@GOT(%ebx)` indirection.
- External function calls use `call symbol@PLT`.

### CRT Objects Location

| Distribution | Path |
|---|---|
| Debian / Ubuntu (native) | `/usr/lib/i386-linux-gnu/` |
| Cross-compilation sysroot | `/usr/i686-linux-gnu/lib/` |

Required CRT objects: `crt1.o`, `crti.o`, `crtn.o`
Required library: `libc.a` (static) or `libc.so` (dynamic)

---

## AArch64 (`aarch64-linux-gnu`)

### Architecture Overview

AArch64 (also known as ARM64) is the 64-bit execution state of the ARMv8-A and later architectures. It features a clean, fixed-width 32-bit instruction encoding, a large register file, and a load/store architecture where all data processing operates on registers rather than memory.

### ABI: AAPCS64

The AArch64 target follows the **Arm Architecture Procedure Call Standard for 64-bit** (AAPCS64).

**Integer Argument Passing**

| Argument Position | Register |
|---|---|
| 1st | `x0` |
| 2nd | `x1` |
| 3rd | `x2` |
| 4th | `x3` |
| 5th | `x4` |
| 6th | `x5` |
| 7th | `x6` |
| 8th | `x7` |
| 9th+ | stack |

**SIMD/Floating-Point Argument Passing**

| Argument Position | Register |
|---|---|
| 1st | `v0` (or `d0`/`s0` for scalar) |
| 2nd | `v1` |
| 3rd | `v2` |
| 4th | `v3` |
| 5th | `v4` |
| 6th | `v5` |
| 7th | `v6` |
| 8th | `v7` |
| 9th+ | stack |

**Return Values**

| Type | Register(s) |
|---|---|
| Integer (≤64 bits) | `x0` |
| Integer (65–128 bits) | `x0` + `x1` |
| Floating-point | `v0` (or `d0`/`s0`) |
| Struct (small, ≤16 bytes) | `x0` + `x1` or `v0`–`v3` |
| Struct (large) | Hidden pointer in `x8` |

**Register Convention**

| Category | Registers |
|---|---|
| Caller-saved (volatile) | `x0`–`x15`, `x16`–`x17` (IP0/IP1), `x18` (platform register) |
| Callee-saved (non-volatile) | `x19`–`x28` |
| Frame pointer | `x29` (`fp`) |
| Link register | `x30` (`lr`) |
| Stack pointer | `sp` |
| Zero register | `xzr` / `wzr` |

**Stack Layout**

- SP alignment: 16-byte aligned at all times (hardware-enforced on ARMv8).
- No red zone in the standard AAPCS64.

### Register File

| Register Set | Registers | Count | Purpose |
|---|---|---|---|
| General-purpose | `x0`–`x30` | 31 | Integer operations, addressing |
| SIMD/FP | `v0`–`v31` | 32 | Floating-point, NEON SIMD |
| Special | `sp`, `pc`, `xzr`/`wzr` | 3 | Stack pointer, program counter, zero |

Each general-purpose register has a 32-bit alias: `x0` → `w0`, `x1` → `w1`, etc.
Each SIMD register can be accessed at different widths: `v0` (128-bit), `d0` (64-bit), `s0` (32-bit), `h0` (16-bit), `b0` (8-bit).

### ELF Format

| Property | Value |
|---|---|
| ELF class | ELF64 |
| Data encoding | Little-endian (ELFDATA2LSB) |
| Machine type | `EM_AARCH64` (183) |
| ELF header size | 64 bytes |
| Section header entry size | 64 bytes |
| Program header entry size | 56 bytes |

### Relocation Types

| Relocation Type | Value | Description |
|---|---|---|
| `R_AARCH64_NONE` | 0 | No relocation |
| `R_AARCH64_ABS64` | 257 | Direct 64-bit absolute |
| `R_AARCH64_ABS32` | 258 | Direct 32-bit absolute |
| `R_AARCH64_ABS16` | 259 | Direct 16-bit absolute |
| `R_AARCH64_PREL64` | 260 | PC-relative 64-bit |
| `R_AARCH64_PREL32` | 261 | PC-relative 32-bit |
| `R_AARCH64_ADR_PREL_PG_HI21` | 275 | Page-relative ADRP immediate |
| `R_AARCH64_ADD_ABS_LO12_NC` | 277 | Low 12-bit absolute ADD immediate |
| `R_AARCH64_LDST8_ABS_LO12_NC` | 278 | Low 12-bit load/store (byte) |
| `R_AARCH64_LDST16_ABS_LO12_NC` | 284 | Low 12-bit load/store (halfword) |
| `R_AARCH64_LDST32_ABS_LO12_NC` | 285 | Low 12-bit load/store (word) |
| `R_AARCH64_LDST64_ABS_LO12_NC` | 286 | Low 12-bit load/store (doubleword) |
| `R_AARCH64_CALL26` | 283 | Function call within ±128 MB |
| `R_AARCH64_JUMP26` | 282 | Unconditional branch within ±128 MB |
| `R_AARCH64_GLOB_DAT` | 1025 | Create GOT entry |
| `R_AARCH64_JUMP_SLOT` | 1026 | Create PLT entry |
| `R_AARCH64_RELATIVE` | 1027 | Adjust by program base |

### Encoding Notes

- **Fixed-width instructions**: All AArch64 instructions are exactly 32 bits (4 bytes) wide. This simplifies instruction decoding and code size estimation.
- **Barrel shifter operands**: Many data-processing instructions support a shifted second operand (LSL, LSR, ASR, ROR) without an extra instruction.
- **Conditional select**: The `CSEL`/`CSINC`/`CSINV`/`CSNEG` family replaces many conditional branches with predicated operations.
- **Load/store pair**: `LDP`/`STP` instructions load or store two registers in a single instruction, used for efficient prologue/epilogue sequences.

### Position-Independent Code

When `-fPIC` is specified:
- **ADRP + ADD**: PC-relative addressing uses a two-instruction sequence:
  ```
  adrp x0, symbol          # Load 4KB-aligned page of symbol into x0
  add  x0, x0, :lo12:symbol # Add low 12 bits of symbol address
  ```
- **GOT-indirect**: External symbols are accessed via the GOT:
  ```
  adrp x0, :got:symbol       # Load GOT page
  ldr  x0, [x0, :got_lo12:symbol]  # Load address from GOT entry
  ```
- **PLT stubs**: External function calls go through PLT entries.

### CRT Objects Location

| Distribution | Path |
|---|---|
| Cross-compilation sysroot | `/usr/aarch64-linux-gnu/lib/` |

Required CRT objects: `crt1.o`, `crti.o`, `crtn.o`
Required library: `libc.a` (static) or `libc.so` (dynamic)

---

## RISC-V 64 (`riscv64-linux-gnu`)

### Architecture Overview

RISC-V 64 is a 64-bit implementation of the open RISC-V instruction set architecture. The `bcc` compiler targets **RV64GC**, which includes:

- **RV64I**: Base 64-bit integer instruction set
- **G** = **IMAFD**: Standard general-purpose extensions
  - **M**: Integer multiply/divide
  - **A**: Atomic operations
  - **F**: Single-precision floating-point
  - **D**: Double-precision floating-point
- **C**: Compressed 16-bit instructions for improved code density

### ABI: LP64D

The RISC-V 64 target follows the **LP64D** ABI, which passes floating-point arguments in floating-point registers and uses 64-bit `long` and pointers.

**Integer Argument Passing**

| Argument Position | Register | ABI Name |
|---|---|---|
| 1st | `x10` | `a0` |
| 2nd | `x11` | `a1` |
| 3rd | `x12` | `a2` |
| 4th | `x13` | `a3` |
| 5th | `x14` | `a4` |
| 6th | `x15` | `a5` |
| 7th | `x16` | `a6` |
| 8th | `x17` | `a7` |
| 9th+ | stack | — |

**Floating-Point Argument Passing**

| Argument Position | Register | ABI Name |
|---|---|---|
| 1st | `f10` | `fa0` |
| 2nd | `f11` | `fa1` |
| 3rd | `f12` | `fa2` |
| 4th | `f13` | `fa3` |
| 5th | `f14` | `fa4` |
| 6th | `f15` | `fa5` |
| 7th | `f16` | `fa6` |
| 8th | `f17` | `fa7` |
| 9th+ | stack | — |

**Return Values**

| Type | Register(s) |
|---|---|
| Integer (≤64 bits) | `a0` |
| Integer (65–128 bits) | `a0` + `a1` |
| Floating-point | `fa0` |
| Floating-point pair | `fa0` + `fa1` |
| Struct (small) | `a0` + `a1` |
| Struct (large) | Hidden pointer in `a0` |

**Register Convention**

| Category | ABI Names | Hardware Registers |
|---|---|---|
| Caller-saved (volatile) | `t0`–`t6`, `a0`–`a7` | `x5`–`x7`, `x28`–`x31`, `x10`–`x17` |
| Callee-saved (non-volatile) | `s0`–`s11` | `x8`–`x9`, `x18`–`x27` |
| Return address | `ra` | `x1` |
| Stack pointer | `sp` | `x2` |
| Global pointer | `gp` | `x3` |
| Thread pointer | `tp` | `x4` |
| Frame pointer | `s0` / `fp` | `x8` |
| Zero register | `zero` | `x0` (hardwired to 0) |

**Stack Layout**

- SP alignment: 16-byte aligned at all times.
- No red zone in the RISC-V ABI.

### Register File

**Integer Registers**

| Register | ABI Name | Purpose |
|---|---|---|
| `x0` | `zero` | Hardwired zero |
| `x1` | `ra` | Return address |
| `x2` | `sp` | Stack pointer |
| `x3` | `gp` | Global pointer |
| `x4` | `tp` | Thread pointer |
| `x5`–`x7` | `t0`–`t2` | Temporary |
| `x8` | `s0` / `fp` | Saved / Frame pointer |
| `x9` | `s1` | Saved |
| `x10`–`x17` | `a0`–`a7` | Arguments / Return values |
| `x18`–`x27` | `s2`–`s11` | Saved |
| `x28`–`x31` | `t3`–`t6` | Temporary |

**Floating-Point Registers**

| Register | ABI Name | Purpose |
|---|---|---|
| `f0`–`f7` | `ft0`–`ft7` | FP temporary |
| `f8`–`f9` | `fs0`–`fs1` | FP saved |
| `f10`–`f17` | `fa0`–`fa7` | FP arguments / Return values |
| `f18`–`f27` | `fs2`–`fs11` | FP saved |
| `f28`–`f31` | `ft8`–`ft11` | FP temporary |

### ISA Extensions

| Extension | Description | Instructions Added |
|---|---|---|
| **RV64I** | Base integer | `add`, `sub`, `and`, `or`, `xor`, `sll`, `srl`, `sra`, `slt`, `sltu`, `addi`, `andi`, `ori`, `xori`, `slti`, `sltiu`, `lb`, `lh`, `lw`, `ld`, `sb`, `sh`, `sw`, `sd`, `beq`, `bne`, `blt`, `bge`, `bltu`, `bgeu`, `jal`, `jalr`, `lui`, `auipc`, `ecall`, `ebreak`, `fence`, `addiw`, `slliw`, `srliw`, `sraiw`, `addw`, `subw`, `sllw`, `srlw`, `sraw` |
| **M** | Multiply/Divide | `mul`, `mulh`, `mulhsu`, `mulhu`, `div`, `divu`, `rem`, `remu`, `mulw`, `divw`, `divuw`, `remw`, `remuw` |
| **A** | Atomics | `lr.w`, `sc.w`, `amoswap.w`, `amoadd.w`, `amoand.w`, `amoor.w`, `amoxor.w`, `amomin.w`, `amomax.w`, `amominu.w`, `amomaxu.w` (and `.d` 64-bit variants) |
| **F** | Single-precision FP | `flw`, `fsw`, `fadd.s`, `fsub.s`, `fmul.s`, `fdiv.s`, `fsqrt.s`, `fmin.s`, `fmax.s`, `fmadd.s`, `fmsub.s`, `fnmadd.s`, `fnmsub.s`, `fcvt.*`, `fmv.*`, `feq.s`, `flt.s`, `fle.s`, `fclass.s` |
| **D** | Double-precision FP | `fld`, `fsd`, `fadd.d`, `fsub.d`, `fmul.d`, `fdiv.d`, `fsqrt.d`, `fmin.d`, `fmax.d`, `fmadd.d`, `fmsub.d`, `fnmadd.d`, `fnmsub.d`, `fcvt.*`, `fmv.*`, `feq.d`, `flt.d`, `fle.d`, `fclass.d` |
| **C** | Compressed (16-bit) | `c.addi`, `c.li`, `c.lui`, `c.addi16sp`, `c.addi4spn`, `c.slli`, `c.srli`, `c.srai`, `c.andi`, `c.mv`, `c.add`, `c.and`, `c.or`, `c.xor`, `c.sub`, `c.lw`, `c.sw`, `c.ld`, `c.sd`, `c.j`, `c.jr`, `c.jalr`, `c.beqz`, `c.bnez`, etc. |

### ELF Format

| Property | Value |
|---|---|
| ELF class | ELF64 |
| Data encoding | Little-endian (ELFDATA2LSB) |
| Machine type | `EM_RISCV` (243) |
| ELF header size | 64 bytes |
| Section header entry size | 64 bytes |
| Program header entry size | 56 bytes |
| ELF flags | `EF_RISCV_RVC` (0x0001) for C extension, `EF_RISCV_FLOAT_ABI_DOUBLE` (0x0004) for LP64D |

### Relocation Types

| Relocation Type | Value | Description |
|---|---|---|
| `R_RISCV_NONE` | 0 | No relocation |
| `R_RISCV_32` | 1 | Direct 32-bit absolute |
| `R_RISCV_64` | 2 | Direct 64-bit absolute |
| `R_RISCV_BRANCH` | 16 | 12-bit PC-relative branch |
| `R_RISCV_JAL` | 17 | 20-bit PC-relative JAL |
| `R_RISCV_CALL` | 18 | `auipc` + `jalr` pair (32-bit PC-relative) |
| `R_RISCV_CALL_PLT` | 19 | `auipc` + `jalr` pair via PLT |
| `R_RISCV_GOT_HI20` | 20 | High 20 bits of GOT entry PC-offset |
| `R_RISCV_PCREL_HI20` | 23 | High 20 bits of PC-relative offset |
| `R_RISCV_PCREL_LO12_I` | 24 | Low 12 bits of PC-relative (I-type) |
| `R_RISCV_PCREL_LO12_S` | 25 | Low 12 bits of PC-relative (S-type) |
| `R_RISCV_HI20` | 26 | High 20 bits of absolute address |
| `R_RISCV_LO12_I` | 27 | Low 12 bits of absolute (I-type) |
| `R_RISCV_LO12_S` | 28 | Low 12 bits of absolute (S-type) |
| `R_RISCV_ADD32` | 35 | 32-bit addition |
| `R_RISCV_SUB32` | 39 | 32-bit subtraction |
| `R_RISCV_RELAX` | 51 | Linker relaxation marker |

### Encoding Notes

RISC-V uses a variable-length instruction encoding:

**32-bit Base Instructions (R/I/S/B/U/J formats)**

| Format | Bit Layout | Used For |
|---|---|---|
| **R-type** | `[funct7][rs2][rs1][funct3][rd][opcode]` | Register-register operations |
| **I-type** | `[imm[11:0]][rs1][funct3][rd][opcode]` | Immediate operations, loads |
| **S-type** | `[imm[11:5]][rs2][rs1][funct3][imm[4:0]][opcode]` | Stores |
| **B-type** | `[imm[12|10:5]][rs2][rs1][funct3][imm[4:1|11]][opcode]` | Conditional branches |
| **U-type** | `[imm[31:12]][rd][opcode]` | Upper immediate (LUI, AUIPC) |
| **J-type** | `[imm[20|10:1|11|19:12]][rd][opcode]` | Unconditional jumps (JAL) |

**16-bit Compressed Instructions (C extension)**

Compressed instructions use a 16-bit encoding and are distinguished by their two least-significant bits not being `11`. They map to common 32-bit instructions with reduced register fields and immediate ranges, providing approximately 25–30% code density improvement.

### Position-Independent Code

When `-fPIC` is specified:
- **PC-relative addressing**: The `auipc` (Add Upper Immediate to PC) instruction loads a high 20-bit PC-relative offset, combined with a 12-bit low offset in a subsequent instruction:
  ```
  auipc a0, %pcrel_hi(symbol)    # Load high 20 bits of PC-relative offset
  addi  a0, a0, %pcrel_lo(symbol) # Add low 12 bits
  ```
- **GOT-indirect**: External symbols are accessed through the Global Offset Table:
  ```
  auipc a0, %got_pcrel_hi(symbol)  # Load GOT entry page offset
  ld    a0, %pcrel_lo(symbol)(a0)  # Load address from GOT
  ```
- **PLT stubs**: External function calls go through Procedure Linkage Table entries.
- **Linker relaxation**: The linker may relax `auipc` + `jalr` pairs into shorter sequences when the target is within range, guided by `R_RISCV_RELAX` relocations.

### CRT Objects Location

| Distribution | Path |
|---|---|
| Cross-compilation sysroot | `/usr/riscv64-linux-gnu/lib/` |

Required CRT objects: `crt1.o`, `crti.o`, `crtn.o`
Required library: `libc.a` (static) or `libc.so` (dynamic)

---

## ELF Format Details

### ELF32 vs ELF64 Structural Differences

The `bcc` compiler produces ELF32 binaries for the i686 target and ELF64 binaries for x86-64, AArch64, and RISC-V 64 targets. The key structural differences between the two formats are:

| Property | ELF32 (i686) | ELF64 (x86-64, AArch64, RISC-V 64) |
|---|---|---|
| ELF header size | 52 bytes | 64 bytes |
| Section header entry size | 40 bytes | 64 bytes |
| Program header entry size | 32 bytes | 56 bytes |
| Address field width | 4 bytes (`Elf32_Addr`) | 8 bytes (`Elf64_Addr`) |
| Offset field width | 4 bytes (`Elf32_Off`) | 8 bytes (`Elf64_Off`) |
| Symbol entry size | 16 bytes (`Elf32_Sym`) | 24 bytes (`Elf64_Sym`) |
| Relocation entry size (with addend) | 12 bytes (`Elf32_Rela`) | 24 bytes (`Elf64_Rela`) |
| `e_ident[EI_CLASS]` | `ELFCLASS32` (1) | `ELFCLASS64` (2) |
| Maximum virtual address space | 4 GB | 16 EB (theoretical) |

### Common ELF Sections

The following sections appear in all `bcc`-produced binaries regardless of target architecture:

| Section | Type | Description |
|---|---|---|
| `.text` | `SHT_PROGBITS` | Executable machine code |
| `.data` | `SHT_PROGBITS` | Initialized read-write data |
| `.bss` | `SHT_NOBITS` | Uninitialized data (zero-filled at load time) |
| `.rodata` | `SHT_PROGBITS` | Read-only data (string literals, constants) |
| `.symtab` | `SHT_SYMTAB` | Symbol table |
| `.strtab` | `SHT_STRTAB` | String table for `.symtab` |
| `.shstrtab` | `SHT_STRTAB` | Section header string table |
| `.rela.text` | `SHT_RELA` | Relocations for `.text` (in relocatable objects) |
| `.rela.data` | `SHT_RELA` | Relocations for `.data` (in relocatable objects) |
| `.comment` | `SHT_PROGBITS` | Compiler identification string |

When `-g` is specified, additional DWARF v4 debug sections are included:

| Section | Description |
|---|---|
| `.debug_info` | Debugging Information Entries (DIEs) |
| `.debug_abbrev` | Abbreviation tables for `.debug_info` |
| `.debug_line` | Line number program |
| `.debug_str` | Debug string table |
| `.debug_aranges` | Address range lookup table |
| `.debug_frame` | Call Frame Information (CFI) |
| `.debug_loc` | Location lists for variable tracking |

### Output Modes

#### Static Executable (Default)

The default output mode produces a statically linked ELF executable.

**Characteristics:**
- Contains `PT_LOAD` program headers for loadable segments
- Entry point set to `_start` (from `crt1.o`)
- Linked with CRT objects: `crt1.o` (startup), `crti.o` (init prologue), `crtn.o` (init epilogue)
- Linked with `libc.a` for C standard library functions
- No dynamic linker or shared library dependencies
- Self-contained binary; can execute on any compatible Linux system

**Segment Layout:**
```
Segment 0: PT_LOAD (R-X)  — .text, .rodata
Segment 1: PT_LOAD (RW-)  — .data, .bss
```

**Usage:**
```bash
bcc hello.c -o hello          # Default: static executable
bcc -static hello.c -o hello  # Explicit static linking
```

#### Shared Library (`-shared`)

Produces a dynamically loadable shared object (`.so` file).

**Characteristics:**
- Contains `.dynamic` section with dynamic linking metadata
- Contains `.dynsym` and `.dynstr` for exported dynamic symbols
- Contains `.plt` (Procedure Linkage Table) and `.got` (Global Offset Table) sections
- Contains `.rela.dyn` and `.rela.plt` for dynamic relocations
- Generates `DT_SONAME`, `DT_NEEDED`, and other dynamic tags
- Requires position-independent code (`-fPIC`)

**Additional Sections:**
| Section | Description |
|---|---|
| `.dynamic` | Dynamic linking metadata (`DT_*` entries) |
| `.dynsym` | Dynamic symbol table |
| `.dynstr` | Dynamic string table |
| `.plt` | Procedure Linkage Table |
| `.got` | Global Offset Table |
| `.got.plt` | GOT entries for PLT |
| `.rela.dyn` | Dynamic relocations |
| `.rela.plt` | PLT relocations |
| `.hash` or `.gnu.hash` | Symbol hash table for fast lookup |

**Usage:**
```bash
bcc -shared -fPIC lib.c -o libfoo.so
```

#### Relocatable Object (`-c`)

Produces an unlinked object file (`.o` file) suitable for later linking.

**Characteristics:**
- No program headers (ELF type `ET_REL`)
- Contains unresolved relocations in `.rela.*` sections
- Contains a full symbol table in `.symtab`
- Can be combined with other object files via the linker

**Usage:**
```bash
bcc -c source.c -o source.o
```

---

## QEMU User-Mode Testing

### Purpose

QEMU user-mode emulation allows running cross-compiled binaries for non-native architectures directly on an x86-64 development host without booting a full virtual machine. This is essential for testing `bcc` output for the i686, AArch64, and RISC-V 64 targets.

### Installation

Install QEMU user-mode emulators and cross-compilation libraries:

```bash
# Install QEMU static user-mode emulators
sudo apt-get install -y qemu-user-static

# Install cross-compilation CRT objects and libc
sudo apt-get install -y libc6-dev-i386-cross        # i686
sudo apt-get install -y libc6-dev-arm64-cross        # AArch64
sudo apt-get install -y libc6-dev-riscv64-cross      # RISC-V 64
```

### QEMU Commands Per Architecture

| Architecture | QEMU Command | Binary Name |
|---|---|---|
| i686 (32-bit x86) | `qemu-i386-static` | `qemu-i386-static ./binary` |
| AArch64 (ARM 64-bit) | `qemu-aarch64-static` | `qemu-aarch64-static ./binary` |
| RISC-V 64 | `qemu-riscv64-static` | `qemu-riscv64-static ./binary` |

> **Note:** Native x86-64 binaries run directly on the host without QEMU.

### Sysroot Configuration

When running dynamically linked binaries, QEMU needs to know where to find the target's shared libraries. Use the `QEMU_LD_PREFIX` environment variable:

```bash
# For AArch64 dynamically linked binaries
export QEMU_LD_PREFIX=/usr/aarch64-linux-gnu
qemu-aarch64-static ./hello

# For RISC-V 64 dynamically linked binaries
export QEMU_LD_PREFIX=/usr/riscv64-linux-gnu
qemu-riscv64-static ./hello

# For i686 dynamically linked binaries
export QEMU_LD_PREFIX=/usr/i686-linux-gnu
qemu-i386-static ./hello
```

> **Tip:** Statically linked binaries (the default `bcc` output mode) do not require `QEMU_LD_PREFIX` because they contain no dynamic library references.

### Using binfmt_misc

On systems with `binfmt_misc` configured (common on Debian/Ubuntu with `qemu-user-static` installed), cross-architecture binaries can be executed directly without explicitly invoking `qemu-*-static`:

```bash
# If binfmt_misc is configured, you can run directly:
./hello_arm64    # Kernel automatically invokes qemu-aarch64-static
./hello_riscv    # Kernel automatically invokes qemu-riscv64-static
./hello_32       # Kernel automatically invokes qemu-i386-static
```

Check if binfmt_misc is configured:
```bash
ls /proc/sys/fs/binfmt_misc/
# Should list entries like qemu-aarch64, qemu-riscv64, qemu-i386
```

---

## Cross-Compilation Workflow Examples

### Example 1: AArch64

```bash
# 1. Write a simple C program
cat > hello.c << 'EOF'
#include <stdio.h>
int main(void) {
    printf("Hello, World!\n");
    return 0;
}
EOF

# 2. Compile for AArch64
bcc --target aarch64-linux-gnu hello.c -o hello_arm64

# 3. Verify ELF format
file hello_arm64
# Expected: ELF 64-bit LSB executable, ARM aarch64, version 1 (SYSV), statically linked

# 4. Inspect with readelf
readelf -h hello_arm64 | grep -E "Class|Machine"
# Expected:
#   Class:                             ELF64
#   Machine:                           AArch64

# 5. Run via QEMU
qemu-aarch64-static ./hello_arm64
# Expected output: Hello, World!
```

### Example 2: i686

```bash
# 1. Compile the same program for i686
bcc --target i686-linux-gnu hello.c -o hello_32

# 2. Verify ELF format
file hello_32
# Expected: ELF 32-bit LSB executable, Intel 80386, version 1 (SYSV), statically linked

# 3. Inspect with readelf
readelf -h hello_32 | grep -E "Class|Machine"
# Expected:
#   Class:                             ELF32
#   Machine:                           Intel 80386

# 4. Run via QEMU
qemu-i386-static ./hello_32
# Expected output: Hello, World!
```

### Example 3: RISC-V 64

```bash
# 1. Compile the same program for RISC-V 64
bcc --target riscv64-linux-gnu hello.c -o hello_riscv

# 2. Verify ELF format
file hello_riscv
# Expected: ELF 64-bit LSB executable, UCB RISC-V, version 1 (SYSV), statically linked

# 3. Inspect with readelf
readelf -h hello_riscv | grep -E "Class|Machine"
# Expected:
#   Class:                             ELF64
#   Machine:                           RISC-V

# 4. Run via QEMU
qemu-riscv64-static ./hello_riscv
# Expected output: Hello, World!
```

### Example 4: Multi-Architecture Build Script

```bash
#!/bin/bash
# Build and test a C program for all four architectures

SOURCE="$1"
BASENAME="${SOURCE%.c}"

declare -A TARGETS=(
    ["x86_64"]="x86_64-linux-gnu"
    ["i686"]="i686-linux-gnu"
    ["aarch64"]="aarch64-linux-gnu"
    ["riscv64"]="riscv64-linux-gnu"
)

declare -A QEMU=(
    ["i686"]="qemu-i386-static"
    ["aarch64"]="qemu-aarch64-static"
    ["riscv64"]="qemu-riscv64-static"
)

for arch in "${!TARGETS[@]}"; do
    triple="${TARGETS[$arch]}"
    output="${BASENAME}_${arch}"

    echo "=== Compiling for $arch ($triple) ==="
    bcc --target "$triple" "$SOURCE" -o "$output"

    if [ $? -eq 0 ]; then
        echo "=== Running $output ==="
        if [ "$arch" = "x86_64" ]; then
            ./"$output"
        else
            ${QEMU[$arch]} ./"$output"
        fi
    else
        echo "FAILED to compile for $arch"
    fi
    echo
done
```

---

## Target Configuration Internals

### How `--target` Maps to Internal Configuration

When the `--target` flag is provided, `bcc` parses the target triple and constructs a `TargetConfig` struct that propagates through the entire compilation pipeline. This struct is defined in `src/driver/target.rs` and affects every phase of compilation.

### TargetConfig Fields

The `TargetConfig` struct contains the following architecture-specific parameters:

| Field | Description | Example (x86-64) |
|---|---|---|
| `arch` | Architecture identifier | `Arch::X86_64` |
| `pointer_width` | Pointer size in bytes | `8` |
| `endianness` | Byte order | `Endian::Little` |
| `elf_class` | ELF32 or ELF64 | `ElfClass::Elf64` |
| `elf_machine` | ELF `e_machine` value | `62` (`EM_X86_64`) |
| `abi` | ABI convention identifier | `Abi::SystemVAmd64` |
| `long_size` | Size of `long` type in bytes | `8` |
| `long_double_size` | Size of `long double` in bytes | `16` |
| `max_align` | Maximum natural alignment | `16` |
| `register_file` | Available registers for allocation | Architecture-specific |
| `relocation_model` | Default relocation model | `RelocModel::Static` |
| `has_red_zone` | Whether the ABI specifies a red zone | `true` |

### Pipeline Impact

The target configuration influences every compilation phase:

| Pipeline Phase | Target-Dependent Behavior |
|---|---|
| **Preprocessor** | Predefined macros (`__x86_64__`, `__i386__`, `__aarch64__`, `__riscv`, `__LP64__`, etc.) |
| **Semantic Analysis** | Type sizes (`sizeof(long)`, `sizeof(void*)`, alignment requirements) |
| **IR Generation** | Size of IR integer types, pointer width, struct layout |
| **Optimization** | Target-specific constant folding (e.g., pointer arithmetic widths) |
| **Code Generation** | Instruction selection, register allocation, ABI calling conventions |
| **Linker** | ELF class (32/64), relocation types, CRT object paths, section alignment |
| **Debug Info** | DWARF register number mappings, address size in compilation unit headers |

### Predefined Macros Per Target

Each target defines architecture-specific predefined macros during preprocessing:

| Macro | x86-64 | i686 | AArch64 | RISC-V 64 |
|---|---|---|---|---|
| `__x86_64__` | ✓ | — | — | — |
| `__amd64__` | ✓ | — | — | — |
| `__i386__` | — | ✓ | — | — |
| `__i686__` | — | ✓ | — | — |
| `__aarch64__` | — | — | ✓ | — |
| `__ARM_ARCH` | — | — | `8` | — |
| `__riscv` | — | — | — | ✓ |
| `__riscv_xlen` | — | — | — | `64` |
| `__LP64__` | ✓ | — | ✓ | ✓ |
| `__ILP32__` | — | ✓ | — | — |
| `__linux__` | ✓ | ✓ | ✓ | ✓ |
| `__unix__` | ✓ | ✓ | ✓ | ✓ |
| `__ELF__` | ✓ | ✓ | ✓ | ✓ |

### Adding a New Target (Future)

While `bcc` currently supports four targets, the architecture is designed for extensibility. Adding a new target requires:

1. **Create a new codegen backend** in `src/codegen/<arch>/` implementing the `CodeGen` trait (instruction selection, encoding, ABI)
2. **Add relocation types** to `src/linker/relocations.rs` for the new architecture
3. **Register the target triple** in `src/driver/target.rs` with appropriate `TargetConfig` values
4. **Add ELF machine type** and any format-specific handling to `src/linker/elf.rs`
5. **Add predefined macros** for the new architecture in the preprocessor
6. **Add DWARF register mappings** in `src/debug/frame.rs`
7. **Add QEMU testing support** in the validation test harness

---

## See Also

- [CLI Reference](cli.md) — Complete documentation of all command-line flags including `--target`
- [Architecture Overview](architecture.md) — Compiler pipeline design and module relationships
- [Linker Internals](internals/linker.md) — Detailed ELF format handling and relocation processing
- [DWARF Internals](internals/dwarf.md) — Debug information generation per architecture
