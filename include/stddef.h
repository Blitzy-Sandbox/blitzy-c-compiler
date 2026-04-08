/*
 * stddef.h — Common Definitions (C11 Freestanding Header)
 *
 * Provides the following C11 §7.19 definitions:
 *   - NULL        — Null pointer constant
 *   - size_t      — Unsigned integer type of sizeof result (target-adaptive)
 *   - ptrdiff_t   — Signed integer type of pointer subtraction result (target-adaptive)
 *   - wchar_t     — Wide character type
 *   - max_align_t — Type with maximum scalar alignment (C11)
 *   - offsetof    — Byte offset of struct member
 *
 * Target-width-adaptive:
 *   - 32-bit (i686):                size_t = unsigned int,  ptrdiff_t = int
 *   - 64-bit (x86-64/AArch64/RV64): size_t = unsigned long, ptrdiff_t = long
 *
 * This header is self-contained with no dependencies on system headers or
 * other bundled headers. Individual typedef guards allow selective inclusion
 * when types may be defined by other headers (e.g., <stdlib.h>).
 *
 * Part of the bcc (Blitzy C Compiler) bundled freestanding header set.
 */

#ifndef _STDDEF_H
#define _STDDEF_H

/*
 * NULL — Null pointer constant (C11 §7.19)
 *
 * Defined as ((void *)0) which is the standard C form providing type safety.
 * The void-pointer cast form is preferred over plain 0 or 0L as it enables
 * the compiler to issue warnings on incorrect pointer/integer conversions.
 * Guarded against redefinition since NULL may be defined by other headers.
 */
#ifndef NULL
#define NULL ((void *)0)
#endif

/*
 * size_t — Unsigned integer type of the result of sizeof (C11 §7.19)
 *
 * Must be pointer-width to correctly represent the size of any object:
 *   - LP64 targets (x86-64, AArch64, RISC-V 64): unsigned long (8 bytes)
 *   - ILP32 targets (i686): unsigned int (4 bytes)
 *
 * Note: On LP64 targets, size_t is unsigned long (NOT unsigned long long).
 * This distinction matters for printf format specifiers (%zu vs %llu).
 *
 * The bcc compiler predefines __x86_64__, __aarch64__, __riscv, and
 * __riscv_xlen macros based on the --target flag for these conditionals.
 */
#ifndef _SIZE_T_DEFINED
#define _SIZE_T_DEFINED
#if defined(__x86_64__) || defined(__aarch64__) || (defined(__riscv) && (__riscv_xlen == 64))
typedef unsigned long size_t;
#else
typedef unsigned int size_t;
#endif
#endif /* _SIZE_T_DEFINED */

/*
 * ptrdiff_t — Signed integer type of the result of subtracting two pointers (C11 §7.19)
 *
 * Must be pointer-width to correctly represent the difference between any
 * two pointers into the same array:
 *   - LP64 targets (x86-64, AArch64, RISC-V 64): long (8 bytes)
 *   - ILP32 targets (i686): int (4 bytes)
 *
 * Note: On LP64 targets, ptrdiff_t is long (NOT long long).
 */
#ifndef _PTRDIFF_T_DEFINED
#define _PTRDIFF_T_DEFINED
#if defined(__x86_64__) || defined(__aarch64__) || (defined(__riscv) && (__riscv_xlen == 64))
typedef long ptrdiff_t;
#else
typedef int ptrdiff_t;
#endif
#endif /* _PTRDIFF_T_DEFINED */

/*
 * wchar_t — Wide character type (C11 §7.19)
 *
 * On Linux, wchar_t is a 32-bit signed integer (int) across all four
 * supported architectures. This provides full Unicode scalar value coverage.
 */
#ifndef _WCHAR_T_DEFINED
#define _WCHAR_T_DEFINED
typedef int wchar_t;
#endif /* _WCHAR_T_DEFINED */

/*
 * max_align_t — Type whose alignment is at least as great as that of every
 *               scalar type (C11 §7.19)
 *
 * This struct contains members of the most-aligned fundamental types to
 * ensure that sizeof(max_align_t) and _Alignof(max_align_t) are at least
 * as large as any scalar alignment requirement:
 *   - long long:   8-byte alignment on all targets
 *   - long double: varies (8, 16, or architecture-specific alignment)
 *
 * This type is used for correct alignment of memory allocation results
 * (e.g., malloc must return memory suitably aligned for max_align_t).
 */
typedef struct {
    long long __ll;
    long double __ld;
} max_align_t;

/*
 * offsetof(type, member) — Byte offset of a member within a struct (C11 §7.19)
 *
 * Uses the compiler builtin __builtin_offsetof which:
 *   - Correctly handles all struct layouts including those with padding
 *   - Works with bitfield members (implementation-defined)
 *   - Generates a compile-time constant expression of type size_t
 *   - Is recognized by bcc's parser/semantic analyzer as a compiler intrinsic
 *
 * The builtin approach is preferred over the traditional cast-based macro
 * ((size_t)&((type *)0)->member) because it handles edge cases correctly
 * and avoids undefined behavior from null pointer dereference expressions.
 */
#define offsetof(type, member) __builtin_offsetof(type, member)

#endif /* _STDDEF_H */
