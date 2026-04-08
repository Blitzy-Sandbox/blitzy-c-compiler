/*
 * stdint.h - Fixed-width integer types and limits
 *
 * C11 Standard (ISO/IEC 9899:2011) Section 7.20
 *
 * Bundled freestanding header for the bcc C compiler.
 * Self-contained: no dependencies on system headers or other bundled headers.
 * Target-width-adaptive for pointer-width types across four architectures:
 *   - x86-64  (LP64: 64-bit pointers, 64-bit long)
 *   - AArch64 (LP64: 64-bit pointers, 64-bit long)
 *   - RISC-V 64 (LP64: 64-bit pointers, 64-bit long)
 *   - i686   (ILP32: 32-bit pointers, 32-bit long)
 *
 * The bcc compiler predefines the following macros per target:
 *   __x86_64__            - x86-64 target
 *   __i386__              - i686 target
 *   __aarch64__           - AArch64 target
 *   __riscv, __riscv_xlen - RISC-V target with register width
 */

#ifndef _STDINT_H
#define _STDINT_H

/* ============================================================
 * Exact-width integer types (C11 Section 7.20.1.1)
 *
 * These types have exactly the specified width in bits.
 * Definitions are identical across all four target architectures:
 *   signed char    = 8 bits on all targets
 *   short          = 16 bits on all targets
 *   int            = 32 bits on all targets
 *   long long      = 64 bits on all targets
 *
 * Note: int64_t uses `long` on LP64 targets to match glibc's
 * <bits/types.h> which defines __int64_t as `signed long int`.
 * On ILP32 (i686) where long is 32-bit, `long long` is used.
 * ============================================================ */

typedef signed char        int8_t;
typedef short              int16_t;
typedef int                int32_t;
#if defined(__x86_64__) || defined(__aarch64__) || (defined(__riscv) && __riscv_xlen == 64)
typedef long               int64_t;
#else
typedef long long          int64_t;
#endif

typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__x86_64__) || defined(__aarch64__) || (defined(__riscv) && __riscv_xlen == 64)
typedef unsigned long      uint64_t;
#else
typedef unsigned long long uint64_t;
#endif

/* ============================================================
 * Minimum-width integer types (C11 Section 7.20.1.2)
 *
 * Each type is at least as wide as the specified width.
 * On all bcc targets, the minimum-width types are exact
 * aliases for the corresponding exact-width types.
 * ============================================================ */

typedef int8_t    int_least8_t;
typedef int16_t   int_least16_t;
typedef int32_t   int_least32_t;
typedef int64_t   int_least64_t;

typedef uint8_t   uint_least8_t;
typedef uint16_t  uint_least16_t;
typedef uint32_t  uint_least32_t;
typedef uint64_t  uint_least64_t;

/* ============================================================
 * Fastest minimum-width integer types (C11 Section 7.20.1.3)
 *
 * These types are at least as wide as the specified width
 * and are chosen for the fastest operation on the target.
 *
 * On most architectures, 32-bit int operations are the fastest
 * for widths between 9 and 32 bits, so int_fast16_t and
 * int_fast32_t are both typedef'd to int.
 * ============================================================ */

typedef int8_t       int_fast8_t;
typedef int          int_fast16_t;
typedef int          int_fast32_t;
typedef int64_t      int_fast64_t;

typedef uint8_t      uint_fast8_t;
typedef unsigned int uint_fast16_t;
typedef unsigned int uint_fast32_t;
typedef uint64_t     uint_fast64_t;

/* ============================================================
 * Pointer-width integer types (C11 Section 7.20.1.4)
 *
 * intptr_t and uintptr_t are wide enough to hold any valid
 * pointer value converted to an integer.
 *
 * LP64 targets (x86-64, AArch64, RISC-V 64): 64-bit pointers
 * ILP32 targets (i686): 32-bit pointers
 * ============================================================ */

#if defined(__x86_64__) || defined(__aarch64__) || \
    (defined(__riscv) && (__riscv_xlen == 64))
/* LP64: pointers are 64-bit */
typedef long          intptr_t;
typedef unsigned long uintptr_t;
#else
/* ILP32: pointers are 32-bit (i686) */
typedef int           intptr_t;
typedef unsigned int  uintptr_t;
#endif

/* ============================================================
 * Greatest-width integer types (C11 Section 7.20.1.5)
 *
 * intmax_t and uintmax_t represent the widest integer type
 * supported by the implementation. Using long long ensures
 * 64-bit width on both ILP32 and LP64 targets.
 * ============================================================ */

#if defined(__x86_64__) || defined(__aarch64__) || (defined(__riscv) && __riscv_xlen == 64)
typedef long               intmax_t;
typedef unsigned long      uintmax_t;
#else
typedef long long          intmax_t;
typedef unsigned long long uintmax_t;
#endif

/* ============================================================
 * Limits of exact-width integer types (C11 Section 7.20.2.1)
 *
 * Signed minimum values use the (-MAX - 1) idiom to avoid
 * issues where the magnitude of the minimum value cannot be
 * represented as a positive constant of the same type in the
 * preprocessor.
 * ============================================================ */

#define INT8_MIN    (-128)
#define INT8_MAX    127
#define UINT8_MAX   255

#define INT16_MIN   (-32768)
#define INT16_MAX   32767
#define UINT16_MAX  65535

#define INT32_MIN   (-2147483647 - 1)
#define INT32_MAX   2147483647
#define UINT32_MAX  4294967295U

#define INT64_MIN   (-9223372036854775807LL - 1LL)
#define INT64_MAX   9223372036854775807LL
#define UINT64_MAX  18446744073709551615ULL

/* ============================================================
 * Limits of minimum-width integer types (C11 Section 7.20.2.2)
 *
 * Since minimum-width types are aliases for exact-width types
 * on all bcc targets, limits are identical.
 * ============================================================ */

#define INT_LEAST8_MIN    INT8_MIN
#define INT_LEAST8_MAX    INT8_MAX
#define UINT_LEAST8_MAX   UINT8_MAX

#define INT_LEAST16_MIN   INT16_MIN
#define INT_LEAST16_MAX   INT16_MAX
#define UINT_LEAST16_MAX  UINT16_MAX

#define INT_LEAST32_MIN   INT32_MIN
#define INT_LEAST32_MAX   INT32_MAX
#define UINT_LEAST32_MAX  UINT32_MAX

#define INT_LEAST64_MIN   INT64_MIN
#define INT_LEAST64_MAX   INT64_MAX
#define UINT_LEAST64_MAX  UINT64_MAX

/* ============================================================
 * Limits of fastest minimum-width integer types (C11 Section 7.20.2.3)
 *
 * fast8 maps to int8_t: same limits as INT8_*.
 * fast16 and fast32 map to int/unsigned int: 32-bit range.
 * fast64 maps to int64_t: same limits as INT64_*.
 * ============================================================ */

#define INT_FAST8_MIN     INT8_MIN
#define INT_FAST8_MAX     INT8_MAX
#define UINT_FAST8_MAX    UINT8_MAX

#define INT_FAST16_MIN    INT32_MIN
#define INT_FAST16_MAX    INT32_MAX
#define UINT_FAST16_MAX   UINT32_MAX

#define INT_FAST32_MIN    INT32_MIN
#define INT_FAST32_MAX    INT32_MAX
#define UINT_FAST32_MAX   UINT32_MAX

#define INT_FAST64_MIN    INT64_MIN
#define INT_FAST64_MAX    INT64_MAX
#define UINT_FAST64_MAX   UINT64_MAX

/* ============================================================
 * Limits of pointer-width integer types (C11 Section 7.20.2.4)
 * and other pointer-width limits — TARGET-ADAPTIVE
 *
 * INTPTR_MIN/MAX and UINTPTR_MAX: limits of intptr_t/uintptr_t.
 * PTRDIFF_MIN/MAX: limits of ptrdiff_t (pointer-width signed).
 * SIZE_MAX: maximum value of size_t (pointer-width unsigned).
 * ============================================================ */

#if defined(__x86_64__) || defined(__aarch64__) || \
    (defined(__riscv) && (__riscv_xlen == 64))
/* LP64: 64-bit pointer-width limits */
#define INTPTR_MIN    (-9223372036854775807L - 1L)
#define INTPTR_MAX    9223372036854775807L
#define UINTPTR_MAX   18446744073709551615UL
#define PTRDIFF_MIN   (-9223372036854775807L - 1L)
#define PTRDIFF_MAX   9223372036854775807L
#define SIZE_MAX      18446744073709551615UL
#else
/* ILP32: 32-bit pointer-width limits (i686) */
#define INTPTR_MIN    (-2147483647 - 1)
#define INTPTR_MAX    2147483647
#define UINTPTR_MAX   4294967295U
#define PTRDIFF_MIN   (-2147483647 - 1)
#define PTRDIFF_MAX   2147483647
#define SIZE_MAX      4294967295U
#endif

/* ============================================================
 * Limits of greatest-width integer types (C11 Section 7.20.2.5)
 * ============================================================ */

#define INTMAX_MIN    (-9223372036854775807LL - 1LL)
#define INTMAX_MAX    9223372036854775807LL
#define UINTMAX_MAX   18446744073709551615ULL

/* ============================================================
 * Limits of other integer types (C11 Section 7.20.3)
 *
 * These limits apply to types defined in other headers but
 * whose limit macros are required to appear in <stdint.h>.
 * ============================================================ */

/* sig_atomic_t is int on Linux for all targets */
#define SIG_ATOMIC_MIN  (-2147483647 - 1)
#define SIG_ATOMIC_MAX  2147483647

/* wchar_t is int (signed 32-bit) on Linux for all bcc targets */
#define WCHAR_MIN       (-2147483647 - 1)
#define WCHAR_MAX       2147483647

/* wint_t is unsigned int on Linux for all bcc targets */
#define WINT_MIN        0U
#define WINT_MAX        4294967295U

/* ============================================================
 * Macros for integer constant expressions (C11 Section 7.20.4)
 *
 * These macros expand their argument to an integer constant
 * expression with the correct type suffix. They use the ##
 * token pasting operator to append type suffixes.
 *
 * For types narrower than int (8-bit, 16-bit), no suffix is
 * needed because integer constants are already at least int.
 * ============================================================ */

/* Macros for minimum-width integer constants (C11 Section 7.20.4.1) */
#define INT8_C(c)     c
#define INT16_C(c)    c
#define INT32_C(c)    c
#define INT64_C(c)    c ## LL

#define UINT8_C(c)    c
#define UINT16_C(c)   c
#define UINT32_C(c)   c ## U
#define UINT64_C(c)   c ## ULL

/* Macros for greatest-width integer constants (C11 Section 7.20.4.2) */
#define INTMAX_C(c)   c ## LL
#define UINTMAX_C(c)  c ## ULL

#endif /* _STDINT_H */
