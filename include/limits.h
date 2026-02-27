/*
 * limits.h — Integer Limits (C11 Freestanding Header)
 *
 * Bundled freestanding header for the bcc (Blitzy C Compiler).
 * Conforms to C11 standard §5.2.4.2.1 "Sizes of integer types <limits.h>".
 *
 * This header is target-width-adaptive: `long`-related limits differ between
 * 32-bit targets (i686, ILP32 data model) and 64-bit targets (x86-64,
 * AArch64, RISC-V 64, LP64 data model).
 *
 * Self-contained — no dependencies on system headers or other bundled headers.
 *
 * Architecture notes:
 *   - CHAR_BIT is always 8 on all Linux targets
 *   - short is always 16-bit across all four targets
 *   - int is always 32-bit across all four targets
 *   - long is 32-bit on i686 (ILP32) and 64-bit on x86-64/AArch64/RISC-V 64 (LP64)
 *   - long long is always 64-bit across all four targets
 *   - char is signed on x86-64 and i686; unsigned on AArch64 (AAPCS64) and RISC-V 64 (psABI)
 *
 * The bcc compiler predefines __x86_64__, __i386__, __aarch64__, __riscv,
 * and __riscv_xlen based on the --target flag for these conditionals to work.
 */

#ifndef _LIMITS_H
#define _LIMITS_H

/* ========================================================================
 * Architecture-independent limits
 * These values are identical across all four supported architectures.
 * ======================================================================== */

/* Number of bits in the smallest object that is not a bit-field (byte) */
#define CHAR_BIT    8

/* Minimum and maximum values for signed char */
#define SCHAR_MIN   (-128)
#define SCHAR_MAX   127

/* Maximum value for unsigned char */
#define UCHAR_MAX   255

/* ========================================================================
 * char signedness — architecture-dependent
 *
 * On Linux:
 *   - x86-64: char is signed
 *   - i686:   char is signed
 *   - RISC-V 64: char is unsigned (per RISC-V psABI)
 *   - AArch64: char is unsigned (per AAPCS64)
 * ======================================================================== */
#if defined(__aarch64__) || defined(__riscv)
/* AAPCS64 and RISC-V psABI: char is unsigned */
#define CHAR_MIN    0
#define CHAR_MAX    UCHAR_MAX
#else
/* x86-64 and i686: char is signed */
#define CHAR_MIN    SCHAR_MIN
#define CHAR_MAX    SCHAR_MAX
#endif

/* Maximum number of bytes in a multibyte character for any supported locale */
#define MB_LEN_MAX  16

/* Minimum and maximum values for short int (always 16-bit) */
#define SHRT_MIN    (-32768)
#define SHRT_MAX    32767

/* Maximum value for unsigned short int */
#define USHRT_MAX   65535

/* Minimum and maximum values for int (always 32-bit) */
/* NOTE: INT_MIN uses the (-MAX - 1) trick to avoid preprocessor constant
 * overflow that would occur with a direct -2147483648 literal, since the
 * preprocessor parses that as unary minus applied to 2147483648 which
 * exceeds the positive range of a 32-bit signed integer. */
#define INT_MIN     (-2147483647 - 1)
#define INT_MAX     2147483647

/* Maximum value for unsigned int */
#define UINT_MAX    4294967295U

/* ========================================================================
 * Target-width-adaptive long limits
 *
 * LP64 data model (x86-64, AArch64, RISC-V 64): long is 64-bit
 * ILP32 data model (i686): long is 32-bit
 *
 * Detection uses target-specific predefined macros:
 *   __x86_64__  — defined for x86-64 targets
 *   __aarch64__ — defined for AArch64 targets
 *   __riscv     — defined for RISC-V targets
 *   __riscv_xlen — set to 64 for RV64 targets
 * ======================================================================== */
#if defined(__x86_64__) || defined(__aarch64__) || (defined(__riscv) && (__riscv_xlen == 64))
/* LP64: long is 64-bit */
#define LONG_MIN    (-9223372036854775807L - 1L)
#define LONG_MAX    9223372036854775807L
#define ULONG_MAX   18446744073709551615UL
#else
/* ILP32: long is 32-bit (i686) */
#define LONG_MIN    (-2147483647L - 1L)
#define LONG_MAX    2147483647L
#define ULONG_MAX   4294967295UL
#endif

/* ========================================================================
 * Long long limits (always 64-bit on all targets)
 * ======================================================================== */

/* Minimum value for long long int */
/* NOTE: Uses the (-MAX - 1) trick as with INT_MIN above */
#define LLONG_MIN   (-9223372036854775807LL - 1LL)

/* Maximum value for long long int */
#define LLONG_MAX   9223372036854775807LL

/* Maximum value for unsigned long long int */
#define ULLONG_MAX  18446744073709551615ULL

#endif /* _LIMITS_H */
