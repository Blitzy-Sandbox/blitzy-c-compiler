/*
 * float.h — Floating-Point Characteristics (C11 Freestanding Header)
 *
 * Bundled freestanding header for the bcc (Blitzy C Compiler).
 * Conforms to C11 standard §5.2.4.2.2 "Characteristics of floating types
 * <float.h>".
 *
 * All four bcc target architectures (x86-64, i686, AArch64, RISC-V 64) use
 * IEEE 754 binary floating-point, so `float` (binary32) and `double`
 * (binary64) characteristics are identical across all architectures.
 *
 * For `long double`, bcc treats it as equivalent to `double` (64-bit
 * IEEE 754 binary64) across all architectures. This is a deliberate
 * simplification: real platforms vary (x87 80-bit extended on x86,
 * 128-bit quad on AArch64/RISC-V), but using double-equivalent values
 * guarantees correct compilation of real-world codebases (SQLite, Lua,
 * zlib, Redis) which rarely depend on extended long double precision.
 *
 * Self-contained — no dependencies on system headers or other bundled headers.
 *
 * IEEE 754 format summary:
 *   float  (binary32): 1 sign + 8 exponent + 23 significand = 32 bits
 *   double (binary64): 1 sign + 11 exponent + 52 significand = 64 bits
 *
 * C standard floating-point model (§5.2.4.2.2):
 *   x = s * b^e * SUM(f_k * b^(-k)) for k = 1 to p
 *   where b = FLT_RADIX, p = *_MANT_DIG, emin <= e <= emax
 *   The leading significand digit f_1 is nonzero for normalized numbers,
 *   so the significand lies in [b^(-1), 1) — i.e., [0.5, 1) for binary.
 *
 * The bcc compiler predefines __x86_64__, __i386__, __aarch64__, __riscv,
 * and __riscv_xlen based on the --target flag.
 */

#ifndef _FLOAT_H
#define _FLOAT_H

/* ========================================================================
 * Floating-point model parameters (C11 §5.2.4.2.2)
 * ======================================================================== */

/* Radix of exponent representation — all IEEE 754 types use base 2 */
#define FLT_RADIX 2

/* ========================================================================
 * Rounding mode and evaluation method
 * ======================================================================== */

/*
 * FLT_ROUNDS: Current rounding mode for floating-point addition.
 *   -1 = indeterminate
 *    0 = toward zero
 *    1 = to nearest (IEEE 754 default: round-to-nearest-ties-to-even)
 *    2 = toward positive infinity
 *    3 = toward negative infinity
 *
 * IEEE 754 specifies round-to-nearest as the default rounding mode.
 * All four bcc targets default to this mode.
 */
#define FLT_ROUNDS 1

/*
 * FLT_EVAL_METHOD: Precision used for intermediate floating-point evaluation.
 *   -1 = indeterminate
 *    0 = evaluate to the precision and range of the type
 *    1 = evaluate float and double as double
 *    2 = evaluate all as long double
 *
 * bcc evaluates each type at its own precision (no excess precision),
 * consistent with SSE/NEON/RISC-V F/D extension semantics.
 */
#define FLT_EVAL_METHOD 0

/*
 * DECIMAL_DIG: Number of decimal digits n such that any floating-point
 * number with p radix-b digits can be rounded to a floating-point number
 * with n decimal digits and back again without change to the value.
 *
 * Must be >= max(FLT_DECIMAL_DIG, DBL_DECIMAL_DIG, LDBL_DECIMAL_DIG).
 * Since bcc treats long double as double, DECIMAL_DIG = DBL_DECIMAL_DIG = 17.
 */
#define DECIMAL_DIG 17

/* ========================================================================
 * float (IEEE 754 binary32) characteristics
 *
 * Layout: 1 sign bit + 8 exponent bits + 23 significand bits
 * Exponent bias: 127
 * Normal exponent range (biased): 1 to 254 (unbiased: -126 to +127)
 * C model emin = -125 (because significand is in [0.5, 1), not [1, 2))
 * C model emax = 128
 * ======================================================================== */

/* Number of base-FLT_RADIX (binary) digits in the significand.
 * 23 stored bits + 1 implicit leading bit = 24 digits in the C model. */
#define FLT_MANT_DIG    24

/* Number of decimal digits q such that any floating-point number with q
 * decimal digits can be rounded into a float and back to q decimal digits
 * without change. floor((p-1) * log10(2)) = floor(23 * 0.30103) = 6 */
#define FLT_DIG         6

/* Number of decimal digits n needed to distinguish any two distinct float
 * values. ceil(1 + p * log10(2)) = ceil(1 + 24 * 0.30103) = ceil(8.225) = 9 */
#define FLT_DECIMAL_DIG 9

/* Minimum negative integer e such that FLT_RADIX^(e-1) is a normalized float.
 * IEEE 754 binary32 min biased exponent for normals is 1, unbiased is -126.
 * In the C model where significand is in [0.5, 1): emin = -126 + 1 = -125. */
#define FLT_MIN_EXP     (-125)

/* Minimum negative integer such that 10 raised to that power is in the
 * range of normalized float values. ceil(log10(FLT_MIN)) = ceil(-37.93) = -37 */
#define FLT_MIN_10_EXP  (-37)

/* Maximum integer e such that FLT_RADIX^(e-1) is representable.
 * IEEE 754 binary32 max biased exponent for normals is 254, unbiased is 127.
 * In the C model: emax = 127 + 1 = 128. */
#define FLT_MAX_EXP     128

/* Maximum integer such that 10 raised to that power is in the range of
 * representable finite float values. floor(log10(FLT_MAX)) = floor(38.23) = 38 */
#define FLT_MAX_10_EXP  38

/* Maximum representable finite float value.
 * (2 - 2^(-23)) * 2^127 = 3.40282347 * 10^38 */
#define FLT_MAX         3.40282347e+38F

/* Minimum normalized positive float value.
 * 2^(-126) = 1.17549435 * 10^(-38) */
#define FLT_MIN         1.17549435e-38F

/* Smallest positive float epsilon such that 1.0F + epsilon != 1.0F.
 * 2^(-23) = 1.19209290 * 10^(-7) */
#define FLT_EPSILON     1.19209290e-07F

/* Minimum positive subnormal float value (C11 addition).
 * 2^(-149) = 2^(-126) * 2^(-23) = 1.40129846 * 10^(-45) */
#define FLT_TRUE_MIN    1.40129846e-45F

/* Whether the type supports subnormal (denormalized) numbers (C11 addition).
 *   -1 = indeterminate
 *    0 = absent (does not support subnormals)
 *    1 = present (supports subnormals)
 * IEEE 754 mandates subnormal support. */
#define FLT_HAS_SUBNORM 1

/* ========================================================================
 * double (IEEE 754 binary64) characteristics
 *
 * Layout: 1 sign bit + 11 exponent bits + 52 significand bits
 * Exponent bias: 1023
 * Normal exponent range (biased): 1 to 2046 (unbiased: -1022 to +1023)
 * C model emin = -1021 (significand in [0.5, 1))
 * C model emax = 1024
 * ======================================================================== */

/* Number of base-FLT_RADIX (binary) digits in the significand.
 * 52 stored bits + 1 implicit leading bit = 53 digits in the C model. */
#define DBL_MANT_DIG    53

/* Number of decimal digits of precision.
 * floor((p-1) * log10(2)) = floor(52 * 0.30103) = floor(15.654) = 15 */
#define DBL_DIG         15

/* Number of decimal digits to distinguish any two distinct doubles.
 * ceil(1 + p * log10(2)) = ceil(1 + 53 * 0.30103) = ceil(16.954) = 17 */
#define DBL_DECIMAL_DIG 17

/* Minimum exponent (C model). IEEE 754: unbiased emin = -1022.
 * C model: -1022 + 1 = -1021. */
#define DBL_MIN_EXP     (-1021)

/* Minimum base-10 exponent for normalized doubles.
 * ceil(log10(DBL_MIN)) = ceil(-307.65) = -307 */
#define DBL_MIN_10_EXP  (-307)

/* Maximum exponent (C model). IEEE 754: unbiased emax = 1023.
 * C model: 1023 + 1 = 1024. */
#define DBL_MAX_EXP     1024

/* Maximum base-10 exponent for representable finite doubles.
 * floor(log10(DBL_MAX)) = floor(308.25) = 308 */
#define DBL_MAX_10_EXP  308

/* Maximum representable finite double value.
 * (2 - 2^(-52)) * 2^1023 = 1.7976931348623157 * 10^308 */
#define DBL_MAX         1.7976931348623157e+308

/* Minimum normalized positive double value.
 * 2^(-1022) = 2.2250738585072014 * 10^(-308) */
#define DBL_MIN         2.2250738585072014e-308

/* Smallest positive double epsilon such that 1.0 + epsilon != 1.0.
 * 2^(-52) = 2.2204460492503131 * 10^(-16) */
#define DBL_EPSILON     2.2204460492503131e-16

/* Minimum positive subnormal double value (C11 addition).
 * 2^(-1074) = 2^(-1022) * 2^(-52) = 4.9406564584124654 * 10^(-324) */
#define DBL_TRUE_MIN    4.9406564584124654e-324

/* Whether double supports subnormal numbers (C11 addition).
 * IEEE 754 mandates subnormal support. */
#define DBL_HAS_SUBNORM 1

/* ========================================================================
 * long double characteristics
 *
 * In bcc, long double is implemented as equivalent to double (IEEE 754
 * binary64) across all four target architectures. This is a deliberate
 * simplification — the values below match the double characteristics.
 *
 * Platform-native long double formats for reference (not used by bcc):
 *   x86-64/i686: 80-bit x87 extended (LDBL_MANT_DIG=64, LDBL_DIG=18)
 *   AArch64:     128-bit IEEE 754 quad (LDBL_MANT_DIG=113, LDBL_DIG=33)
 *   RISC-V 64:   128-bit IEEE 754 quad (LDBL_MANT_DIG=113, LDBL_DIG=33)
 *
 * Suffix L is used on long double constants per C11 §6.4.4.2.
 * ======================================================================== */

/* Number of binary digits in the long double significand.
 * Same as double: 53 (bcc long double == double). */
#define LDBL_MANT_DIG   53

/* Decimal digits of precision for long double.
 * Same as double: 15. */
#define LDBL_DIG        15

/* Decimal digits to distinguish any two distinct long doubles.
 * Same as double: 17. */
#define LDBL_DECIMAL_DIG 17

/* Minimum exponent (C model). Same as double: -1021. */
#define LDBL_MIN_EXP    (-1021)

/* Minimum base-10 exponent. Same as double: -307. */
#define LDBL_MIN_10_EXP (-307)

/* Maximum exponent (C model). Same as double: 1024. */
#define LDBL_MAX_EXP    1024

/* Maximum base-10 exponent. Same as double: 308. */
#define LDBL_MAX_10_EXP 308

/* Maximum representable finite long double value. */
#define LDBL_MAX        1.7976931348623157e+308L

/* Minimum normalized positive long double value. */
#define LDBL_MIN        2.2250738585072014e-308L

/* Smallest long double epsilon such that 1.0L + epsilon != 1.0L. */
#define LDBL_EPSILON    2.2204460492503131e-16L

/* Minimum positive subnormal long double value (C11 addition). */
#define LDBL_TRUE_MIN   4.9406564584124654e-324L

/* Whether long double supports subnormal numbers (C11 addition). */
#define LDBL_HAS_SUBNORM 1

#endif /* _FLOAT_H */
