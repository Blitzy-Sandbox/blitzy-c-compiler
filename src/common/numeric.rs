// src/common/numeric.rs — Arbitrary-Precision Numeric Constants
//
// Provides `BigInt`, an arbitrary-width (at least 128-bit) integer type used for
// compile-time constant evaluation in the preprocessor expression evaluator
// (`#if` directive integer arithmetic and `defined()`) and by the constant folding
// optimization pass. The representation uses Rust's native `i128` internally and
// tracks signedness to apply correct C semantics for comparisons, shifts, and
// arithmetic overflow detection.
//
// Design rationale: The C standard requires preprocessor arithmetic at `intmax_t`
// width (typically 64 bits). 128 bits exceeds this requirement and is sufficient
// for all compile-time evaluation scenarios in this compiler. The i128-based
// approach provides excellent performance (all operations map to native CPU
// instructions on 64-bit hosts) with zero heap allocation, making it ideal for
// the constant folding hot path.
//
// Zero external dependencies — uses only the Rust standard library.

use std::cmp::Ordering;
use std::fmt;

/// Arbitrary-width integer type for compile-time constant evaluation.
///
/// Supports signed values in the range \[-2^127, 2^127-1\] and unsigned values
/// in \[0, 2^128-1\]. The `is_unsigned` flag controls interpretation for
/// signedness-sensitive operations (comparisons, right shifts, overflow checks).
///
/// Internally, values are stored as `i128` bit patterns. Unsigned values exceeding
/// `i128::MAX` are stored via wrapping reinterpretation — the bit pattern is
/// preserved and correctly recovered via `to_u128()`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BigInt {
    /// The value stored as a 128-bit signed integer (bit pattern).
    /// For unsigned values > i128::MAX, this holds the wrapping reinterpretation.
    value: i128,
    /// Whether this value should be interpreted as unsigned.
    /// Affects comparison, right-shift, and overflow detection semantics.
    is_unsigned: bool,
}

// =============================================================================
// Construction and Conversion
// =============================================================================

impl BigInt {
    /// Creates a `BigInt` from a signed 64-bit integer.
    ///
    /// The result is marked as signed (`is_unsigned = false`).
    #[inline]
    pub fn from_i64(value: i64) -> Self {
        BigInt {
            value: value as i128,
            is_unsigned: false,
        }
    }

    /// Creates a `BigInt` from an unsigned 64-bit integer.
    ///
    /// The result is marked as unsigned (`is_unsigned = true`).
    #[inline]
    pub fn from_u64(value: u64) -> Self {
        BigInt {
            value: value as i128,
            is_unsigned: true,
        }
    }

    /// Creates a `BigInt` from a signed 128-bit integer.
    ///
    /// The result is marked as signed (`is_unsigned = false`).
    #[inline]
    pub fn from_i128(value: i128) -> Self {
        BigInt {
            value,
            is_unsigned: false,
        }
    }

    /// Creates a `BigInt` from an unsigned 128-bit integer.
    ///
    /// Values exceeding `i128::MAX` are stored via wrapping reinterpretation.
    /// The result is marked as unsigned (`is_unsigned = true`).
    #[inline]
    pub fn from_u128(value: u128) -> Self {
        BigInt {
            value: value as i128,
            is_unsigned: true,
        }
    }

    /// Returns the additive identity (zero), marked as signed.
    #[inline]
    pub fn zero() -> Self {
        BigInt {
            value: 0,
            is_unsigned: false,
        }
    }

    /// Returns the multiplicative identity (one), marked as signed.
    #[inline]
    pub fn one() -> Self {
        BigInt {
            value: 1,
            is_unsigned: false,
        }
    }

    /// Attempts to convert to `i64`. Returns `None` if the value is out of range.
    ///
    /// For unsigned values, the raw bit pattern is checked against the `i64` range.
    #[inline]
    pub fn to_i64(&self) -> Option<i64> {
        if self.is_unsigned {
            let v = self.value as u128;
            if v <= i64::MAX as u128 {
                Some(v as i64)
            } else {
                None
            }
        } else if self.value >= i64::MIN as i128 && self.value <= i64::MAX as i128 {
            Some(self.value as i64)
        } else {
            None
        }
    }

    /// Attempts to convert to `u64`. Returns `None` if the value is out of range.
    ///
    /// Negative signed values return `None`. Unsigned values are checked against
    /// the `u64` range.
    #[inline]
    pub fn to_u64(&self) -> Option<u64> {
        if self.is_unsigned {
            let v = self.value as u128;
            if v <= u64::MAX as u128 {
                Some(v as u64)
            } else {
                None
            }
        } else {
            if self.value < 0 {
                return None;
            }
            if self.value <= u64::MAX as i128 {
                Some(self.value as u64)
            } else {
                None
            }
        }
    }

    /// Converts to `i128` directly. For unsigned values exceeding `i128::MAX`,
    /// this returns the wrapping reinterpretation (matching C's signed cast semantics).
    #[inline]
    pub fn to_i128(&self) -> i128 {
        self.value
    }

    /// Converts to `u128`. For negative signed values, this returns the two's
    /// complement unsigned reinterpretation (matching C's unsigned cast semantics).
    #[inline]
    pub fn to_u128(&self) -> u128 {
        self.value as u128
    }

    /// Returns `true` if the value is zero (regardless of signedness).
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.value == 0
    }

    /// Returns `true` if the value is negative.
    ///
    /// Unsigned values are never negative, even if the bit pattern would be
    /// negative when interpreted as signed.
    #[inline]
    pub fn is_negative(&self) -> bool {
        if self.is_unsigned {
            false
        } else {
            self.value < 0
        }
    }

    /// Returns `true` if this value is unsigned.
    #[inline]
    pub fn is_unsigned(&self) -> bool {
        self.is_unsigned
    }

    /// Sets the unsigned flag. This changes the interpretation of the bit pattern
    /// without modifying the underlying value.
    #[inline]
    pub fn set_unsigned(&mut self, unsigned: bool) {
        self.is_unsigned = unsigned;
    }
}

// =============================================================================
// Arithmetic Operations (with overflow detection)
// =============================================================================

impl BigInt {
    /// Adds two `BigInt` values, returning the result and an overflow flag.
    ///
    /// If either operand is unsigned, the operation uses unsigned arithmetic
    /// and the result is marked unsigned (C's "usual arithmetic conversions").
    /// The overflow flag indicates whether the result wrapped around.
    pub fn add(&self, other: &BigInt) -> (BigInt, bool) {
        let result_unsigned = self.is_unsigned || other.is_unsigned;
        if result_unsigned {
            let a = self.value as u128;
            let b = other.value as u128;
            let (result, overflow) = a.overflowing_add(b);
            (
                BigInt {
                    value: result as i128,
                    is_unsigned: true,
                },
                overflow,
            )
        } else {
            let (result, overflow) = self.value.overflowing_add(other.value);
            (
                BigInt {
                    value: result,
                    is_unsigned: false,
                },
                overflow,
            )
        }
    }

    /// Subtracts `other` from `self`, returning the result and an overflow flag.
    ///
    /// If either operand is unsigned, the operation uses unsigned arithmetic
    /// and the result is marked unsigned.
    pub fn sub(&self, other: &BigInt) -> (BigInt, bool) {
        let result_unsigned = self.is_unsigned || other.is_unsigned;
        if result_unsigned {
            let a = self.value as u128;
            let b = other.value as u128;
            let (result, overflow) = a.overflowing_sub(b);
            (
                BigInt {
                    value: result as i128,
                    is_unsigned: true,
                },
                overflow,
            )
        } else {
            let (result, overflow) = self.value.overflowing_sub(other.value);
            (
                BigInt {
                    value: result,
                    is_unsigned: false,
                },
                overflow,
            )
        }
    }

    /// Multiplies two `BigInt` values, returning the result and an overflow flag.
    ///
    /// If either operand is unsigned, the operation uses unsigned arithmetic.
    pub fn mul(&self, other: &BigInt) -> (BigInt, bool) {
        let result_unsigned = self.is_unsigned || other.is_unsigned;
        if result_unsigned {
            let a = self.value as u128;
            let b = other.value as u128;
            let (result, overflow) = a.overflowing_mul(b);
            (
                BigInt {
                    value: result as i128,
                    is_unsigned: true,
                },
                overflow,
            )
        } else {
            let (result, overflow) = self.value.overflowing_mul(other.value);
            (
                BigInt {
                    value: result,
                    is_unsigned: false,
                },
                overflow,
            )
        }
    }

    /// Divides `self` by `other`. Returns `None` if `other` is zero.
    ///
    /// If either operand is unsigned, unsigned division is performed.
    /// For signed division where `self == i128::MIN` and `other == -1`,
    /// the result wraps (matching C's implementation-defined behavior).
    pub fn div(&self, other: &BigInt) -> Option<BigInt> {
        let result_unsigned = self.is_unsigned || other.is_unsigned;
        if result_unsigned {
            let a = self.value as u128;
            let b = other.value as u128;
            if b == 0 {
                return None;
            }
            Some(BigInt {
                value: (a / b) as i128,
                is_unsigned: true,
            })
        } else {
            if other.value == 0 {
                return None;
            }
            // wrapping_div handles i128::MIN / -1 without panicking
            Some(BigInt {
                value: self.value.wrapping_div(other.value),
                is_unsigned: false,
            })
        }
    }

    /// Computes `self % other`. Returns `None` if `other` is zero.
    ///
    /// If either operand is unsigned, unsigned modulo is performed.
    pub fn rem(&self, other: &BigInt) -> Option<BigInt> {
        let result_unsigned = self.is_unsigned || other.is_unsigned;
        if result_unsigned {
            let a = self.value as u128;
            let b = other.value as u128;
            if b == 0 {
                return None;
            }
            Some(BigInt {
                value: (a % b) as i128,
                is_unsigned: true,
            })
        } else {
            if other.value == 0 {
                return None;
            }
            // wrapping_rem handles i128::MIN % -1 without panicking
            Some(BigInt {
                value: self.value.wrapping_rem(other.value),
                is_unsigned: false,
            })
        }
    }

    /// Negates the value using wrapping arithmetic.
    ///
    /// For unsigned values, this computes the two's complement negation
    /// (equivalent to `UINT_MAX - value + 1` in C), preserving the unsigned flag.
    /// For signed values, `i128::MIN.wrapping_neg()` wraps to `i128::MIN`.
    pub fn neg(&self) -> BigInt {
        BigInt {
            value: self.value.wrapping_neg(),
            is_unsigned: self.is_unsigned,
        }
    }
}

// =============================================================================
// Shift Operations
// =============================================================================

impl BigInt {
    /// Left-shifts the value by `amount` bits.
    ///
    /// Shifts exceeding 127 bits produce zero. The signedness flag is preserved.
    pub fn shl(&self, amount: u32) -> BigInt {
        if amount >= 128 {
            return BigInt {
                value: 0,
                is_unsigned: self.is_unsigned,
            };
        }
        BigInt {
            value: self.value.wrapping_shl(amount),
            is_unsigned: self.is_unsigned,
        }
    }

    /// Right-shifts the value by `amount` bits.
    ///
    /// For signed values, this performs an arithmetic right shift (sign-extending).
    /// For unsigned values, this performs a logical right shift (zero-filling).
    /// Shifts exceeding 127 bits produce zero (unsigned) or 0/-1 (signed).
    pub fn shr(&self, amount: u32) -> BigInt {
        if amount >= 128 {
            if self.is_unsigned {
                return BigInt {
                    value: 0,
                    is_unsigned: true,
                };
            } else {
                // Arithmetic shift: fills with sign bit
                let fill = if self.value < 0 { -1i128 } else { 0i128 };
                return BigInt {
                    value: fill,
                    is_unsigned: false,
                };
            }
        }
        if self.is_unsigned {
            // Logical right shift via u128
            let shifted = (self.value as u128) >> amount;
            BigInt {
                value: shifted as i128,
                is_unsigned: true,
            }
        } else {
            // Arithmetic right shift (i128 >> naturally sign-extends in Rust)
            BigInt {
                value: self.value >> amount,
                is_unsigned: false,
            }
        }
    }
}

// =============================================================================
// C-Specific Semantics
// =============================================================================

impl BigInt {
    /// Truncates the value to the specified bit width by masking upper bits.
    ///
    /// For example, `truncate_to_width(8)` retains only the lowest 8 bits.
    /// The signedness flag is preserved. A width of 0 produces zero.
    /// A width of 128 or more returns the value unchanged.
    pub fn truncate_to_width(&self, bits: u32) -> BigInt {
        if bits == 0 {
            return BigInt {
                value: 0,
                is_unsigned: self.is_unsigned,
            };
        }
        if bits >= 128 {
            return self.clone();
        }
        let mask = (1u128 << bits) - 1;
        let truncated = (self.value as u128) & mask;
        BigInt {
            value: truncated as i128,
            is_unsigned: self.is_unsigned,
        }
    }

    /// Sign-extends a value from `from_bits` width to `to_bits` width.
    ///
    /// The value is first truncated to `from_bits`, then the sign bit
    /// (bit `from_bits - 1`) is replicated into bits `from_bits..to_bits`.
    /// If `from_bits >= to_bits` or either is zero, returns the value unchanged.
    pub fn sign_extend(&self, from_bits: u32, to_bits: u32) -> BigInt {
        if from_bits == 0 || to_bits == 0 || from_bits >= to_bits {
            return self.clone();
        }
        if from_bits >= 128 {
            return self.clone();
        }

        let from_mask = (1u128 << from_bits) - 1;
        let value = (self.value as u128) & from_mask;

        // Check the sign bit of the from_bits representation
        let sign_bit_set = (value >> (from_bits - 1)) & 1 == 1;

        let result = if sign_bit_set {
            // Fill bits [from_bits, to_bits) with 1s
            let to_mask = if to_bits >= 128 {
                u128::MAX
            } else {
                (1u128 << to_bits) - 1
            };
            let fill_mask = to_mask & !from_mask;
            (value | fill_mask) & to_mask
        } else {
            // Upper bits are already zero from truncation
            value
        };

        BigInt {
            value: result as i128,
            is_unsigned: self.is_unsigned,
        }
    }

    /// Zero-extends a value from `from_bits` width to `to_bits` width.
    ///
    /// The value is truncated to `from_bits`, with upper bits zero-filled.
    /// If `from_bits >= to_bits` or either is zero, returns the value unchanged.
    pub fn zero_extend(&self, from_bits: u32, to_bits: u32) -> BigInt {
        if from_bits == 0 || to_bits == 0 || from_bits >= to_bits {
            return self.clone();
        }
        if from_bits >= 128 {
            return self.clone();
        }

        let mask = (1u128 << from_bits) - 1;
        let value = (self.value as u128) & mask;

        BigInt {
            value: value as i128,
            is_unsigned: self.is_unsigned,
        }
    }

    /// Checks whether the value would overflow a target type of the given
    /// bit width and signedness.
    ///
    /// Returns `true` if the value cannot be represented in the target type:
    /// - For unsigned targets: value must be in \[0, 2^bits - 1\]
    /// - For signed targets: value must be in \[-2^(bits-1), 2^(bits-1) - 1\]
    pub fn would_overflow(&self, bits: u32, target_unsigned: bool) -> bool {
        if bits == 0 {
            return !self.is_zero();
        }

        if target_unsigned {
            // Target is unsigned: value must be in [0, 2^bits - 1]
            let val = if self.is_unsigned {
                self.value as u128
            } else {
                if self.value < 0 {
                    return true; // Negative value never fits in unsigned
                }
                self.value as u128
            };
            if bits >= 128 {
                false // Any u128 value fits in 128+ unsigned bits
            } else {
                val > ((1u128 << bits) - 1)
            }
        } else {
            // Target is signed: value must be in [-2^(bits-1), 2^(bits-1) - 1]
            let val = if self.is_unsigned {
                let uval = self.value as u128;
                if bits >= 128 {
                    return uval > i128::MAX as u128;
                }
                // Convert unsigned value to check against signed range
                if uval > i128::MAX as u128 {
                    return true; // Exceeds maximum possible signed i128
                }
                uval as i128
            } else {
                self.value
            };
            if bits >= 128 {
                false // Any i128 value fits in 128 signed bits
            } else {
                let min = -(1i128 << (bits - 1));
                let max = (1i128 << (bits - 1)) - 1;
                val < min || val > max
            }
        }
    }
}

// =============================================================================
// Comparison Helpers
// =============================================================================

impl BigInt {
    /// Value equality ignoring the signedness flag.
    ///
    /// Two `BigInt` values are "value-equal" if their underlying 128-bit
    /// patterns are identical, regardless of whether one is signed and the
    /// other unsigned. This differs from `PartialEq`, which also considers
    /// the `is_unsigned` flag.
    #[inline]
    pub fn eq_value(&self, other: &BigInt) -> bool {
        self.value == other.value
    }

    /// Returns `true` if `self` is less than `other` under C's "usual
    /// arithmetic conversions": if either operand is unsigned, both are
    /// compared as unsigned (u128). Otherwise, signed comparison is used.
    #[inline]
    pub fn less_than(&self, other: &BigInt) -> bool {
        if self.is_unsigned || other.is_unsigned {
            (self.value as u128) < (other.value as u128)
        } else {
            self.value < other.value
        }
    }

    /// Returns `true` if `self` is greater than `other` under C's "usual
    /// arithmetic conversions".
    #[inline]
    pub fn greater_than(&self, other: &BigInt) -> bool {
        if self.is_unsigned || other.is_unsigned {
            (self.value as u128) > (other.value as u128)
        } else {
            self.value > other.value
        }
    }
}

// =============================================================================
// Formatting
// =============================================================================

impl BigInt {
    /// Converts the value to a string in the given radix (base 2 through 36).
    ///
    /// Unsigned values are formatted as non-negative. Signed negative values
    /// are prefixed with `-`. Uses lowercase letters for digits above 9.
    ///
    /// # Panics
    ///
    /// Panics if `radix` is not in the range `2..=36`.
    pub fn to_string_radix(&self, radix: u32) -> String {
        assert!(
            (2..=36).contains(&radix),
            "radix must be in range 2..=36, got {}",
            radix
        );

        if self.is_zero() {
            return "0".to_string();
        }

        let negative = !self.is_unsigned && self.value < 0;

        // Compute magnitude as u128
        let mut magnitude = if self.is_unsigned {
            self.value as u128
        } else if self.value < 0 {
            // Wrapping negation handles i128::MIN correctly:
            // (i128::MIN as u128).wrapping_neg() == 2^127 == |i128::MIN|
            (self.value as u128).wrapping_neg()
        } else {
            self.value as u128
        };

        let radix_u128 = radix as u128;
        let mut digits = Vec::new();

        while magnitude > 0 {
            let digit = (magnitude % radix_u128) as u8;
            let c = if digit < 10 {
                b'0' + digit
            } else {
                b'a' + (digit - 10)
            };
            digits.push(c);
            magnitude /= radix_u128;
        }

        if negative {
            digits.push(b'-');
        }

        digits.reverse();
        // All bytes are valid ASCII, so this conversion is safe
        String::from_utf8(digits).expect("digits are valid ASCII")
    }
}

// =============================================================================
// Display and LowerHex Trait Implementations
// =============================================================================

impl fmt::Display for BigInt {
    /// Formats the value in decimal. Unsigned values are formatted as `u128`;
    /// signed values are formatted as `i128`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_unsigned {
            write!(f, "{}", self.value as u128)
        } else {
            write!(f, "{}", self.value)
        }
    }
}

impl fmt::LowerHex for BigInt {
    /// Formats the value in lowercase hexadecimal.
    ///
    /// Unsigned values use `u128` hex formatting. Signed negative values are
    /// shown with a `-` prefix followed by the magnitude in hex.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_unsigned {
            fmt::LowerHex::fmt(&(self.value as u128), f)
        } else if self.value < 0 {
            let magnitude = (self.value as u128).wrapping_neg();
            write!(f, "-")?;
            fmt::LowerHex::fmt(&magnitude, f)
        } else {
            fmt::LowerHex::fmt(&(self.value as u128), f)
        }
    }
}

// =============================================================================
// Arithmetic Operator Trait Implementations
// =============================================================================

impl std::ops::Add for BigInt {
    type Output = BigInt;

    /// Adds two `BigInt` values, discarding the overflow flag.
    #[inline]
    fn add(self, rhs: BigInt) -> BigInt {
        BigInt::add(&self, &rhs).0
    }
}

impl std::ops::Sub for BigInt {
    type Output = BigInt;

    /// Subtracts `rhs` from `self`, discarding the overflow flag.
    #[inline]
    fn sub(self, rhs: BigInt) -> BigInt {
        BigInt::sub(&self, &rhs).0
    }
}

impl std::ops::Mul for BigInt {
    type Output = BigInt;

    /// Multiplies two `BigInt` values, discarding the overflow flag.
    #[inline]
    fn mul(self, rhs: BigInt) -> BigInt {
        BigInt::mul(&self, &rhs).0
    }
}

impl std::ops::Div for BigInt {
    type Output = BigInt;

    /// Divides `self` by `rhs`.
    ///
    /// # Panics
    ///
    /// Panics if `rhs` is zero, following Rust's division-by-zero convention.
    #[inline]
    fn div(self, rhs: BigInt) -> BigInt {
        BigInt::div(&self, &rhs).expect("division by zero")
    }
}

impl std::ops::Rem for BigInt {
    type Output = BigInt;

    /// Computes `self % rhs`.
    ///
    /// # Panics
    ///
    /// Panics if `rhs` is zero, following Rust's remainder-by-zero convention.
    #[inline]
    fn rem(self, rhs: BigInt) -> BigInt {
        BigInt::rem(&self, &rhs).expect("remainder by zero")
    }
}

impl std::ops::Neg for BigInt {
    type Output = BigInt;

    /// Negates the value using wrapping arithmetic.
    #[inline]
    fn neg(self) -> BigInt {
        BigInt::neg(&self)
    }
}

// =============================================================================
// Bitwise Operator Trait Implementations
// =============================================================================

impl std::ops::BitAnd for BigInt {
    type Output = BigInt;

    /// Bitwise AND. If either operand is unsigned, the result is unsigned.
    #[inline]
    fn bitand(self, rhs: BigInt) -> BigInt {
        BigInt {
            value: self.value & rhs.value,
            is_unsigned: self.is_unsigned || rhs.is_unsigned,
        }
    }
}

impl std::ops::BitOr for BigInt {
    type Output = BigInt;

    /// Bitwise OR. If either operand is unsigned, the result is unsigned.
    #[inline]
    fn bitor(self, rhs: BigInt) -> BigInt {
        BigInt {
            value: self.value | rhs.value,
            is_unsigned: self.is_unsigned || rhs.is_unsigned,
        }
    }
}

impl std::ops::BitXor for BigInt {
    type Output = BigInt;

    /// Bitwise XOR. If either operand is unsigned, the result is unsigned.
    #[inline]
    fn bitxor(self, rhs: BigInt) -> BigInt {
        BigInt {
            value: self.value ^ rhs.value,
            is_unsigned: self.is_unsigned || rhs.is_unsigned,
        }
    }
}

impl std::ops::Not for BigInt {
    type Output = BigInt;

    /// Bitwise NOT (complement). Signedness is preserved.
    #[inline]
    fn not(self) -> BigInt {
        BigInt {
            value: !self.value,
            is_unsigned: self.is_unsigned,
        }
    }
}

impl std::ops::Shl<u32> for BigInt {
    type Output = BigInt;

    /// Left-shifts by `rhs` bits.
    #[inline]
    fn shl(self, rhs: u32) -> BigInt {
        BigInt::shl(&self, rhs)
    }
}

impl std::ops::Shr<u32> for BigInt {
    type Output = BigInt;

    /// Right-shifts by `rhs` bits. Arithmetic for signed, logical for unsigned.
    #[inline]
    fn shr(self, rhs: u32) -> BigInt {
        BigInt::shr(&self, rhs)
    }
}

// =============================================================================
// Ordering Trait Implementations
// =============================================================================

impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BigInt {
    /// Total ordering that is consistent with the derived `PartialEq`.
    ///
    /// Values are first compared by signedness (signed < unsigned), then by
    /// value within the same signedness class. This ensures that two `BigInt`
    /// values compare as `Equal` if and only if both their `value` and
    /// `is_unsigned` fields match.
    fn cmp(&self, other: &Self) -> Ordering {
        // First compare by signedness to be consistent with derived PartialEq
        if self.is_unsigned != other.is_unsigned {
            return self.is_unsigned.cmp(&other.is_unsigned);
        }
        // Same signedness: compare values using the appropriate interpretation
        if self.is_unsigned {
            (self.value as u128).cmp(&(other.value as u128))
        } else {
            self.value.cmp(&other.value)
        }
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    #[test]
    fn test_from_i64() {
        let v = BigInt::from_i64(-42);
        assert_eq!(v.to_i128(), -42);
        assert!(!v.is_unsigned());
    }

    #[test]
    fn test_from_u64() {
        let v = BigInt::from_u64(42);
        assert_eq!(v.to_u128(), 42);
        assert!(v.is_unsigned());
    }

    #[test]
    fn test_from_i128() {
        let v = BigInt::from_i128(i128::MIN);
        assert_eq!(v.to_i128(), i128::MIN);
        assert!(!v.is_unsigned());

        let v = BigInt::from_i128(i128::MAX);
        assert_eq!(v.to_i128(), i128::MAX);
    }

    #[test]
    fn test_from_u128() {
        let v = BigInt::from_u128(u128::MAX);
        assert_eq!(v.to_u128(), u128::MAX);
        assert!(v.is_unsigned());

        // Verify round-trip for values exceeding i128::MAX
        let big = u128::MAX - 100;
        let v = BigInt::from_u128(big);
        assert_eq!(v.to_u128(), big);
    }

    #[test]
    fn test_zero_and_one() {
        let z = BigInt::zero();
        assert!(z.is_zero());
        assert_eq!(z.to_i128(), 0);
        assert!(!z.is_unsigned());

        let o = BigInt::one();
        assert!(!o.is_zero());
        assert_eq!(o.to_i128(), 1);
    }

    // -------------------------------------------------------------------------
    // Conversion
    // -------------------------------------------------------------------------

    #[test]
    fn test_to_i64_in_range() {
        assert_eq!(BigInt::from_i64(100).to_i64(), Some(100));
        assert_eq!(BigInt::from_i64(-100).to_i64(), Some(-100));
        assert_eq!(BigInt::from_i64(i64::MIN).to_i64(), Some(i64::MIN));
        assert_eq!(BigInt::from_i64(i64::MAX).to_i64(), Some(i64::MAX));
    }

    #[test]
    fn test_to_i64_out_of_range() {
        let v = BigInt::from_i128(i128::MAX);
        assert_eq!(v.to_i64(), None);

        let v = BigInt::from_i128(i128::MIN);
        assert_eq!(v.to_i64(), None);
    }

    #[test]
    fn test_to_u64_in_range() {
        assert_eq!(BigInt::from_u64(100).to_u64(), Some(100));
        assert_eq!(BigInt::from_u64(u64::MAX).to_u64(), Some(u64::MAX));
    }

    #[test]
    fn test_to_u64_out_of_range() {
        // Negative signed value
        assert_eq!(BigInt::from_i64(-1).to_u64(), None);
        // Unsigned value exceeding u64
        let v = BigInt::from_u128(u128::MAX);
        assert_eq!(v.to_u64(), None);
    }

    #[test]
    fn test_round_trip_i64() {
        for val in [0i64, 1, -1, i64::MIN, i64::MAX, 42, -999] {
            let b = BigInt::from_i64(val);
            assert_eq!(b.to_i64(), Some(val), "round-trip failed for {}", val);
        }
    }

    #[test]
    fn test_round_trip_u128() {
        for val in [0u128, 1, u64::MAX as u128, u128::MAX, u128::MAX / 2] {
            let b = BigInt::from_u128(val);
            assert_eq!(b.to_u128(), val, "round-trip failed for {}", val);
        }
    }

    // -------------------------------------------------------------------------
    // Properties
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_zero() {
        assert!(BigInt::from_i64(0).is_zero());
        assert!(BigInt::from_u64(0).is_zero());
        assert!(!BigInt::from_i64(1).is_zero());
        assert!(!BigInt::from_i64(-1).is_zero());
    }

    #[test]
    fn test_is_negative() {
        assert!(BigInt::from_i64(-1).is_negative());
        assert!(BigInt::from_i128(i128::MIN).is_negative());
        assert!(!BigInt::from_i64(0).is_negative());
        assert!(!BigInt::from_i64(1).is_negative());
        // Unsigned values are never negative
        let v = BigInt::from_u128(u128::MAX); // bit pattern = -1 as i128
        assert!(!v.is_negative());
    }

    #[test]
    fn test_set_unsigned() {
        let mut v = BigInt::from_i64(42);
        assert!(!v.is_unsigned());
        v.set_unsigned(true);
        assert!(v.is_unsigned());
        v.set_unsigned(false);
        assert!(!v.is_unsigned());
    }

    // -------------------------------------------------------------------------
    // Arithmetic: Addition
    // -------------------------------------------------------------------------

    #[test]
    fn test_add_no_overflow() {
        let a = BigInt::from_i64(10);
        let b = BigInt::from_i64(20);
        let (result, overflow) = a.add(&b);
        assert_eq!(result.to_i128(), 30);
        assert!(!overflow);
    }

    #[test]
    fn test_add_signed_overflow() {
        let a = BigInt::from_i128(i128::MAX);
        let b = BigInt::from_i128(1);
        let (result, overflow) = a.add(&b);
        assert_eq!(result.to_i128(), i128::MIN); // wraps
        assert!(overflow);
    }

    #[test]
    fn test_add_unsigned() {
        let a = BigInt::from_u64(10);
        let b = BigInt::from_u64(20);
        let (result, overflow) = a.add(&b);
        assert_eq!(result.to_u128(), 30);
        assert!(result.is_unsigned());
        assert!(!overflow);
    }

    #[test]
    fn test_add_unsigned_overflow() {
        let a = BigInt::from_u128(u128::MAX);
        let b = BigInt::from_u128(1);
        let (result, overflow) = a.add(&b);
        assert_eq!(result.to_u128(), 0); // wraps
        assert!(overflow);
    }

    #[test]
    fn test_add_mixed_signedness() {
        // signed + unsigned = unsigned result
        let a = BigInt::from_i64(5);
        let b = BigInt::from_u64(3);
        let (result, overflow) = a.add(&b);
        assert!(result.is_unsigned());
        assert_eq!(result.to_u128(), 8);
        assert!(!overflow);
    }

    // -------------------------------------------------------------------------
    // Arithmetic: Subtraction
    // -------------------------------------------------------------------------

    #[test]
    fn test_sub_no_overflow() {
        let a = BigInt::from_i64(30);
        let b = BigInt::from_i64(10);
        let (result, overflow) = a.sub(&b);
        assert_eq!(result.to_i128(), 20);
        assert!(!overflow);
    }

    #[test]
    fn test_sub_signed_overflow() {
        let a = BigInt::from_i128(i128::MIN);
        let b = BigInt::from_i128(1);
        let (result, overflow) = a.sub(&b);
        assert_eq!(result.to_i128(), i128::MAX); // wraps
        assert!(overflow);
    }

    #[test]
    fn test_sub_unsigned_underflow() {
        let a = BigInt::from_u64(0);
        let b = BigInt::from_u64(1);
        let (result, overflow) = a.sub(&b);
        assert_eq!(result.to_u128(), u128::MAX); // wraps
        assert!(overflow);
    }

    // -------------------------------------------------------------------------
    // Arithmetic: Multiplication
    // -------------------------------------------------------------------------

    #[test]
    fn test_mul_no_overflow() {
        let a = BigInt::from_i64(6);
        let b = BigInt::from_i64(7);
        let (result, overflow) = a.mul(&b);
        assert_eq!(result.to_i128(), 42);
        assert!(!overflow);
    }

    #[test]
    fn test_mul_signed_overflow() {
        let a = BigInt::from_i128(i128::MAX);
        let b = BigInt::from_i128(2);
        let (result, overflow) = a.mul(&b);
        assert!(overflow);
        // Verify the result wraps correctly
        assert_eq!(result.to_i128(), i128::MAX.wrapping_mul(2));
    }

    #[test]
    fn test_mul_unsigned() {
        let a = BigInt::from_u64(1000);
        let b = BigInt::from_u64(2000);
        let (result, overflow) = a.mul(&b);
        assert_eq!(result.to_u128(), 2_000_000);
        assert!(!overflow);
    }

    // -------------------------------------------------------------------------
    // Arithmetic: Division and Modulo
    // -------------------------------------------------------------------------

    #[test]
    fn test_div_basic() {
        let a = BigInt::from_i64(42);
        let b = BigInt::from_i64(7);
        let result = a.div(&b);
        assert_eq!(result.unwrap().to_i128(), 6);
    }

    #[test]
    fn test_div_by_zero() {
        let a = BigInt::from_i64(42);
        let b = BigInt::from_i64(0);
        assert!(a.div(&b).is_none());
    }

    #[test]
    fn test_div_unsigned() {
        let a = BigInt::from_u128(u128::MAX);
        let b = BigInt::from_u128(2);
        let result = a.div(&b).unwrap();
        assert_eq!(result.to_u128(), u128::MAX / 2);
        assert!(result.is_unsigned());
    }

    #[test]
    fn test_div_negative() {
        let a = BigInt::from_i64(-42);
        let b = BigInt::from_i64(7);
        assert_eq!(a.div(&b).unwrap().to_i128(), -6);
    }

    #[test]
    fn test_div_min_by_neg_one() {
        // i128::MIN / -1 overflows; wrapping_div wraps to i128::MIN
        let a = BigInt::from_i128(i128::MIN);
        let b = BigInt::from_i128(-1);
        let result = a.div(&b).unwrap();
        assert_eq!(result.to_i128(), i128::MIN); // wrapping result
    }

    #[test]
    fn test_rem_basic() {
        let a = BigInt::from_i64(17);
        let b = BigInt::from_i64(5);
        assert_eq!(a.rem(&b).unwrap().to_i128(), 2);
    }

    #[test]
    fn test_rem_by_zero() {
        let a = BigInt::from_i64(42);
        let b = BigInt::from_i64(0);
        assert!(a.rem(&b).is_none());
    }

    #[test]
    fn test_rem_negative() {
        let a = BigInt::from_i64(-17);
        let b = BigInt::from_i64(5);
        assert_eq!(a.rem(&b).unwrap().to_i128(), -2);
    }

    #[test]
    fn test_rem_unsigned() {
        let a = BigInt::from_u64(17);
        let b = BigInt::from_u64(5);
        let result = a.rem(&b).unwrap();
        assert_eq!(result.to_u128(), 2);
        assert!(result.is_unsigned());
    }

    // -------------------------------------------------------------------------
    // Arithmetic: Negation
    // -------------------------------------------------------------------------

    #[test]
    fn test_neg() {
        let v = BigInt::from_i64(42);
        let neg_v = v.neg();
        assert_eq!(neg_v.to_i128(), -42);
    }

    #[test]
    fn test_neg_zero() {
        let v = BigInt::from_i64(0);
        let neg_v = v.neg();
        assert!(neg_v.is_zero());
    }

    #[test]
    fn test_neg_min() {
        // Negating i128::MIN wraps to i128::MIN
        let v = BigInt::from_i128(i128::MIN);
        let neg_v = v.neg();
        assert_eq!(neg_v.to_i128(), i128::MIN);
    }

    #[test]
    fn test_neg_trait() {
        let v = BigInt::from_i64(42);
        let neg_v = -v;
        assert_eq!(neg_v.to_i128(), -42);
    }

    // -------------------------------------------------------------------------
    // Bitwise Operations
    // -------------------------------------------------------------------------

    #[test]
    fn test_bitand() {
        let a = BigInt::from_i64(0b1100);
        let b = BigInt::from_i64(0b1010);
        let result = a & b;
        assert_eq!(result.to_i128(), 0b1000);
    }

    #[test]
    fn test_bitor() {
        let a = BigInt::from_i64(0b1100);
        let b = BigInt::from_i64(0b1010);
        let result = a | b;
        assert_eq!(result.to_i128(), 0b1110);
    }

    #[test]
    fn test_bitxor() {
        let a = BigInt::from_i64(0b1100);
        let b = BigInt::from_i64(0b1010);
        let result = a ^ b;
        assert_eq!(result.to_i128(), 0b0110);
    }

    #[test]
    fn test_bitnot() {
        let v = BigInt::from_i64(0);
        let result = !v;
        assert_eq!(result.to_i128(), -1); // all bits set
    }

    #[test]
    fn test_bitwise_unsigned_propagation() {
        // If either operand is unsigned, result is unsigned
        let a = BigInt::from_i64(0xFF);
        let b = BigInt::from_u64(0x0F);
        let result = a & b;
        assert!(result.is_unsigned());
        assert_eq!(result.to_u128(), 0x0F);
    }

    // -------------------------------------------------------------------------
    // Shift Operations
    // -------------------------------------------------------------------------

    #[test]
    fn test_shl() {
        let v = BigInt::from_i64(1);
        let result = v.shl(10);
        assert_eq!(result.to_i128(), 1024);
    }

    #[test]
    fn test_shl_trait() {
        let v = BigInt::from_i64(3);
        let result = v << 4u32;
        assert_eq!(result.to_i128(), 48);
    }

    #[test]
    fn test_shl_overflow() {
        let v = BigInt::from_i64(1);
        let result = v.shl(128);
        assert!(result.is_zero());
    }

    #[test]
    fn test_shr_signed() {
        // Arithmetic right shift preserves sign
        let v = BigInt::from_i64(-16);
        let result = v.shr(2);
        assert_eq!(result.to_i128(), -4); // -16 >> 2 = -4 (arithmetic)
    }

    #[test]
    fn test_shr_unsigned() {
        // Logical right shift zero-fills
        let mut v = BigInt::from_i64(-1);
        v.set_unsigned(true); // now represents u128::MAX
        let result = v.shr(1);
        assert_eq!(result.to_u128(), u128::MAX >> 1);
    }

    #[test]
    fn test_shr_signed_large_shift() {
        let v = BigInt::from_i64(-1);
        let result = v.shr(200); // shift >= 128
        assert_eq!(result.to_i128(), -1); // sign-extended fill
    }

    #[test]
    fn test_shr_unsigned_large_shift() {
        let v = BigInt::from_u64(42);
        let result = v.shr(200); // shift >= 128
        assert!(result.is_zero()); // zero-filled
    }

    #[test]
    fn test_shr_trait() {
        let v = BigInt::from_i64(1024);
        let result = v >> 3u32;
        assert_eq!(result.to_i128(), 128);
    }

    // -------------------------------------------------------------------------
    // Comparison
    // -------------------------------------------------------------------------

    #[test]
    fn test_ord_signed() {
        let a = BigInt::from_i64(-1);
        let b = BigInt::from_i64(1);
        assert!(a < b);
        assert!(b > a);
    }

    #[test]
    fn test_ord_unsigned() {
        let a = BigInt::from_u128(0);
        let b = BigInt::from_u128(u128::MAX);
        assert!(a < b);
    }

    #[test]
    fn test_eq_value() {
        let a = BigInt::from_i64(42);
        let b = BigInt::from_u64(42);
        // PartialEq considers signedness
        assert_ne!(a, b);
        // eq_value ignores signedness
        assert!(a.eq_value(&b));
    }

    #[test]
    fn test_less_than_c_semantics() {
        // C: when comparing signed vs unsigned, convert signed to unsigned
        let neg = BigInt::from_i64(-1);
        let pos = BigInt::from_u64(0);
        // -1 as u128 = u128::MAX, which is > 0
        assert!(!neg.less_than(&pos));
        assert!(pos.less_than(&neg));
    }

    #[test]
    fn test_greater_than() {
        let a = BigInt::from_i64(10);
        let b = BigInt::from_i64(5);
        assert!(a.greater_than(&b));
        assert!(!b.greater_than(&a));
    }

    // -------------------------------------------------------------------------
    // C-Specific Semantics
    // -------------------------------------------------------------------------

    #[test]
    fn test_truncate_to_width_8() {
        let v = BigInt::from_i64(0x1FF); // 511
        let truncated = v.truncate_to_width(8);
        assert_eq!(truncated.to_u128(), 0xFF);
    }

    #[test]
    fn test_truncate_to_width_16() {
        let v = BigInt::from_i64(0x1_FFFF); // 131071
        let truncated = v.truncate_to_width(16);
        assert_eq!(truncated.to_u128(), 0xFFFF);
    }

    #[test]
    fn test_truncate_to_width_32() {
        let v = BigInt::from_i128(0x1_FFFF_FFFF);
        let truncated = v.truncate_to_width(32);
        assert_eq!(truncated.to_u128(), 0xFFFF_FFFF);
    }

    #[test]
    fn test_truncate_to_width_zero() {
        let v = BigInt::from_i64(42);
        let truncated = v.truncate_to_width(0);
        assert!(truncated.is_zero());
    }

    #[test]
    fn test_truncate_to_width_128_or_more() {
        let v = BigInt::from_i128(i128::MAX);
        let truncated = v.truncate_to_width(128);
        assert_eq!(truncated.to_i128(), i128::MAX);
        let truncated = v.truncate_to_width(200);
        assert_eq!(truncated.to_i128(), i128::MAX);
    }

    #[test]
    fn test_sign_extend_8_to_32() {
        // Value 0xFF in 8 bits: sign bit is set → extend with 1s
        let v = BigInt::from_i64(0xFF);
        let extended = v.sign_extend(8, 32);
        assert_eq!(extended.to_u128(), 0xFFFF_FFFF);
    }

    #[test]
    fn test_sign_extend_positive() {
        // Value 0x7F in 8 bits: sign bit is clear → extend with 0s
        let v = BigInt::from_i64(0x7F);
        let extended = v.sign_extend(8, 32);
        assert_eq!(extended.to_u128(), 0x7F);
    }

    #[test]
    fn test_sign_extend_16_to_64() {
        let v = BigInt::from_i64(0x8000); // 16-bit negative
        let extended = v.sign_extend(16, 64);
        assert_eq!(extended.to_u128(), 0xFFFF_FFFF_FFFF_8000);
    }

    #[test]
    fn test_sign_extend_noop() {
        // from_bits >= to_bits should return unchanged
        let v = BigInt::from_i64(42);
        let extended = v.sign_extend(32, 16);
        assert_eq!(extended.to_i128(), 42);
    }

    #[test]
    fn test_zero_extend_8_to_32() {
        let v = BigInt::from_i64(0xFF);
        let extended = v.zero_extend(8, 32);
        assert_eq!(extended.to_u128(), 0xFF);
    }

    #[test]
    fn test_zero_extend_strips_upper_bits() {
        // Value with bits above from_bits set
        let v = BigInt::from_i64(0x1FF); // 9 bits
        let extended = v.zero_extend(8, 32);
        assert_eq!(extended.to_u128(), 0xFF); // only lower 8 bits preserved
    }

    #[test]
    fn test_would_overflow_unsigned_8() {
        let v = BigInt::from_i64(255);
        assert!(!v.would_overflow(8, true)); // 255 fits in u8
        let v = BigInt::from_i64(256);
        assert!(v.would_overflow(8, true)); // 256 doesn't fit in u8
    }

    #[test]
    fn test_would_overflow_signed_8() {
        let v = BigInt::from_i64(127);
        assert!(!v.would_overflow(8, false)); // 127 fits in i8
        let v = BigInt::from_i64(128);
        assert!(v.would_overflow(8, false)); // 128 doesn't fit in i8
        let v = BigInt::from_i64(-128);
        assert!(!v.would_overflow(8, false)); // -128 fits in i8
        let v = BigInt::from_i64(-129);
        assert!(v.would_overflow(8, false)); // -129 doesn't fit in i8
    }

    #[test]
    fn test_would_overflow_negative_unsigned() {
        // Negative value never fits in unsigned target
        let v = BigInt::from_i64(-1);
        assert!(v.would_overflow(64, true));
    }

    #[test]
    fn test_would_overflow_zero_bits() {
        // Only zero fits in 0 bits
        assert!(!BigInt::from_i64(0).would_overflow(0, true));
        assert!(BigInt::from_i64(1).would_overflow(0, true));
    }

    #[test]
    fn test_would_overflow_128_bits() {
        // Any i128 value fits in 128 signed bits
        assert!(!BigInt::from_i128(i128::MAX).would_overflow(128, false));
        assert!(!BigInt::from_i128(i128::MIN).would_overflow(128, false));
        // u128 values fit in 128 unsigned bits
        assert!(!BigInt::from_u128(u128::MAX).would_overflow(128, true));
    }

    // -------------------------------------------------------------------------
    // Formatting
    // -------------------------------------------------------------------------

    #[test]
    fn test_display_signed() {
        assert_eq!(format!("{}", BigInt::from_i64(42)), "42");
        assert_eq!(format!("{}", BigInt::from_i64(-42)), "-42");
        assert_eq!(format!("{}", BigInt::from_i64(0)), "0");
    }

    #[test]
    fn test_display_unsigned() {
        assert_eq!(format!("{}", BigInt::from_u64(42)), "42");
        assert_eq!(
            format!("{}", BigInt::from_u128(u128::MAX)),
            u128::MAX.to_string()
        );
    }

    #[test]
    fn test_lower_hex() {
        assert_eq!(format!("{:x}", BigInt::from_i64(255)), "ff");
        assert_eq!(format!("{:x}", BigInt::from_i64(-1)), "-1");
        assert_eq!(format!("{:x}", BigInt::from_u64(255)), "ff");
    }

    #[test]
    fn test_lower_hex_unsigned_max() {
        let v = BigInt::from_u128(u128::MAX);
        assert_eq!(format!("{:x}", v), "ffffffffffffffffffffffffffffffff");
    }

    #[test]
    fn test_to_string_radix_decimal() {
        assert_eq!(BigInt::from_i64(42).to_string_radix(10), "42");
        assert_eq!(BigInt::from_i64(-42).to_string_radix(10), "-42");
        assert_eq!(BigInt::from_i64(0).to_string_radix(10), "0");
    }

    #[test]
    fn test_to_string_radix_binary() {
        assert_eq!(BigInt::from_i64(10).to_string_radix(2), "1010");
    }

    #[test]
    fn test_to_string_radix_hex() {
        assert_eq!(BigInt::from_i64(255).to_string_radix(16), "ff");
        assert_eq!(BigInt::from_i64(-255).to_string_radix(16), "-ff");
    }

    #[test]
    fn test_to_string_radix_octal() {
        assert_eq!(BigInt::from_i64(8).to_string_radix(8), "10");
    }

    #[test]
    fn test_to_string_radix_base36() {
        assert_eq!(BigInt::from_i64(35).to_string_radix(36), "z");
    }

    #[test]
    #[should_panic(expected = "radix must be in range 2..=36")]
    fn test_to_string_radix_invalid() {
        BigInt::from_i64(0).to_string_radix(1);
    }

    // -------------------------------------------------------------------------
    // Operator Traits
    // -------------------------------------------------------------------------

    #[test]
    fn test_add_trait() {
        let result = BigInt::from_i64(10) + BigInt::from_i64(20);
        assert_eq!(result.to_i128(), 30);
    }

    #[test]
    fn test_sub_trait() {
        let result = BigInt::from_i64(30) - BigInt::from_i64(10);
        assert_eq!(result.to_i128(), 20);
    }

    #[test]
    fn test_mul_trait() {
        let result = BigInt::from_i64(6) * BigInt::from_i64(7);
        assert_eq!(result.to_i128(), 42);
    }

    #[test]
    fn test_div_trait() {
        let result = BigInt::from_i64(42) / BigInt::from_i64(7);
        assert_eq!(result.to_i128(), 6);
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn test_div_trait_by_zero() {
        let _ = BigInt::from_i64(1) / BigInt::from_i64(0);
    }

    #[test]
    fn test_rem_trait() {
        let result = BigInt::from_i64(17) % BigInt::from_i64(5);
        assert_eq!(result.to_i128(), 2);
    }

    #[test]
    #[should_panic(expected = "remainder by zero")]
    fn test_rem_trait_by_zero() {
        let _ = BigInt::from_i64(1) % BigInt::from_i64(0);
    }

    // -------------------------------------------------------------------------
    // Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_i128_min_operations() {
        let min = BigInt::from_i128(i128::MIN);
        assert!(min.is_negative());
        assert!(!min.is_zero());
        assert_eq!(min.to_i128(), i128::MIN);

        // Display
        assert_eq!(format!("{}", min), i128::MIN.to_string());
    }

    #[test]
    fn test_i128_max_operations() {
        let max = BigInt::from_i128(i128::MAX);
        assert!(!max.is_negative());
        assert!(!max.is_zero());
        assert_eq!(max.to_i128(), i128::MAX);
    }

    #[test]
    fn test_u128_max_operations() {
        let max = BigInt::from_u128(u128::MAX);
        assert!(!max.is_negative()); // unsigned, so not negative
        assert!(!max.is_zero());
        assert_eq!(max.to_u128(), u128::MAX);
        assert_eq!(format!("{}", max), u128::MAX.to_string());
    }

    #[test]
    fn test_display_consistency_with_to_string_radix() {
        for val in [0i64, 1, -1, 42, -42, i64::MAX, i64::MIN] {
            let b = BigInt::from_i64(val);
            assert_eq!(
                format!("{}", b),
                b.to_string_radix(10),
                "Display vs to_string_radix(10) mismatch for {}",
                val
            );
        }
    }

    #[test]
    fn test_truncate_then_sign_extend_roundtrip() {
        // Start with a 32-bit signed value, truncate to 8, then sign-extend back to 32
        let v = BigInt::from_i64(-5); // 0xFFFB in 16 bits; 0xFB in 8 bits
        let truncated = v.truncate_to_width(8);
        let extended = truncated.sign_extend(8, 32);
        // 0xFB sign-extended from 8 to 32 bits = 0xFFFF_FFFB
        assert_eq!(extended.to_u128() & 0xFFFF_FFFF, 0xFFFF_FFFB);
    }

    #[test]
    fn test_mixed_arithmetic_chain() {
        // (10 + 20) * 3 - 5 = 85
        let a = BigInt::from_i64(10);
        let b = BigInt::from_i64(20);
        let c = BigInt::from_i64(3);
        let d = BigInt::from_i64(5);

        let sum = (a + b) * c;
        let result = sum - d;
        assert_eq!(result.to_i128(), 85);
    }

    #[test]
    fn test_bitwise_mask_pattern() {
        // Build a 16-bit mask: (1 << 16) - 1
        let one = BigInt::from_i64(1);
        let shifted = one.shl(16);
        let mask = shifted - BigInt::from_i64(1);
        assert_eq!(mask.to_i128(), 0xFFFF);
    }

    #[test]
    fn test_ord_consistency_with_eq() {
        // Verify Ord is consistent with PartialEq
        let a = BigInt::from_i64(42);
        let b = BigInt::from_i64(42);
        assert_eq!(a.cmp(&b), Ordering::Equal);
        assert_eq!(a, b);

        let c = BigInt::from_u64(42);
        assert_ne!(a, c); // different signedness
        assert_ne!(a.cmp(&c), Ordering::Equal);
    }
}
