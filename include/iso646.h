/*
 * iso646.h - Alternative operator spellings
 *
 * C11 Standard (ISO/IEC 9899:2011), Section 7.9
 * Freestanding header for the bcc compiler.
 *
 * This header defines macros that expand to the corresponding
 * C tokens for logical and bitwise operators, providing
 * alternative spellings as specified by the C11 standard.
 *
 * Architecture-independent: identical content for x86-64, i686,
 * AArch64, and RISC-V 64 targets.
 */

#ifndef _ISO646_H
#define _ISO646_H

#define and    &&
#define and_eq &=
#define bitand &
#define bitor  |
#define compl  ~
#define not    !
#define not_eq !=
#define or     ||
#define or_eq  |=
#define xor    ^
#define xor_eq ^=

#endif /* _ISO646_H */
