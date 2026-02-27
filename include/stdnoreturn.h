/*
 * stdnoreturn.h — C11 Noreturn Convenience Macro (Freestanding Header)
 *
 * Part of bcc (Blitzy C Compiler)
 *
 * Conforms to ISO/IEC 9899:2011 (C11) §7.23:
 *   The header <stdnoreturn.h> defines the macro
 *       noreturn
 *   which expands to _Noreturn.
 *
 * _Noreturn is a C11 function specifier (§6.7.4) indicating that the
 * function does not return to its caller. This header provides the
 * convenience spelling "noreturn" as a macro.
 *
 * This header is self-contained with no dependencies on other headers.
 * Content is architecture-independent and valid for all four bcc targets:
 *   x86-64, i686, AArch64, RISC-V 64
 *
 * Usage example:
 *   #include <stdnoreturn.h>
 *   noreturn void fatal_error(const char *msg);
 */

#ifndef _STDNORETURN_H
#define _STDNORETURN_H

/* C11 §7.23: noreturn convenience macro mapping to _Noreturn */
#define noreturn _Noreturn

#endif /* _STDNORETURN_H */
