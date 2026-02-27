/*
 * stdbool.h — Boolean type and values (C11 freestanding header)
 *
 * Conforms to ISO/IEC 9899:2011 (C11) §7.18.
 *
 * This is a bundled freestanding header shipped with the bcc compiler.
 * It is self-contained with no dependencies on system headers or other
 * bundled headers, and its content is architecture-independent.
 *
 * Per C11 §7.18:
 *   - bool   expands to _Bool
 *   - true   expands to the integer constant 1
 *   - false  expands to the integer constant 0
 *   - __bool_true_false_are_defined expands to the integer constant 1
 *
 * Note: _Bool is a built-in C11 keyword recognized by the bcc lexer.
 * The macros bool, true, and false are intentionally defined as macros
 * (not keywords) per the C11 standard specification.
 */

#ifndef _STDBOOL_H
#define _STDBOOL_H

/* C11 §7.18p2: The macro bool expands to _Bool. */
#define bool  _Bool

/* C11 §7.18p3: The remaining three macros are suitable for use
   in #if preprocessing directives. */

/* The macro true expands to the integer constant 1. */
#define true  1

/* The macro false expands to the integer constant 0. */
#define false 0

/* C11 §7.18p4: The macro __bool_true_false_are_defined expands
   to the integer constant 1. */
#define __bool_true_false_are_defined 1

#endif /* _STDBOOL_H */
