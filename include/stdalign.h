/*
 * stdalign.h — Alignment macros (C11 freestanding header)
 *
 * Defines convenience macros 'alignas' and 'alignof' that map to the
 * C11 keywords '_Alignas' and '_Alignof', as specified by C11 §7.15.
 *
 * This header is part of the bcc bundled freestanding header set and
 * has no dependencies on system headers or other bundled headers.
 * Content is architecture-independent.
 */

#ifndef _STDALIGN_H
#define _STDALIGN_H

/* C11 §7.15p2 — alignas expands to _Alignas */
#define alignas _Alignas

/* C11 §7.15p3 — alignof expands to _Alignof */
#define alignof _Alignof

/* C11 §7.15p4 — defined to 1 to indicate availability */
#define __alignas_is_defined 1
#define __alignof_is_defined 1

#endif /* _STDALIGN_H */
