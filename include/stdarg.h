/*
 * stdarg.h — Variable argument list handling (C11 §7.16)
 *
 * Freestanding header for the bcc (Blitzy C Compiler).
 * Provides va_list type and macros for accessing variable arguments
 * in variadic functions. All operations delegate to bcc compiler
 * builtins which generate architecture-specific code for:
 *
 *   - x86-64  (System V AMD64 ABI)
 *   - i686    (cdecl / System V i386 ABI)
 *   - AArch64 (AAPCS64)
 *   - RISC-V 64 (LP64D)
 *
 * The compiler backend handles the distinct va_list memory layouts
 * required by each target ABI. This header is intentionally
 * self-contained with no dependencies on other headers.
 */

/*
 * Support the __need___va_list partial-include pattern used by glibc's
 * <stdio.h>.  When __need___va_list is defined, only __gnuc_va_list is
 * provided so that <stdio.h> can declare vfprintf() and friends without
 * pulling in the full va_start/va_arg/va_end machinery.
 */
#ifndef __GNUC_VA_LIST
#define __GNUC_VA_LIST
typedef __builtin_va_list __gnuc_va_list;
#endif

#ifdef __need___va_list
#undef __need___va_list
/* Only __gnuc_va_list was needed — stop here. */
#else

#ifndef _STDARG_H
#define _STDARG_H

/*
 * va_list — Opaque type for traversing variable argument lists.
 *
 * The underlying __builtin_va_list is defined by the compiler and
 * its layout varies per target architecture:
 *
 *   x86-64:   struct { unsigned gp_offset; unsigned fp_offset;
 *                      void *overflow_arg_area; void *reg_save_area; }
 *   i686:     char *  (pointer into the stack frame)
 *   AArch64:  struct { void *__stack; void *__gr_top;
 *                      void *__vr_top; int __gr_offs; int __vr_offs; }
 *   RISC-V:   void *  (pointer into the stack frame)
 */
typedef __builtin_va_list va_list;

/*
 * va_start — Initialize a va_list for argument traversal (C11 §7.16.1.4).
 *
 * Must be called before any va_arg invocations. The parameter 'param'
 * shall be the identifier of the rightmost named parameter in the
 * function's parameter list.
 *
 * Usage:
 *   void my_printf(const char *fmt, ...) {
 *       va_list ap;
 *       va_start(ap, fmt);
 *       // ... use va_arg(ap, type) ...
 *       va_end(ap);
 *   }
 */
#define va_start(ap, param) __builtin_va_start(ap, param)

/*
 * va_arg — Retrieve the next argument from a va_list (C11 §7.16.1.1).
 *
 * Each call advances the va_list to the next argument. The 'type'
 * parameter must match the actual type of the next argument after
 * default argument promotions. Behavior is undefined if the type
 * does not match or if va_arg is called after all arguments have
 * been consumed.
 */
#define va_arg(ap, type) __builtin_va_arg(ap, type)

/*
 * va_end — Clean up a va_list after traversal (C11 §7.16.1.3).
 *
 * Must be called before the function returns. After va_end, the
 * va_list is invalid and must not be used unless re-initialized
 * with va_start or va_copy.
 */
#define va_end(ap) __builtin_va_end(ap)

/*
 * va_copy — Duplicate a va_list's current state (C11 §7.16.1.2).
 *
 * Creates an independent copy of 'src' into 'dest'. Both va_lists
 * can then be used independently. va_end must be called on 'dest'
 * before the function returns, just as with any va_list initialized
 * via va_start.
 *
 * This macro was introduced in C99 and is required by C11. It is
 * essential for functions that need to traverse a variable argument
 * list multiple times.
 */
#define va_copy(dest, src) __builtin_va_copy(dest, src)

#endif /* _STDARG_H */
#endif /* !__need___va_list */
