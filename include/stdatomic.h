/* stdatomic.h — C11 atomic operations (freestanding stub)
 *
 * This bundled header provides the C11 <stdatomic.h> type definitions and
 * memory-ordering constants required to compile real-world codebases such
 * as Redis and SQLite.  Because the BCC compiler does not implement true
 * hardware-atomic instruction emission, _Atomic(T) is mapped to the
 * underlying type T and all atomic operations are defined as simple
 * non-atomic read/modify/write macros.  This is sufficient for
 * single-threaded compilation and link-time validation.
 */
#ifndef __BCC_STDATOMIC_H
#define __BCC_STDATOMIC_H

/* ------------------------------------------------------------------ */
/* Memory ordering constants (C11 §7.17.1)                            */
/* ------------------------------------------------------------------ */
typedef enum {
    memory_order_relaxed = 0,
    memory_order_consume = 1,
    memory_order_acquire = 2,
    memory_order_release = 3,
    memory_order_acq_rel = 4,
    memory_order_seq_cst = 5
} memory_order;

/* ------------------------------------------------------------------ */
/* _Atomic type qualifier                                             */
/* ------------------------------------------------------------------ */
/* Map _Atomic(T) to T — our compiler treats _Atomic as a no-op      */
/* qualifier.  Individual atomic_* typedefs are provided for the      */
/* standard integer types used by glibc and application code.         */
/* ------------------------------------------------------------------ */

typedef _Bool              atomic_bool;
typedef char               atomic_char;
typedef signed char        atomic_schar;
typedef unsigned char      atomic_uchar;
typedef short              atomic_short;
typedef unsigned short     atomic_ushort;
typedef int                atomic_int;
typedef unsigned int       atomic_uint;
typedef long               atomic_long;
typedef unsigned long      atomic_ulong;
typedef long long          atomic_llong;
typedef unsigned long long atomic_ullong;
typedef int                atomic_size_t;
typedef int                atomic_ptrdiff_t;
typedef int                atomic_intmax_t;
typedef unsigned int       atomic_uintmax_t;
typedef int                atomic_intptr_t;
typedef unsigned int       atomic_uintptr_t;

/* Atomic flag type */
typedef struct { int __val; } atomic_flag;

#define ATOMIC_FLAG_INIT { 0 }
#define ATOMIC_VAR_INIT(value) (value)

#define ATOMIC_BOOL_LOCK_FREE     2
#define ATOMIC_CHAR_LOCK_FREE     2
#define ATOMIC_CHAR16_T_LOCK_FREE 2
#define ATOMIC_CHAR32_T_LOCK_FREE 2
#define ATOMIC_WCHAR_T_LOCK_FREE  2
#define ATOMIC_SHORT_LOCK_FREE    2
#define ATOMIC_INT_LOCK_FREE      2
#define ATOMIC_LONG_LOCK_FREE     2
#define ATOMIC_LLONG_LOCK_FREE    2
#define ATOMIC_POINTER_LOCK_FREE  2

/* ------------------------------------------------------------------ */
/* Atomic operations (non-atomic stubs for compilation purposes)      */
/* ------------------------------------------------------------------ */
#define atomic_init(obj, value)             (*(obj) = (value))
#define atomic_store(obj, value)            (*(obj) = (value))
#define atomic_store_explicit(obj, value, order) (*(obj) = (value))
#define atomic_load(obj)                    (*(obj))
#define atomic_load_explicit(obj, order)    (*(obj))
#define atomic_exchange(obj, desired)       __atomic_exchange_n(obj, desired)
#define atomic_exchange_explicit(obj, desired, order) __atomic_exchange_n(obj, desired)

/* Fetch-and-modify operations */
#define atomic_fetch_add(obj, arg)          __atomic_fetch_add(obj, arg)
#define atomic_fetch_add_explicit(obj, arg, order) __atomic_fetch_add(obj, arg)
#define atomic_fetch_sub(obj, arg)          __atomic_fetch_sub(obj, arg)
#define atomic_fetch_sub_explicit(obj, arg, order) __atomic_fetch_sub(obj, arg)
#define atomic_fetch_or(obj, arg)           __atomic_fetch_or(obj, arg)
#define atomic_fetch_or_explicit(obj, arg, order) __atomic_fetch_or(obj, arg)
#define atomic_fetch_and(obj, arg)          __atomic_fetch_and(obj, arg)
#define atomic_fetch_and_explicit(obj, arg, order) __atomic_fetch_and(obj, arg)
#define atomic_fetch_xor(obj, arg)          __atomic_fetch_xor(obj, arg)
#define atomic_fetch_xor_explicit(obj, arg, order) __atomic_fetch_xor(obj, arg)

/* Compare-and-swap */
#define atomic_compare_exchange_strong(obj, expected, desired) \
    __atomic_compare_exchange(obj, expected, desired)
#define atomic_compare_exchange_strong_explicit(obj, expected, desired, succ, fail) \
    __atomic_compare_exchange(obj, expected, desired)
#define atomic_compare_exchange_weak(obj, expected, desired) \
    __atomic_compare_exchange(obj, expected, desired)
#define atomic_compare_exchange_weak_explicit(obj, expected, desired, succ, fail) \
    __atomic_compare_exchange(obj, expected, desired)

/* Atomic flag operations */
#define atomic_flag_test_and_set(obj)       __atomic_flag_test_and_set(obj)
#define atomic_flag_test_and_set_explicit(obj, order) __atomic_flag_test_and_set(obj)
#define atomic_flag_clear(obj)              ((obj)->__val = 0)
#define atomic_flag_clear_explicit(obj, order) ((obj)->__val = 0)

/* Fence */
#define atomic_thread_fence(order)          ((void)0)
#define atomic_signal_fence(order)          ((void)0)

/* Kill dependency — returns the value unchanged */
#define kill_dependency(y)                  (y)

/* Builtin helpers used by the macros above — implemented as simple
 * non-atomic operations since BCC doesn't emit real atomic instructions. */
#define __atomic_exchange_n(ptr, val) \
    ({ __typeof__(*(ptr)) __old = *(ptr); *(ptr) = (val); __old; })
#define __atomic_fetch_add(ptr, val) \
    ({ __typeof__(*(ptr)) __old = *(ptr); *(ptr) += (val); __old; })
#define __atomic_fetch_sub(ptr, val) \
    ({ __typeof__(*(ptr)) __old = *(ptr); *(ptr) -= (val); __old; })
#define __atomic_fetch_or(ptr, val) \
    ({ __typeof__(*(ptr)) __old = *(ptr); *(ptr) |= (val); __old; })
#define __atomic_fetch_and(ptr, val) \
    ({ __typeof__(*(ptr)) __old = *(ptr); *(ptr) &= (val); __old; })
#define __atomic_fetch_xor(ptr, val) \
    ({ __typeof__(*(ptr)) __old = *(ptr); *(ptr) ^= (val); __old; })
#define __atomic_compare_exchange(ptr, expected, desired) \
    ({ int __r = (*(ptr) == *(expected)); if (__r) *(ptr) = (desired); else *(expected) = *(ptr); __r; })
#define __atomic_flag_test_and_set(obj) \
    ({ int __old = (obj)->__val; (obj)->__val = 1; __old; })

#endif /* __BCC_STDATOMIC_H */
