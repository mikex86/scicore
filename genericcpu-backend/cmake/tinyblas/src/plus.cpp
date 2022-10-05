#include "minus.h"
#include "forceinline.h"

template<typename A, typename B>
FORCE_INLINE void tblas_tensor_geplus_inplace(A *a, B b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] += b;
    }
}

template<typename A, typename B>
FORCE_INLINE void tblas_tensor_geplus_inplace(A *a, const B *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] += b[i];
    }
}

template<typename A, typename B, typename C>
FORCE_INLINE void tblas_tensor_geplus(const A *a, B b, C *c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + b;
    }
}

template<typename A, typename B, typename C>
FORCE_INLINE void tblas_tensor_geplus(A a, const B *b, C *c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a + b[i];
    }
}

template<typename A, typename B, typename C>
FORCE_INLINE void tblas_tensor_geplus(const A *a, const B *b, C *c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

template<typename A, typename B>
FORCE_INLINE void tblas_tensor_geplus_broadcast_inplace(A *a, const B *b, size_t n, size_t r) {
    for (size_t i = 0; i < n; i++) {
        size_t j = i % r;
        a[i] += b[j];
    }
}

template<typename A, typename B, typename C>
FORCE_INLINE void tblas_tensor_geplus_broadcast_right(const A *a, const B *b, C *c, size_t n, size_t r) {
    for (size_t i = 0; i < n; i++) {
        size_t j = i % r;
        c[i] = a[i] + b[j];
    }
}

template<typename A, typename B, typename C>
FORCE_INLINE void tblas_tensor_geplus_broadcast_left(const A *a, const B *b, C *c, size_t n, size_t r) {
    for (size_t i = 0; i < n; i++) {
        size_t j = i % r;
        c[i] = a[j] + b[i];
    }
}

BINARY_OP_FOR_ALL_TYPES_ALL_VARIANTS_IMPL(plus);