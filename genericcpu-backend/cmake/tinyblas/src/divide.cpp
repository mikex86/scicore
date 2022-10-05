#include "divide.h"
#include "forceinline.h"

template<typename A, typename B>
FORCE_INLINE void tblas_tensor_gedivide_inplace(A *a, B b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] /= b;
    }
}

template<typename A, typename B>
FORCE_INLINE void tblas_tensor_gedivide_inplace(A *a, const B *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] /= b[i];
    }
}

template<typename A, typename B, typename C>
FORCE_INLINE void tblas_tensor_gedivide(const A *a, B b, C *c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] / b;
    }
}

template<typename A, typename B, typename C>
FORCE_INLINE void tblas_tensor_gedivide(A a, const B *b, C *c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a / b[i];
    }
}

template<typename A, typename B, typename C>
FORCE_INLINE void tblas_tensor_gedivide(const A *a, const B *b, C *c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] / b[i];
    }
}

template<typename A, typename B>
FORCE_INLINE void tblas_tensor_gedivide_broadcast_inplace(A *a, const B *b, size_t n, size_t r) {
    for (size_t i = 0; i < n; i++) {
        size_t j = i % r;
        a[i] /= b[j];
    }
}

template<typename A, typename B, typename C>
FORCE_INLINE void tblas_tensor_gedivide_broadcast_right(const A *a, const B *b, C *c, size_t na, size_t nb, size_t nc, size_t p) {
    for (size_t i = 0; i < nc; i++) {
        size_t aIdx = i % na;
        size_t bIdx = (i / p) % nb;
        c[i] = a[aIdx] / b[bIdx];
    }
}

template<typename A, typename B, typename C>
FORCE_INLINE void tblas_tensor_gedivide_broadcast_left(const A *a, const B *b, C *c, size_t na, size_t nb, size_t nc, size_t p) {
    for (size_t i = 0; i < nc; i++) {
        size_t aIdx = (i / p) % na;
        size_t bIdx = i % nb;
        c[i] = a[aIdx] / b[bIdx];
    }
}

BINARY_OP_FOR_ALL_TYPES_ALL_VARIANTS_IMPL(divide)