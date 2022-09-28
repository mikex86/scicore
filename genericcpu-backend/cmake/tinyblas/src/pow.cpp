#include "pow.h"
#include "forceinline.h"
#include <cmath>

template<typename A, typename B>
FORCE_INLINE void tblas_tensor_gepow_inplace(A *a, B b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] = pow(a[i], b);
    }
}

template<typename A, typename B>
FORCE_INLINE void tblas_tensor_gepow_inplace(A *a, const B *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] = pow(a[i], b[i]);
    }
}

template<typename A, typename B, typename C>
FORCE_INLINE void tblas_tensor_gepow(const A *a, B b, C *c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = pow(a[i], b);
    }
}

template<typename A, typename B, typename C>
FORCE_INLINE void tblas_tensor_gepow(const A *a, const B *b, C *c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = pow(a[i], b[i]);
    }
}

template<typename A, typename B>
FORCE_INLINE void tblas_tensor_gepow_broadcast_inplace(A *a, const B *b, size_t n, size_t r) {
    for (size_t i = 0; i < n; i++) {
        size_t j = i % r;
        a[i] = pow(a[i], b[j]);
    }
}

template<typename A, typename B, typename C>
FORCE_INLINE void tblas_tensor_gepow_broadcast_right(const A *a, const B *b, C *c, size_t n, size_t r) {
    for (size_t i = 0; i < n; i++) {
        size_t j = i % r;
        c[i] = pow(a[i], b[j]);
    }
}

template<typename A, typename B, typename C>
FORCE_INLINE void tblas_tensor_gepow_broadcast_left(const A *a, const B *b, C *c, size_t n, size_t r) {
    for (size_t i = 0; i < n; i++) {
        size_t j = i % r;
        c[i] = pow(a[j], b[i]);
    }
}

BINARY_OP_FOR_ALL_TYPES_ALL_VARIANTS_IMPL(pow);