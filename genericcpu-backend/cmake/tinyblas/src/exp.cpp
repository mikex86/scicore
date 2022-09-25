#include "exp.h"
#include <cmath>
#include "forceinline.h"

template<typename A>
FORCE_INLINE void tblas_tensor_geexp_inplace(A *a, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] = exp(a[i]);
    }
}

template<typename A, typename B>
FORCE_INLINE void tblas_tensor_geexp(const A *a, B *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        b[i] = exp(a[i]);
    }
}

UNARY_OP_FOR_ALL_TYPES_ALL_VARIANTS_IMPL(exp)