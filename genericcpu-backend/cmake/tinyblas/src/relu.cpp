#include "relu.h"

#define FORCE_INLINE __attribute__((always_inline)) inline

template<typename A>
FORCE_INLINE void tblas_tensor_gerelu_inplace(A *a, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] = a[i] > 0 ? a[i] : 0;
    }
}

template<typename A, typename B>
FORCE_INLINE void tblas_tensor_gerelu(const A *a, B *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        b[i] = a[i] > 0 ? a[i] : 0;
    }
}

UNARY_OP_FOR_ALL_TYPES_ALL_VARIANTS_IMPL(relu)