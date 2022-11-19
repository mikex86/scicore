#include "relu.h"

template<typename T>
void tblas_relu(const T *in, T *out, size_t nElements) {
    for (size_t i = 0; i < nElements; i++) {
        out[i] = in[i] > 0 ? in[i] : 0; // TODO: OPTIMIZE WITH SIMD
    }
}

UNARY_OPERATION_FOR_ALL_DATA_TYPES_IMPL(tblas_relu)


template<typename T>
void tblas_relu_gradients(const T *in, T *out, size_t nElements) {
    for (size_t i = 0; i < nElements; i++) {
        out[i] = in[i] > 0 ? 1 : 0;
    }
}

UNARY_OPERATION_FOR_ALL_DATA_TYPES_IMPL(tblas_relu_gradients)
