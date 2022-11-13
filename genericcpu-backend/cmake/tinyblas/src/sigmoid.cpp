#include "sigmoid.h"

template<typename T>
void tblas_sigmoid(const T *in, T *out, size_t nElements) {
    // TODO: OPTIMIZE
    for (size_t i = 0; i < nElements; i++) {
        out[i] = 1 / (1 + exp(-in[i]));
    }
}

UNARY_OPERATION_FOR_ALL_DATA_TYPES_IMPL(tblas_sigmoid)