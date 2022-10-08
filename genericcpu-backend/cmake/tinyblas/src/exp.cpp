#include "exp.h"
#include <cmath>

template<typename T>
void tblas_exp(const T *in, T *out, size_t nElements) {
    for (size_t i = 0; i < nElements; i++) {
        out[i] = std::exp(in[i]);
    }
}

UNARY_OPERATION_FOR_ALL_DATA_TYPES_IMPL(tblas_exp)