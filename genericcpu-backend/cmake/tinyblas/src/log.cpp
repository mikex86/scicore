#include "log.h"
#include <cmath>

template<typename T>
void tblas_log(const T *in, T *out, size_t nElements) {
    // TODO: OPTIMIZE
    for (size_t i = 0; i < nElements; i++) {
        out[i] = std::log(in[i]);
    }
}

UNARY_OPERATION_FOR_ALL_DATA_TYPES_IMPL(tblas_log);